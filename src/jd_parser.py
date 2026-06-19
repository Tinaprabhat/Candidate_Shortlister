"""
jd_parser.py — JD Parsing via local Ollama Mistral.

IMPORTANT — COMPETITION COMPLIANCE:
  This module parses the JD using a local Ollama model. It writes data/jd.json.
  rank.py reads that file from disk and makes NO API calls.

Usage (standalone, run once before ranking):
    python -m src.jd_parser --jd ./data/job_description.pdf --out ./data/jd.json
"""

import os
import re
import json
import argparse
import logging
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

# Expected jd.json schema
JD_SCHEMA_FIELDS = [
    "explicit_required",
    "inferred_required",
    "explicit_bonus",
    "inferred_bonus",
    "semantic_neighbors",
    "role_description",
    "company_description",
    "candidate_work",
]

PROMPT_TEMPLATE = """You are an expert technical recruiter parsing a job description.
Read the JD below and extract structured intent. Return ONLY valid JSON — no preamble, no markdown fences.

Rules:
- explicit_required: skills the JD directly states as required / must-have. List every
  technology, language, framework, tool, or domain explicitly mentioned.
- inferred_required: skills a strong candidate for THIS role would almost certainly
  have even if not written (reason from the role + seniority). E.g. "Senior Backend
  Engineer" implies REST APIs, system design, cloud (AWS/GCP), CI/CD.
- explicit_bonus: skills the JD explicitly marks nice-to-have / preferred / bonus.
- inferred_bonus: adjacent skills typical for the role but not mentioned.
- semantic_neighbors: for each required skill, 2-3 common alternative names/related
  terms (e.g. "Python": ["python3","Django","FastAPI"], "LLM": ["large language models","transformers"]).
- role_description: 2-4 sentence summary of what this role is, its place in the org,
  and the core problem it solves. Quote or closely paraphrase the JD — do not invent.
- company_description: 2-3 sentence summary of the company — what it does, its stage/size
  (if mentioned), and its domain. Extract from the JD; leave blank if absent.
- candidate_work: bullet list (as a JSON array of strings) of the concrete day-to-day
  responsibilities and deliverables the candidate will own. Use the JD's own language.
- location: the primary city AND country where the job is based, written as
  "City, Country" (e.g. "Bangalore, India", "London, UK", "New York, USA").
  Write "Remote" if the role is explicitly remote. Leave empty string if not mentioned.
- experience_min: minimum years of relevant professional experience required, as an
  integer. Read from phrases like "5+ years", "minimum 5 years", "4-8 years" (→ 4).
  Use 0 if not stated.
- experience_max: maximum years of experience preferred, as an integer. Read from
  ranges like "4-8 years" (→ 8) or "up to 10 years". Use 99 if not stated.

Output strictly this JSON shape:
{{
  "explicit_required": [...],
  "inferred_required": [...],
  "explicit_bonus": [...],
  "inferred_bonus": [...],
  "semantic_neighbors": {{ "<skill>": ["<alt1>", "<alt2>"], ... }},
  "job_title": "<the role title as written>",
  "required_seniority": "<one of: intern, junior, associate, mid, senior, lead, staff, principal, director>",
  "location": "<primary work city or region (e.g. 'Bangalore', 'Mumbai', 'Delhi'); 'Remote' if fully remote; empty string if not mentioned>",
  "experience_min": <minimum years of relevant experience required as an integer; 0 if not stated>,
  "experience_max": <maximum years of relevant experience preferred as an integer; 99 if not stated>,
  "role_description": "<paragraph>",
  "company_description": "<paragraph>",
  "candidate_work": ["<responsibility 1>", "<responsibility 2>", ...]
}}

JOB DESCRIPTION:
{jd_text}
"""


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from JD PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    if not text.strip():
        raise ValueError(f"No text extracted from {pdf_path} (scanned PDF?)")
    return text.strip()


def extract_docx_text(docx_path: Path) -> str:
    """Extract text from a .docx file without extra dependencies."""
    if not zipfile.is_zipfile(docx_path):
        raise ValueError(f"Invalid .docx file: {docx_path}")

    with zipfile.ZipFile(docx_path, "r") as zf:
        try:
            xml_bytes = zf.read("word/document.xml")
        except KeyError:
            raise ValueError(f"Invalid .docx file: missing word/document.xml in {docx_path}")

    root = ET.fromstring(xml_bytes)
    texts = []
    for node in root.iter():
        tag = node.tag
        if tag.endswith("}t"):
            texts.append(node.text or "")
        elif tag.endswith("}tab"):
            texts.append("\t")
        elif tag.endswith("}br") or tag.endswith("}cr") or tag.endswith("}p"):
            texts.append("\n")
    text = "".join(texts).strip()
    if not text:
        raise ValueError(f"No text extracted from {docx_path}")
    return text


def extract_text_file(text_path: Path) -> str:
    """Extract text from plain-text job description files."""
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = text_path.read_text(encoding=encoding)
            break
        except UnicodeError:
            continue
    else:
        raise UnicodeError(f"Unable to decode {text_path} as utf-8, utf-16, or latin-1")

    if not text.strip():
        raise ValueError(f"No text extracted from {text_path}")
    return text.strip()


_CSI_RE = re.compile(r'\x1b\[([0-9;?]*)([A-Za-z])')


def _render_ansi(text: str) -> str:
    """Simulate a terminal to produce clean text from ANSI-escape-laden output.

    Ollama streams its response live into the terminal using cursor-movement
    sequences (e.g. ESC[2D ESC[K = "go back 2 cols, erase to EOL") to rewrite
    characters mid-stream. When captured by subprocess, those sequences land
    verbatim in stdout and corrupt the JSON. We replay them against a list
    buffer so the final content matches what a real terminal would display.
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != '\x1b':
            result.append(ch)
            i += 1
            continue
        # ESC — try to parse a CSI sequence (ESC [ params final)
        m = _CSI_RE.match(text, i)
        if m:
            params_str, final = m.group(1), m.group(2)
            i = m.end()
            if final == 'D':          # cursor back N cols → erase last N chars
                try:
                    n = int(params_str or '1')
                except ValueError:
                    n = 1
                if n > 0:
                    del result[-n:]
            # K (erase to EOL), G (cursor column), h/l (mode set/reset), etc. → ignore
        else:
            # Non-CSI escape or bare ESC — skip ESC + next char
            i += 2 if i + 1 < len(text) else 1
    return ''.join(result)


def _parse_ollama_stdout(stdout: str) -> dict:
    """Extract a JSON dict from Ollama's raw stdout.

    Applies terminal simulation first (strips ANSI cursor-movement sequences),
    then tries progressively looser parse strategies.
    """
    # Pre-process: replay ANSI sequences so we get clean text
    clean = _render_ansi(stdout)

    for candidate in (clean, stdout):           # prefer clean; fall back to raw
        text = candidate.strip()

        # Strategy 1: direct parse
        try:
            r = json.loads(text)
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass

        # Strategy 2: collapse literal newlines (Ollama word-wraps inside strings)
        flat = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        try:
            r = json.loads(flat)
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass

        # Strategy 3: trim preamble/postamble, keep outermost { … }
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end > start:
            block = text[start:end + 1]
            try:
                r = json.loads(block)
                if isinstance(r, dict):
                    return r
            except json.JSONDecodeError:
                pass
            # Strategy 4: outermost block + collapsed newlines
            try:
                r = json.loads(block.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' '))
                if isinstance(r, dict):
                    return r
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("Could not extract valid JSON dict from Ollama output", stdout, 0)


def parse_jd_with_ollama(jd_text: str, out_path: Path | None = None) -> dict:
    """Parse JD text with a local Ollama model and write jd.json immediately."""
    if shutil.which("ollama") is None:
        raise EnvironmentError(
            "Ollama CLI not found. Install Ollama and ensure the local model is available."
        )

    model_name = os.environ.get("OLLAMA_MODEL", "mistral:latest")
    prompt = PROMPT_TEMPLATE.format(jd_text=jd_text[:12000])
    cmd = ["ollama", "run", model_name, "--format", "json"]

    logger.info(f"Calling local Ollama model {model_name} for JD parsing...")
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    # ── Diagnostic: show exactly where output landed ───────────────────────────
    logger.info(f"[ollama] returncode={proc.returncode}")
    logger.info(f"[ollama] stdout bytes={len(proc.stdout)}  stderr bytes={len(proc.stderr)}")
    logger.info(f"[ollama] stdout preview: {repr(proc.stdout[:400])}")
    if proc.stderr.strip():
        logger.info(f"[ollama] stderr preview: {repr(proc.stderr[:400])}")

    if proc.returncode != 0:
        logger.error(f"Ollama command failed: {proc.stderr.strip()}")
        raise RuntimeError(f"Ollama fallback failed with exit code {proc.returncode}")

    # ── Step 1: write raw stdout directly to jd.json right now ────────────────
    # The file exists on disk even if the parse step below fails.
    raw_stdout = proc.stdout
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(raw_stdout, encoding="utf-8")
        logger.info(f"[ollama] Raw stdout written to {out_path}")

    # ── Step 2: if stdout is empty, try stderr (some Ollama builds differ) ─────
    raw_source = raw_stdout
    if not raw_source.strip():
        logger.warning("[ollama] stdout is empty — attempting stderr as fallback")
        raw_source = proc.stderr

    # ── Step 3: parse ──────────────────────────────────────────────────────────
    try:
        container = _parse_ollama_stdout(raw_source)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Ollama output as JSON after all strategies")
        logger.error(f"stdout repr: {repr(raw_stdout[:800])}")
        logger.error(f"stderr repr: {repr(proc.stderr[:400])}")
        raise RuntimeError("Failed to parse Ollama output") from e

    # ── Step 4: handle Ollama envelope vs. direct JD JSON ─────────────────────
    _JD_KEYS = {"job_title", "explicit_required", "inferred_required"}
    if isinstance(container, dict) and _JD_KEYS & container.keys():
        # Direct JD JSON — overwrite the raw file with pretty-printed version
        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(container, f, indent=2, ensure_ascii=False)
            logger.info(f"[ollama] Parsed jd.json written to {out_path}")
        return container

    # Ollama envelope: dig out the nested text blob
    def _extract_text(obj):
        if isinstance(obj, dict):
            if "assistant" in obj and isinstance(obj["assistant"], dict):
                result = obj["assistant"].get("result")
                if isinstance(result, dict) and "text" in result:
                    return result["text"]
            if "content" in obj:
                return obj["content"]
            for value in obj.values():
                t = _extract_text(value)
                if t:
                    return t
        elif isinstance(obj, list):
            for item in obj:
                t = _extract_text(item)
                if t:
                    return t
        return ""

    raw_text = _extract_text(container).strip().replace("```json", "").replace("```", "").strip()
    if not raw_text:
        logger.error("[ollama] No text blob found inside Ollama envelope")
        logger.error(f"container keys: {list(container.keys()) if isinstance(container, dict) else type(container)}")
        raise RuntimeError("Failed to parse Ollama output: unexpected envelope structure")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Last resort: collapse newlines inside the text blob and retry
        parsed = json.loads(raw_text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' '))

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        logger.info(f"[ollama] Parsed jd.json written to {out_path}")
    return parsed


def normalize_gemini_model_name(model_name: str) -> str:
    """Normalize a Gemini model name to the supported API format."""
    if not model_name:
        return model_name
    model_name = model_name.strip()
    if not model_name.startswith("models/"):
        return f"models/{model_name}"
    return model_name


def parse_jd_with_api(jd_text: str, out_path: Path | None = None) -> dict:
    """Parse JD text via Gemini API (primary path).

    Requires GEMINI_API_KEY env var.  Override model with GEMINI_MODEL
    (default: gemini-flash-latest).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai required: pip install google-generativeai")

    genai.configure(api_key=api_key)
    model_name = normalize_gemini_model_name(os.environ.get("GEMINI_MODEL", "gemini-flash-latest"))

    prompt = PROMPT_TEMPLATE.format(jd_text=jd_text[:12000])
    logger.info(f"Calling Gemini API ({model_name}) for JD parsing...")

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            ),
        )
    except Exception as first_exc:
        logger.warning(f"Gemini API generation failed for {model_name}: {first_exc}")
        fallback_model = "models/gemini-flash-latest"
        if model_name != fallback_model:
            try:
                logger.info(f"Retrying Gemini API with fallback model {fallback_model}")
                model = genai.GenerativeModel(fallback_model)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                    ),
                )
            except Exception:
                logger.warning("Fallback Gemini model also failed; retrying without response_mime_type")
                response = model.generate_content(prompt)
        else:
            logger.warning("Retrying Gemini API without response_mime_type")
            response = model.generate_content(prompt)

    raw = response.text.strip()

    # Strip markdown fences if the model wrapped the output anyway
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    # Try direct parse, then outermost-brace extraction
    parsed = None
    for candidate in (raw, raw.replace("\n", " ")):
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            pass
    if parsed is None:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Gemini response was not a JSON object: {raw[:200]}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        logger.info(f"[api] jd.json written to {out_path}")

    return parsed


def parse_jd_with_llm(jd_text: str, out_path: Path | None = None) -> dict:
    """Primary: Gemini API (if GEMINI_API_KEY is set). Fallback: local Ollama."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            return parse_jd_with_api(jd_text, out_path=out_path)
        except Exception as exc:
            logger.warning(f"Gemini API failed ({exc}); falling back to local Ollama")
    else:
        logger.info("GEMINI_API_KEY not set — using local Ollama")

    return parse_jd_with_ollama(jd_text, out_path=out_path)


def validate_and_fill(parsed: dict) -> dict:
    """Ensure all schema fields exist; fill missing with sensible defaults."""
    _list_fields = {"explicit_required", "inferred_required", "explicit_bonus", "inferred_bonus", "candidate_work"}
    _str_fields = {"role_description", "company_description"}

    for field in JD_SCHEMA_FIELDS:
        if field not in parsed:
            if field == "semantic_neighbors":
                parsed[field] = {}
            elif field in _str_fields:
                parsed[field] = ""
            else:
                parsed[field] = []

    parsed.setdefault("job_title", "")
    parsed.setdefault("required_seniority", "mid")
    parsed.setdefault("location", "")

    # Normalize experience bounds to integers
    try:
        parsed["experience_min"] = int(parsed.get("experience_min") or 0)
    except (ValueError, TypeError):
        parsed["experience_min"] = 0
    try:
        parsed["experience_max"] = int(parsed.get("experience_max") or 99)
    except (ValueError, TypeError):
        parsed["experience_max"] = 99

    # Normalize: lowercase + dedupe skill lists (not candidate_work — preserve wording)
    for field in ["explicit_required", "inferred_required", "explicit_bonus", "inferred_bonus"]:
        seen = set()
        cleaned = []
        for s in parsed[field]:
            sl = str(s).strip().lower()
            if sl and sl not in seen:
                seen.add(sl)
                cleaned.append(str(s).strip())
        parsed[field] = cleaned

    # candidate_work: ensure it's a list of non-empty strings
    parsed["candidate_work"] = [
        str(item).strip() for item in parsed.get("candidate_work", []) if str(item).strip()
    ]

    return parsed


def parse_jd(source_path: Path, out_path: Path) -> dict:
    """Full pipeline: source file → text → LLM → validated jd.json on disk."""
    logger.info(f"Extracting text from {source_path}")
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        jd_text = extract_pdf_text(source_path)
    elif suffix == ".docx":
        jd_text = extract_docx_text(source_path)
    elif suffix in {".txt", ".md"}:
        jd_text = extract_text_file(source_path)
    else:
        raise ValueError(
            "Unsupported job description format. Use .pdf, .docx, .txt, or .md"
        )

    # Pass out_path so Ollama writes jd.json immediately after parsing
    parsed = parse_jd_with_llm(jd_text, out_path=out_path)
    parsed = validate_and_fill(parsed)

    # Always write the validated, normalised version to the caller's out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    # Also persist a canonical copy at data/jd.json so rank.py can read it
    # without needing to know where the caller stored the file.
    try:
        from . import constants as _C
        canonical = _C.JD_JSON_PATH
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if canonical.resolve() != out_path.resolve():
            with open(canonical, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            logger.info(f"Also wrote canonical jd.json → {canonical}")
    except Exception as _e:
        logger.debug(f"Skipped canonical jd.json copy: {_e}")

    logger.info(f"Wrote jd.json → {out_path}")
    logger.info(f"  job_title:          {parsed.get('job_title', '')}")
    logger.info(f"  location:           {parsed.get('location', '')}")
    logger.info(f"  experience:         {parsed.get('experience_min', 0)}-{parsed.get('experience_max', 99)} yrs")
    logger.info(f"  explicit_required:  {len(parsed['explicit_required'])} skills")
    logger.info(f"  inferred_required:  {len(parsed['inferred_required'])} skills")
    logger.info(f"  candidate_work:     {len(parsed['candidate_work'])} responsibilities")
    logger.info(f"  role_description:   {len(parsed['role_description'])} chars")
    logger.info(f"  company_description:{len(parsed['company_description'])} chars")
    return parsed


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Parse JD into jd.json using local Ollama")
    ap.add_argument("--jd", required=True, type=Path, help="Path to JD PDF")
    ap.add_argument("--out", type=Path, default=Path("./data/jd.json"), help="Output jd.json path")
    args = ap.parse_args()

    # Load .env if present
    def load_env_file(path: Path) -> dict:
        text = None
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeError:
                continue
        if text is None:
            raise UnicodeError(f"Unable to decode {path} as UTF-8, UTF-16, or Latin-1")

        env_values = {}
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_values[k.strip()] = v.strip()
        return env_values

    env_file = Path(".env")
    if env_file.exists():
        for k, v in load_env_file(env_file).items():
            os.environ.setdefault(k, v)

    parse_jd(args.jd, args.out)
    print(f"✅ jd.json written to {args.out}")


if __name__ == "__main__":
    main()
