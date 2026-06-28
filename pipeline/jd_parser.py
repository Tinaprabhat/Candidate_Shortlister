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
import time
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
    "what_you_will_do",
    "donts",
    "industry",
]

PROMPT_TEMPLATE = """You are an expert technical recruiter and hiring intelligence system.
Parse the job description below with extreme care. Return ONLY a valid JSON object — no preamble,
no markdown fences, no explanation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD-BY-FIELD INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

job_title
  The exact role title as written in the JD.

role
  One concise sentence naming what this person does and where they sit in the org.
  Example: "Senior AI Engineer owning the candidate-JD matching stack on a founding team."

industry
  The industry/vertical the company operates in.
  Examples: "HR Tech", "Fintech", "E-commerce", "HealthTech", "EdTech", "SaaS B2B".
  Derive from the company description if not stated explicitly.

location
  Primary work city and country: "City, Country" (e.g. "Bangalore, India").
  Write "Remote" if fully remote. Empty string if not mentioned.

experience_min
  Minimum years of relevant professional experience, as an integer.
  Parse from "5+ years", "minimum 5", "4-8 years" (→ 4). Use 0 if not stated.

experience_max
  Maximum years preferred, as an integer. From "4-8 years" (→ 8). Use 99 if not stated.

required_seniority
  One of: intern | junior | associate | mid | senior | lead | staff | principal | director

explicit_required  ← READ THE SKILLS SECTION VERY CAREFULLY
  Every technology, language, framework, tool, platform, methodology, or domain the JD
  directly states as required / must-have / essential.

  CRITICAL OUTPUT RULES — MUST FOLLOW:
  • Write SHORT skill keywords or tool names (1–4 words max). NEVER write context phrases.
    BAD:  "Production experience with embeddings-based retrieval systems"
    GOOD: "embeddings", "sentence-transformers", "faiss", "dense retrieval"
    BAD:  "Hands-on experience designing evaluation frameworks for ranking systems"
    GOOD: "learning to rank", "ndcg", "evaluation frameworks for ranking systems"
    BAD:  "Strong Python" or "Expert-level Python"
    GOOD: "python"
    BAD:  "Experience with vector databases or hybrid search infrastructure"
    GOOD: "vector databases", "hybrid search"
  • Do NOT summarise or group — list each skill item individually as its own entry.
  • If one JD bullet contains multiple skills joined by "or" / "and", list each separately.
  • Include individual tool names (FAISS, Pinecone, Weaviate, BM25, Qdrant) whenever
    the JD describes a category that implies them.
  • Include abbreviations AND their expansions when both appear.
  • Minimum 6 entries; extract every distinct skill the JD marks as required/must-have.

inferred_required
  Skills a strong candidate for THIS specific role would almost certainly have even if
  the JD never mentioned them. Reason from role + seniority + domain.
  Example: a "Senior ML Engineer building search pipelines" implies Git, Linux, REST APIs,
  Docker, basic SQL, experiment tracking (MLflow/W&B), system design fundamentals.
  Do NOT repeat items already in explicit_required.

explicit_bonus
  Skills the JD explicitly marks as nice-to-have / preferred / a plus / bonus.

inferred_bonus
  Adjacent skills typical for the role/seniority/industry that weren't mentioned.
  Do NOT repeat items from explicit_required or inferred_required.

semantic_neighbors
  For every skill in explicit_required, list 2-3 common alternative names or closely
  related terms that mean the same thing or would appear in a matching résumé.
  Format: {{ "<skill>": ["<alt1>", "<alt2>", "<alt3>"], ... }}
  Example: {{ "vector databases": ["FAISS", "Pinecone", "Weaviate", "Milvus"] }}

role_description
  2-4 sentence factual summary: what the role is, its place in the org, the core
  problem it solves. Quote or closely paraphrase the JD — do not invent.

company_description
  2-3 sentence summary of the company: what it does, stage/size if mentioned, domain.
  Extract from the JD. Leave empty string if absent.

what_you_will_do  ← DETAILED CANDIDATE ROLE
  Comprehensive prose paragraph (200-400 words) describing the full scope of the
  candidate's day-to-day work, ownership areas, key deliverables, and how success
  is measured. Use the JD's own language and specifics. This is the richest field
  and is used downstream for semantic matching — be thorough.

candidate_work
  JSON array of bullet-point strings: each concrete responsibility or deliverable the
  candidate will own. Minimum 6 items. Use the JD's own verb-first phrasing.
  Example: ["Own the ranking pipeline end-to-end.", "Write production Python daily."]

donts  ← WHAT THE COMPANY DOES NOT WANT
  JSON array of strings. Each entry is a specific type of candidate, background,
  skill-profile, or behaviour the JD explicitly or implicitly disqualifies.
  Derive from:
    - Explicit disqualifiers stated in the JD ("must not be a pure researcher")
    - Mismatches implied by the seniority + domain (e.g. a pure consulting background
      for a product-engineering role, or a candidate with only academic ML experience
      for a role requiring shipped production systems)
    - Anti-patterns implied by the role's requirements (e.g. if the JD emphasises speed
      and shipping, "someone who only writes papers and never ships code" is a dont)
  Be specific. Minimum 4 items. Examples:
    "Candidates with only academic/research experience and no shipped production systems"
    "Pure consulting or IT-services background (TCS, Infosys, Wipro) with no product ownership"
    "Overqualified researchers expecting publication-centric work"
    "Candidates who require full technical specs before working — this role demands autonomy"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT JSON SHAPE (strict)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "job_title": "<exact role title>",
  "role": "<one-sentence role summary>",
  "industry": "<industry vertical>",
  "location": "<City, Country | Remote | ''>",
  "experience_min": <int>,
  "experience_max": <int>,
  "required_seniority": "<level>",
  "explicit_required": ["<skill>", ...],
  "inferred_required": ["<skill>", ...],
  "explicit_bonus": ["<skill>", ...],
  "inferred_bonus": ["<skill>", ...],
  "semantic_neighbors": {{ "<skill>": ["<alt1>", "<alt2>"], ... }},
  "role_description": "<paragraph>",
  "company_description": "<paragraph>",
  "what_you_will_do": "<detailed prose 200-400 words>",
  "candidate_work": ["<bullet 1>", "<bullet 2>", ...],
  "donts": ["<dont 1>", "<dont 2>", ...]
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

    t = time.perf_counter()
    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    logger.info(f"[jd-io] extract_pdf_text: {pdf_path.name} ({len(text)} chars) in {round(time.perf_counter()-t, 3)}s")
    if not text.strip():
        raise ValueError(f"No text extracted from {pdf_path} (scanned PDF?)")
    return text.strip()


def extract_docx_text(docx_path: Path) -> str:
    """Extract text from a .docx file without extra dependencies."""
    if not zipfile.is_zipfile(docx_path):
        raise ValueError(f"Invalid .docx file: {docx_path}")

    t = time.perf_counter()
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
    logger.info(f"[jd-io] extract_docx_text: {docx_path.name} ({len(text)} chars) in {round(time.perf_counter()-t, 3)}s")
    if not text:
        raise ValueError(f"No text extracted from {docx_path}")
    return text


def extract_text_file(text_path: Path) -> str:
    """Extract text from plain-text job description files."""
    t = time.perf_counter()
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
    logger.info(f"[jd-io] extract_text_file: {text_path.name} ({len(text)} chars) in {round(time.perf_counter()-t, 3)}s")
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
    t_ollama = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    logger.info(f"[jd-llm] ollama subprocess elapsed: {round(time.perf_counter()-t_ollama, 3)}s")

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


def parse_jd_with_groq(jd_text: str, out_path: Path | None = None) -> dict:
    """Parse JD text via Groq API (primary path).

    Requires GROQ_API_KEY env var. Override model with GROQ_MODEL
    (default: llama-3.3-70b-versatile).
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq required: pip install groq")

    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client     = Groq(api_key=api_key)
    prompt     = PROMPT_TEMPLATE.format(jd_text=jd_text[:12000])

    logger.info(f"Calling Groq API ({model_name}) for JD parsing...")
    t_groq = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=4096,
    )
    logger.info(f"[jd-llm] groq API elapsed: {round(time.perf_counter()-t_groq, 3)}s")
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wrapped anyway
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    parsed = None
    for attempt in (raw, raw.replace("\n", " ")):
        try:
            parsed = json.loads(attempt)
            break
        except json.JSONDecodeError:
            pass
    if parsed is None:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Groq response was not a JSON object: {raw[:200]}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        logger.info(f"[groq] jd.json written to {out_path}")

    return parsed


def parse_jd_with_llm(jd_text: str, out_path: Path | None = None) -> dict:
    """Primary: Groq API (if GROQ_API_KEY is set). Fallback: local Ollama."""
    if os.environ.get("GROQ_API_KEY", "").strip():
        try:
            return parse_jd_with_groq(jd_text, out_path=out_path)
        except Exception as exc:
            logger.warning(f"Groq API failed ({exc}); falling back to local Ollama")
    else:
        logger.info("GROQ_API_KEY not set — using local Ollama")

    return parse_jd_with_ollama(jd_text, out_path=out_path)


def validate_and_fill(parsed: dict) -> dict:
    """Ensure all schema fields exist; fill missing with sensible defaults."""
    _list_fields = {
        "explicit_required", "inferred_required", "explicit_bonus", "inferred_bonus",
        "candidate_work", "donts",
    }
    _str_fields = {"role_description", "company_description", "what_you_will_do", "industry", "role"}

    for field in JD_SCHEMA_FIELDS:
        if field not in parsed:
            if field == "semantic_neighbors":
                parsed[field] = {}
            elif field in _str_fields:
                parsed[field] = ""
            else:
                parsed[field] = []

    parsed.setdefault("job_title", "")
    parsed.setdefault("role", "")
    parsed.setdefault("industry", "")
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

    # Dedupe skill lists (preserve original casing; dedupe by lowercase)
    for field in ["explicit_required", "inferred_required", "explicit_bonus", "inferred_bonus"]:
        seen: set = set()
        cleaned = []
        for s in parsed[field]:
            sl = str(s).strip().lower()
            if sl and sl not in seen:
                seen.add(sl)
                cleaned.append(str(s).strip())
        parsed[field] = cleaned

    # Lists that preserve original wording — just strip and drop blanks
    for field in ["candidate_work", "donts"]:
        parsed[field] = [
            str(item).strip() for item in parsed.get(field, []) if str(item).strip()
        ]

    # If what_you_will_do is missing, synthesise it from candidate_work
    if not parsed.get("what_you_will_do") and parsed.get("candidate_work"):
        parsed["what_you_will_do"] = " ".join(parsed["candidate_work"])

    return parsed


def parse_jd(source_path: Path, out_path: Path) -> dict:
    """Full pipeline: source file → text → LLM → validated jd.json on disk."""
    t_total = time.perf_counter()
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
    t_llm = time.perf_counter()
    parsed = parse_jd_with_llm(jd_text, out_path=out_path)
    logger.info(f"[jd-parse] LLM parse elapsed: {round(time.perf_counter()-t_llm, 3)}s")
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
    logger.info(f"  industry:           {parsed.get('industry', '')}")
    logger.info(f"  location:           {parsed.get('location', '')}")
    logger.info(f"  experience:         {parsed.get('experience_min', 0)}-{parsed.get('experience_max', 99)} yrs")
    logger.info(f"  explicit_required:  {len(parsed['explicit_required'])} skills")
    logger.info(f"  inferred_required:  {len(parsed['inferred_required'])} skills")
    logger.info(f"  what_you_will_do:   {len(parsed.get('what_you_will_do', ''))} chars")
    logger.info(f"  candidate_work:     {len(parsed['candidate_work'])} responsibilities")
    logger.info(f"  donts:              {len(parsed.get('donts', []))} items")
    logger.info(f"  role_description:   {len(parsed['role_description'])} chars")
    logger.info(f"  company_description:{len(parsed['company_description'])} chars")
    logger.info(f"[jd-parse] total parse_jd elapsed: {round(time.perf_counter()-t_total, 3)}s")
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
