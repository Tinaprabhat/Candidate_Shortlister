"""
jd_parser.py — JD schema constants and validation for RedRob pipeline.

jd.json is pre-existing on disk; rank.py reads it directly.
This module exposes the schema definition, prompt template, and
validate_and_fill() so tests and downstream code can reference them.
"""

import logging

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
