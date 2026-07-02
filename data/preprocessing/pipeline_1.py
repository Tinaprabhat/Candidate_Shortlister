# pipeline_1.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Single-pass pipeline.
#   Reads candidates.jsonl → cleans (fixes + flags) → stamps fabrication
#   → classifies into 5-dimension tree → writes Dataset/ folder tree
#
# TREE STRUCTURE
#   Dataset/
#   └── {code_status}        code | no_code
#       └── {availability}   available | not_available
#           └── {experience} 0_to_3 | 4_to_9 | 10_plus
#               └── {domain}  engineering | devops_and_cloud |
#                              product_and_design | operations |
#                              business | marketing | finance |
#                              hr_and_people | non_tech_engineering | other
#                   └── {role}.json
#
# CODE vs NO_CODE LOGIC:
#   code    = current title IS a hands-on technical role that requires
#             writing production code right now.
#   no_code = everything else:
#             - management/leadership drift (tech lead, architect, EM)
#             - non-technical roles (PM, BA, HR, marketing, finance, etc.)
#             - zero career history (pure academic)
#             - entire career in pure research/academic titles
#
# COMPUTER VISION SPECIAL RULE (from JD):
#   "People whose primary expertise is computer vision, speech, or robotics
#    without significant NLP/IR exposure — we respect your work but you'd be
#    re-learning fundamentals here."
#
#   A candidate routes to computer_vision_engineer.json ONLY IF:
#     - Current title matches CV keywords AND
#     - They do NOT have significant retrieval/NLP/IR skills
#       (embeddings, vector DBs, sentence transformers, FAISS, Pinecone,
#        Qdrant, Weaviate, Milvus, OpenSearch, NLP, semantic search,
#        information retrieval, learning to rank, BM25, RAG, LLMs,
#        fine-tuning LLMs, haystack, pgvector)
#
#   If they DO have those skills → they are NOT pure CV →
#   route to ml_engineer.json instead (they cross the JD's threshold)
#
# USAGE
#   python pipeline_1.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import time
import hashlib
from datetime import date, datetime
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "candidates.jsonl"
DATASET_DIR = "Dataset"


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

POSTGRAD_DEGREES   = {"m.tech", "m.e.", "m.s.", "m.sc", "m.sc.", "ph.d", "mba", "m.b.a"}
UNDERGRAD_DEGREES  = {"b.tech", "b.e.", "b.sc", "b.sc.", "b.a.", "be", "btech", "b.eng"}
ENG_DEGREE_SET     = {"b.tech", "m.tech", "b.e.", "m.e.", "be", "btech", "mtech", "b.eng"}
INVALID_ENG_FIELDS = {"mba", "m.b.a", "commerce"}

AI_SKILLS = {
    "machine learning", "deep learning", "nlp", "computer vision",
    "object detection", "image classification", "rag", "llms",
    "fine-tuning llms", "vector search", "embeddings", "faiss",
    "pinecone", "qdrant", "weaviate", "sentence transformers",
    "hugging face transformers", "langchain", "llamaindex",
    "diffusion models", "gans", "yolo", "cnn", "rnn", "lstm",
    "transformers", "bert", "gpt", "reinforcement learning",
    "recommendation systems", "information retrieval",
    "semantic search", "prompt engineering", "mlops", "mlflow",
    "weights & biases", "kubeflow", "bentoml",
    "speech recognition", "tts", "asr", "forecasting",
    "time series", "statistical modeling", "learning to rank",
    "bm25", "opensearch", "haystack", "qlora", "pgvector",
    "feature engineering", "data science", "opencv",
    "scikit-learn", "xgboost", "lightgbm", "prophet",
    "tensorflow", "pytorch", "keras"
}

AI_CAREER_KEYWORDS = {
    "ml engineer", "machine learning", "data scientist", "ai specialist",
    "ai engineer", "nlp engineer", "computer vision", "research engineer",
    "deep learning", "junior ml", "senior ml", "ai research",
    "applied scientist", "mlops engineer"
}

AI_INDUSTRIES_EXACT = {
    "ai/ml", "ai", "ml", "machine learning", "artificial intelligence",
    "edtech ai", "healthtech ai", "deep learning", "nlp"
}

# _SCORE_FIELDS (unused — was fl26_honeypot_score's input; honeypot scoring removed)
# _SCORE_FIELDS = [
#     "skill_career_domain_mismatch", "education_overlap",
#     "reverse_degree_order", "second_undergrad_after_first",
#     "education_career_gap_flag", "active_before_signup",
#     "duplicate_job_descriptions",
#     "low_engagement_flag", "_any_invalid_degree_field",
# ]

# ── Titles that ARE hands-on production coding roles ─────────────────────────
# code = current title is in this set
CODE_TITLES = {
    # Software Engineering
    "software engineer", "software developer", "sde", "sde-",
    "backend engineer", "backend developer",
    "frontend engineer", "frontend developer",
    "full stack", "fullstack",
    "mobile developer", "android developer", "ios developer",
    "mobile engineer", "android engineer", "ios engineer",
    "java developer", "java engineer",
    ".net developer", ".net engineer",
    "embedded engineer", "firmware engineer",
    "qa engineer", "quality assurance engineer", "test engineer",
    "automation engineer", "sdet",
    # Data & AI
    "ml engineer", "machine learning engineer",
    "data scientist", "applied scientist",
    "ai engineer", "ai specialist",
    "nlp engineer",
    "computer vision engineer",
    "deep learning engineer",
    "research engineer",
    "mlops engineer",
    "data engineer", "analytics engineer",
    "data architect",
    "etl developer", "spark engineer", "kafka engineer",
    "databricks engineer",
    "data analyst",
    # DevOps & Cloud (still write code / IaC)
    "devops engineer",
    "cloud engineer",
    "platform engineer",
    "site reliability engineer", "sre",
    "systems engineer",
    "security engineer",
    # Senior IC titles that still write code
    "senior software engineer", "senior data engineer",
    "senior ml engineer", "senior backend engineer",
    "senior frontend engineer", "principal engineer",
    "staff engineer",
}

# ── Pure research titles — entire career these → no_code ─────────────────────
PURE_RESEARCH_TITLES = {
    "research scientist", "postdoc", "postdoctoral",
    "professor", "assistant professor", "associate professor",
    "phd researcher", "doctoral researcher", "research fellow",
    "ai researcher", "ml researcher", "research intern",
}

# ── Retrieval/NLP/IR skills from JD — if CV engineer has these,
#    they cross the JD threshold and are NOT pure CV ─────────────────────────
RETRIEVAL_NLP_SKILLS = {
    # Embeddings & retrieval
    "embeddings", "sentence transformers", "faiss", "pinecone", "qdrant",
    "weaviate", "milvus", "opensearch", "elasticsearch", "pgvector",
    "vector search", "vector databases", "hybrid search",
    # NLP & IR
    "nlp", "information retrieval", "semantic search", "learning to rank",
    "bm25", "haystack", "rag", "llms", "fine-tuning llms",
    "hugging face transformers", "langchain", "llamaindex", "qlora",
    # Ranking evaluation signals
    "ndcg", "mrr", "map", "a/b test", "retrieval quality",
}

# ── CV title keywords ─────────────────────────────────────────────────────────
CV_TITLE_KEYWORDS = {
    "computer vision engineer", "computer vision",
    "vision engineer", "image recognition engineer",
}


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN + ROLE TAXONOMY
#  tech + data_and_ai merged into single "engineering" domain
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_ROLE_RULES = [

    # ── ENGINEERING (tech + data_and_ai merged) ───────────────────────────────
    ("engineering", "ml_engineer",
        ["ml engineer", "machine learning engineer", "junior ml", "senior ml"]),
    ("engineering", "data_scientist",
        ["data scientist", "applied scientist"]),
    ("engineering", "ai_engineer",
        ["ai engineer", "ai specialist", "ai research"]),
    ("engineering", "nlp_engineer",
        ["nlp engineer"]),
    # computer_vision_engineer has special routing logic — see _domain_and_role()
    ("engineering", "computer_vision_engineer",
        ["computer vision engineer", "computer vision"]),
    ("engineering", "deep_learning_engineer",
        ["deep learning engineer"]),
    ("engineering", "research_engineer",
        ["research engineer", "research scientist"]),
    ("engineering", "mlops_engineer",
        ["mlops engineer", "mlops"]),
    ("engineering", "data_engineer",
        ["data engineer", "analytics engineer"]),
    ("engineering", "data_architect",
        ["data architect"]),
    ("engineering", "etl_developer",
        ["etl developer", "spark engineer", "kafka engineer",
         "databricks engineer"]),
    ("engineering", "data_analyst",
        ["data analyst", "business intelligence", "bi analyst"]),
    ("engineering", "backend_engineer",
        ["backend engineer", "backend developer"]),
    ("engineering", "frontend_engineer",
        ["frontend engineer", "frontend developer"]),
    ("engineering", "full_stack_developer",
        ["full stack", "fullstack"]),
    ("engineering", "mobile_developer",
        ["mobile developer", "android developer", "ios developer",
         "mobile engineer", "android engineer", "ios engineer"]),
    ("engineering", "software_engineer",
        ["software engineer", "software developer", "sde"]),
    ("engineering", "qa_engineer",
        ["qa engineer", "quality assurance", "test engineer",
         "automation engineer", "sdet"]),
    ("engineering", "java_developer",
        ["java developer", "java engineer"]),
    ("engineering", "net_developer",
        [".net developer", ".net engineer"]),
    ("engineering", "embedded_engineer",
        ["embedded engineer", "firmware engineer"]),

    # ── DEVOPS & CLOUD ────────────────────────────────────────────────────────
    ("devops_and_cloud", "devops_engineer",        ["devops engineer"]),
    ("devops_and_cloud", "cloud_engineer",          ["cloud engineer"]),
    ("devops_and_cloud", "platform_engineer",       ["platform engineer"]),
    ("devops_and_cloud", "site_reliability_engineer",["site reliability", "sre"]),
    ("devops_and_cloud", "systems_engineer",        ["systems engineer"]),
    ("devops_and_cloud", "security_engineer",       ["security engineer", "cybersecurity"]),

    # ── PRODUCT & DESIGN ──────────────────────────────────────────────────────
    ("product_and_design", "product_manager",
        ["product manager", "associate product manager", "senior product manager"]),
    ("product_and_design", "product_designer",  ["product designer"]),
    ("product_and_design", "ux_designer",        ["ux designer", "user experience", "ux researcher"]),
    ("product_and_design", "ui_designer",        ["ui designer", "user interface"]),
    ("product_and_design", "graphic_designer",   ["graphic designer"]),
    ("product_and_design", "brand_designer",     ["brand designer", "creative director", "visual designer"]),

    # ── OPERATIONS ────────────────────────────────────────────────────────────
    ("operations", "operations_manager",  ["operations manager", "ops manager"]),
    ("operations", "supply_chain_manager",["supply chain", "procurement manager", "logistics manager"]),
    ("operations", "project_manager",     ["project manager"]),
    ("operations", "program_manager",     ["program manager"]),
    ("operations", "scrum_master",        ["scrum master", "agile coach"]),
    ("operations", "customer_support",    ["customer support", "customer success", "customer service"]),

    # ── BUSINESS ──────────────────────────────────────────────────────────────
    ("business", "business_analyst",  ["business analyst"]),
    ("business", "sales_executive",
        ["sales executive", "sales manager", "account executive",
         "account manager", "business development"]),
    ("business", "consultant",
        ["consultant", "strategy consultant", "management consultant"]),

    # ── MARKETING ─────────────────────────────────────────────────────────────
    ("marketing", "marketing_manager",    ["marketing manager"]),
    ("marketing", "digital_marketing",    ["digital marketing", "performance marketing", "growth marketer"]),
    ("marketing", "seo_specialist",       ["seo specialist", "seo manager"]),
    ("marketing", "content_writer",       ["content writer", "content strategist", "copywriter", "technical writer"]),
    ("marketing", "social_media_manager", ["social media manager", "social media"]),

    # ── FINANCE ───────────────────────────────────────────────────────────────
    ("finance", "finance_manager",   ["finance manager", "financial controller", "cfo"]),
    ("finance", "financial_analyst", ["financial analyst", "investment analyst"]),
    ("finance", "accountant",        ["accountant", "senior accountant"]),
    ("finance", "tax_consultant",    ["tax consultant", "tax manager"]),
    ("finance", "auditor",           ["auditor", "internal auditor"]),

    # ── HR & PEOPLE ───────────────────────────────────────────────────────────
    ("hr_and_people", "hr_manager",
        ["hr manager", "human resources manager", "people manager"]),
    ("hr_and_people", "recruiter",
        ["recruiter", "talent acquisition", "technical recruiter"]),
    ("hr_and_people", "learning_and_development",
        ["learning and development", "l&d", "training manager"]),
    ("hr_and_people", "hr_business_partner",
        ["hr business partner", "hrbp"]),

    # ── NON-TECH ENGINEERING ──────────────────────────────────────────────────
    ("non_tech_engineering", "mechanical_engineer",    ["mechanical engineer"]),
    ("non_tech_engineering", "civil_engineer",          ["civil engineer"]),
    ("non_tech_engineering", "electrical_engineer",     ["electrical engineer"]),
    ("non_tech_engineering", "chemical_engineer",       ["chemical engineer"]),
    ("non_tech_engineering", "structural_engineer",     ["structural engineer"]),
    ("non_tech_engineering", "manufacturing_engineer",  ["manufacturing engineer"]),
    ("non_tech_engineering", "quality_engineer",        ["quality engineer", "quality manager"]),
    ("non_tech_engineering", "biomedical_engineer",     ["biomedical engineer"]),
]

OTHER_DOMAIN    = "other"
OTHER_ROLE_SLUG = "unclassified"


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _s(val) -> str:
    return (val or "").lower().strip()

def _parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None

def _has_ai_career(career: list) -> bool:
    for job in career:
        title = _s(job.get("title"))
        ind   = _s(job.get("industry"))
        if any(kw in title for kw in AI_CAREER_KEYWORDS):
            return True
        if ind in AI_INDUSTRIES_EXACT:
            return True
        if "machine learning" in ind or "artificial intelligence" in ind:
            return True
    return False

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  FIXES
# ══════════════════════════════════════════════════════════════════════════════

def _fix_salary(sig):
    sal = sig.setdefault("expected_salary_range_inr_lpa", {})
    mn, mx = sal.get("min"), sal.get("max")
    if mn is not None and mx is not None and mn > mx:
        sal["min"], sal["max"] = mx, mn
        sal["salary_was_inverted"] = True
    else:
        sal["salary_was_inverted"] = False

def _fix_github(sig):
    score = sig.get("github_activity_score")
    if score == -1 or score is None:
        sig["github_activity_score"] = None
        sig["github_not_linked"]     = True
    else:
        sig["github_not_linked"] = False

def _fix_offer(sig):
    if sig.get("offer_acceptance_rate") == -1:
        sig["offer_acceptance_rate"] = None
        sig["no_offer_history"]      = True
    else:
        sig["no_offer_history"] = False

def apply_fixes(c):
    sig = c.setdefault("redrob_signals", {})
    _fix_salary(sig)
    _fix_github(sig)
    _fix_offer(sig)
    sal = sig.get("expected_salary_range_inr_lpa", {})
    c["salary_was_inverted"] = sal.get("salary_was_inverted", False)
    c["github_not_linked"]   = sig.get("github_not_linked",   False)
    c["no_offer_history"]    = sig.get("no_offer_history",    False)


# ══════════════════════════════════════════════════════════════════════════════
#  FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def fl3_skill_career_domain_mismatch(c):
    skills   = c.get("skills", [])
    career   = c.get("career_history", [])
    ai_count = sum(1 for sk in skills if _s(sk.get("name")) in AI_SKILLS)
    if ai_count < 3:
        return False
    return not _has_ai_career(career)

def fl5_education_overlap(edu_list):
    pairs = []
    for i in range(len(edu_list)):
        for j in range(i + 1, len(edu_list)):
            s1 = edu_list[i].get("start_year")
            e1 = edu_list[i].get("end_year")
            s2 = edu_list[j].get("start_year")
            e2 = edu_list[j].get("end_year")
            if None in (s1, e1, s2, e2): continue
            if max(s1, s2) < min(e1, e2):
                pairs.append([i, j])
    return bool(pairs), pairs

def fl6_reverse_degree_order(edu_list):
    pg_ends, ug_starts = [], []
    for e in edu_list:
        d  = _s(e.get("degree"))
        ey = e.get("end_year")
        sy = e.get("start_year")
        if d in POSTGRAD_DEGREES  and ey is not None: pg_ends.append(ey)
        if d in UNDERGRAD_DEGREES and sy is not None: ug_starts.append(sy)
    if not pg_ends or not ug_starts: return False
    return min(pg_ends) <= max(ug_starts)

def fl7_second_undergrad_after_first(edu_list):
    ugs = sorted(
        [e for e in edu_list
         if _s(e.get("degree")) in UNDERGRAD_DEGREES
         and e.get("start_year") is not None
         and e.get("end_year")   is not None],
        key=lambda e: e["start_year"]
    )
    if len(ugs) < 2: return False
    return ugs[1]["start_year"] >= ugs[0]["end_year"] + 2

def fl8_education_career_gap(c):
    edu_ends = [
        e["end_year"] for e in c.get("education", [])
        if e.get("end_year") is not None
    ]
    career_starts = [
        _parse_date(j.get("start_date")).year
        for j in c.get("career_history", [])
        if _parse_date(j.get("start_date"))
    ]
    if not edu_ends or not career_starts: return False, 0.0
    gap = float(min(career_starts) - max(edu_ends))
    return gap > 1, round(gap, 1)

def fl11_active_before_signup(c):
    sig    = c.get("redrob_signals", {})
    signup = _parse_date(sig.get("signup_date"))
    last   = _parse_date(sig.get("last_active_date"))
    if signup is None or last is None: return False, 0
    if last < signup: return True, (signup - last).days
    return False, 0

def fl16_invalid_degree_field(edu):
    deg   = _s(edu.get("degree"))
    field = _s(edu.get("field_of_study"))
    if deg in ENG_DEGREE_SET and field in INVALID_ENG_FIELDS: return True
    if deg in {"m.sc", "m.sc.", "m.s.", "m.s"} and field in INVALID_ENG_FIELDS: return True
    return False

def fl18_duplicate_descriptions(career):
    """
    FL18 — Within-candidate duplicate description check (single flag).
    True if ANY two (or more) of this candidate's OWN career descriptions
    are character-for-character identical — whether it's just 2 matching,
    or all of them matching.

    Replaces the old FL18/FL19 split (any-duplicate vs all-identical)
    with one unified flag: duplicate_job_descriptions.
    """
    descs = [(j.get("description") or "").strip() for j in career]
    non_empty = [d for d in descs if d]
    if len(non_empty) < 2:
        return False
    # True if there are fewer unique descriptions than total descriptions
    # i.e. at least one description repeats within this candidate
    return len(set(non_empty)) < len(non_empty)

def fl20_low_engagement(c):
    sig  = c.get("redrob_signals", {})
    rate = sig.get("recruiter_response_rate")
    hrs  = sig.get("avg_response_time_hours")
    if rate is None or hrs is None: return False
    return rate < 0.10 and hrs > 200

# fl25_possible_honeypot / fl26_honeypot_score — REMOVED.
# possible_honeypot boolean is no longer computed (raw structural-flag count
# was too noisy); honeypot_score value retained nowhere since fl26 depended
# on it. L1a's possible_honeypot hard-reject check in layers.py is now
# permanently inert (field is never set) as a direct consequence.


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER CLEAN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def process_candidate(c):
    edu_list = c.get("education", [])
    career   = c.get("career_history", [])

    apply_fixes(c)

    c["skill_career_domain_mismatch"] = fl3_skill_career_domain_mismatch(c)

    ov_flag, ov_pairs = fl5_education_overlap(edu_list)
    c["education_overlap"]             = ov_flag
    c["overlapping_education_indices"] = ov_pairs

    c["reverse_degree_order"]         = fl6_reverse_degree_order(edu_list)
    c["second_undergrad_after_first"] = fl7_second_undergrad_after_first(edu_list)

    gap_flag, gap_val = fl8_education_career_gap(c)
    c["education_career_gap_flag"]    = gap_flag
    c["education_career_gap_years"]   = gap_val

    abs_flag, abs_days = fl11_active_before_signup(c)
    c["active_before_signup"]         = abs_flag
    c["signup_active_gap_days"]       = abs_days

    c["duplicate_job_descriptions"] = fl18_duplicate_descriptions(career)

    c["low_engagement_flag"] = fl20_low_engagement(c)

    any_invalid = False
    for edu in edu_list:
        val = fl16_invalid_degree_field(edu)
        edu["invalid_degree_field_combination"] = val
        if val: any_invalid = True
    c["_any_invalid_degree_field"] = any_invalid

    # fabrication_bandwidth — stamped in Pass 2 after full freq table + max are built
    c["fabrication_bandwidth"] = 0.0

    return c


# ══════════════════════════════════════════════════════════════════════════════
#  DESCRIPTION FREQUENCY
# ══════════════════════════════════════════════════════════════════════════════

def collect_desc_freq(freq, c):
    for job in c.get("career_history", []):
        desc = (job.get("description") or "").strip()
        if desc:
            h = _md5(desc)
            freq[h] = freq.get(h, 0) + 1

def _raw_bandwidth(c, freq):
    return sum(
        freq.get(_md5((job.get("description") or "").strip()), 0)
        for job in c.get("career_history", [])
        if (job.get("description") or "").strip()
    )


def stamp_fabrication_bandwidth(c, freq, max_bandwidth):
    """
    Called in Pass 2 after the full freq table is built.

    fabrication_bandwidth normalized to [0, 1]:
      raw_bandwidth = sum of global frequencies of all career descriptions
                       this candidate carries (each description's global
                       template-reuse count, summed across their career)
      normalized    = raw_bandwidth / max_bandwidth  (max across entire dataset)
      If max_bandwidth = 0 (edge case: all unique descriptions), result = 0.0

    Why divide by max (not full min-max):
      min_bandwidth is always 0 (a candidate with all-unique descriptions).
      So min-max normalization reduces to raw / max.
      This gives a clean 0-1 scale where 1.0 = the most fabricated candidate
      in the entire dataset — and keeps layers.py L3 condition h
      (h = 0.06 * fabrication_bandwidth + ...) numerically sane, since h
      is meant to stay within [0, 1].

    Note: duplicate_job_descriptions (FL18) is computed earlier in
    process_candidate() — it is a within-candidate check and does not
    depend on the global frequency table.
    """
    raw = _raw_bandwidth(c, freq)
    c["fabrication_bandwidth"] = round(raw / max_bandwidth, 6) if max_bandwidth > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION  (5-dimension tree routing)
# ══════════════════════════════════════════════════════════════════════════════

def _is_pure_cv(c) -> bool:
    """
    Returns True if the candidate is a pure computer vision specialist
    WITHOUT significant NLP/retrieval/IR exposure.

    From JD: "People whose primary expertise is computer vision, speech,
    or robotics without significant NLP/IR exposure — we respect your work
    but you'd be re-learning fundamentals here."

    Pure CV = title matches CV keywords AND skill set does NOT contain
    retrieval/NLP/IR skills (embeddings, vector DBs, sentence transformers,
    FAISS, Pinecone, Qdrant, Weaviate, Milvus, OpenSearch, NLP,
    semantic search, IR, learning to rank, BM25, RAG, LLMs, etc.)

    If they have even one retrieval/NLP skill → NOT pure CV →
    route to ml_engineer instead.
    """
    title       = _s(c.get("profile", {}).get("current_title", ""))
    skill_names = {_s(sk.get("name")) for sk in c.get("skills", [])}

    is_cv_title   = any(kw in title for kw in CV_TITLE_KEYWORDS)
    has_retrieval = bool(skill_names & RETRIEVAL_NLP_SKILLS)

    return is_cv_title and not has_retrieval


def _code_status(c) -> str:
    """
    code    = current title IS a hands-on production coding role.
    no_code = everything else.

    We check current title against CODE_TITLES (positive match).
    This is correct because:
    - A Business Analyst is no_code regardless of duration
    - A Software Engineer is code regardless of duration
    - We are not trying to detect "drift" — we are asking
      "is this person writing code RIGHT NOW"

    Edge cases:
    - Zero career history → no_code (pure academic)
    - Entire career in pure research titles → no_code
    """
    career = c.get("career_history", [])

    # No career history at all → no_code
    if not career:
        return "no_code"

    # Entire career is pure research/academic → no_code
    all_research = all(
        any(rt in _s(job.get("title", "")) for rt in PURE_RESEARCH_TITLES)
        for job in career
    )
    if all_research:
        return "no_code"

    # Check current title against CODE_TITLES
    current_job = next((j for j in career if j.get("is_current")), None)
    if current_job:
        title = _s(current_job.get("title", ""))
        if any(ct in title for ct in CODE_TITLES):
            return "code"
        return "no_code"

    # No current job flagged → fall back to most recent job
    most_recent = sorted(
        [j for j in career if j.get("start_date")],
        key=lambda j: j.get("start_date", ""),
        reverse=True
    )
    if most_recent:
        title = _s(most_recent[0].get("title", ""))
        if any(ct in title for ct in CODE_TITLES):
            return "code"

    return "no_code"


def _availability_bucket(c) -> str:
    """
    Single availability dimension.

    available     = open_to_work_flag=True
                     AND willing_to_relocate=True
                     AND notice_period_days < 90

    not_available = any other combination:
                     - not open to work, OR
                     - not willing to relocate, OR
                     - notice period >= 90 days

    Why all three required:
      - open_to_work=True   : explicit intent signal - actively looking
      - willing_to_relocate : role is Pune/Noida based, relocation needed
      - notice < 90 days    : 90+ days is too long for a startup to wait

    Uses "not_available" rather than the literal "unavailable" — the latter
    is in pruning.py's INACTIVE_FOLDER_NAMES and would cause the entire
    folder to be hard-dropped in Phase 1b, silently discarding candidates
    who are simply not relocation-ready right now (not actually inactive).
    """
    sig          = c.get("redrob_signals", {})
    open_to_work = sig.get("open_to_work_flag", False)
    willing      = sig.get("willing_to_relocate", False)
    notice       = sig.get("notice_period_days")
    if notice is None:
        notice = 999

    if open_to_work and willing and notice < 90:
        return "available"
    return "not_available"


def _experience_band(c) -> str:
    yoe = c.get("profile", {}).get("years_of_experience")
    if yoe is None: return "0_to_3"
    if yoe <= 3:    return "0_to_3"
    if yoe <= 9:    return "4_to_9"
    return "10_plus"


def _domain_and_role(c) -> tuple[str, str]:
    """
    Routes candidate to (domain, role_slug).

    Special rule for computer_vision_engineer:
      If title matches CV keywords:
        - pure CV (no retrieval/NLP skills) → computer_vision_engineer.json
        - has retrieval/NLP skills          → ml_engineer.json
          (they cross the JD threshold and are NOT pure CV)

    All other roles: first title keyword match in DOMAIN_ROLE_RULES wins.
    """
    title       = _s(c.get("profile", {}).get("current_title", ""))
    skill_names = {_s(sk.get("name")) for sk in c.get("skills", [])}

    # Check CV special rule first
    is_cv_title = any(kw in title for kw in CV_TITLE_KEYWORDS)
    if is_cv_title:
        has_retrieval = bool(skill_names & RETRIEVAL_NLP_SKILLS)
        if has_retrieval:
            # Has NLP/retrieval depth → not pure CV → ml_engineer
            return "engineering", "ml_engineer"
        else:
            # Pure CV → computer_vision_engineer
            return "engineering", "computer_vision_engineer"

    # Standard routing
    for domain, role_slug, keywords in DOMAIN_ROLE_RULES:
        if role_slug == "computer_vision_engineer":
            continue  # already handled above
        if any(kw in title for kw in keywords):
            return domain, role_slug

    return OTHER_DOMAIN, OTHER_ROLE_SLUG


def classify(c) -> dict:
    domain, role = _domain_and_role(c)
    return {
        "code_status":  _code_status(c),
        "availability": _availability_bucket(c),
        "experience":   _experience_band(c),
        "domain":       domain,
        "role":         role,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TREE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _leaf_path(dataset_dir, b) -> str:
    return os.path.join(
        dataset_dir,
        b["code_status"],
        b["availability"],
        b["experience"],
        b["domain"],
        b["role"] + ".json"
    )

def build_tree(all_candidates, dataset_dir) -> int:
    grouped = defaultdict(list)
    for c in all_candidates:
        path = _leaf_path(dataset_dir, classify(c))
        grouped[path].append(c)

    for path, candidates in grouped.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(candidates, fh, ensure_ascii=False, indent=2)

    return len(grouped)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    if not os.path.exists(INPUT_FILE):
        print(f"❌  File not found: {INPUT_FILE}")
        return

    wall_start = time.time()
    desc_freq  = {}

    # ── Pass 1: read + clean + collect desc frequencies ───────────────────────
    print(f"📂  Reading and cleaning {INPUT_FILE} ...")
    cleaned   = []
    skipped   = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line: continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"    ⚠  Skipped line {i}: {e}")
                skipped += 1
                continue

            collect_desc_freq(desc_freq, raw)
            c = process_candidate(raw)
            cleaned.append(c)

            if i % 10_000 == 0:
                elapsed = time.time() - wall_start
                print(f"    ... {i:,} processed  ({elapsed:.1f}s elapsed)")

    t1 = time.time() - wall_start
    print(f"    ✅ {len(cleaned):,} cleaned  |  {skipped} skipped  |  {t1:.1f}s")

    # ── Pass 2: stamp global flags (fabrication bandwidth) ───────────────────
    print(f"\n🔖  Stamping fabrication bandwidth ...")
    t = time.time()

    max_bandwidth = max((_raw_bandwidth(c, desc_freq) for c in cleaned), default=1)
    print(f"    max raw bandwidth across dataset : {max_bandwidth:,}")

    for c in cleaned:
        stamp_fabrication_bandwidth(c, desc_freq, max_bandwidth)
    print(f"    ✅ fabrication_bandwidth stamped for all candidates  |  {time.time()-t:.1f}s")

    # ── Pass 3: build tree ────────────────────────────────────────────────────
    print(f"\n🌳  Building Dataset/ tree ...")
    t = time.time()
    buckets_written = build_tree(cleaned, DATASET_DIR)
    print(f"    ✅ {buckets_written} leaf files written  |  {time.time()-t:.1f}s")

    # ── Compute summary counts ─────────────────────────────────────────────────
    total         = len(cleaned)
    code_count    = sum(1 for c in cleaned if _code_status(c)         == "code")
    no_code_count = total - code_count
    avail_count   = sum(1 for c in cleaned if _availability_bucket(c) == "available")
    pure_cv_count = sum(1 for c in cleaned if _is_pure_cv(c))
    total_time    = time.time() - wall_start

    dup_flagged     = sum(1 for c in cleaned if c.get("duplicate_job_descriptions"))
    avg_fabrication = sum(c.get("fabrication_bandwidth", 0) for c in cleaned) / max(total, 1)

    print()
    print("=" * 62)
    print("  PIPELINE COMPLETE")
    print("=" * 62)
    print(f"  Candidates loaded        : {total + skipped:,}")
    print(f"  Candidates cleaned       : {total:,}")
    print(f"  Skipped (malformed)      : {skipped}")
    print(f"  ── Quality ────────────────────────────────────────────")
    print(f"  Duplicate descriptions   : {dup_flagged:,}  ({100*dup_flagged/max(total,1):.1f}%)")
    print(f"  Avg fabrication_bandwidth: {avg_fabrication:.3f}  (0-1 scale)")
    print(f"  Max bandwidth (raw)      : {max_bandwidth:,}")
    print(f"  ── Classification ─────────────────────────────────────")
    print(f"  code                     : {code_count:,}  ({100*code_count/max(total,1):.1f}%)")
    print(f"  no_code                  : {no_code_count:,}  ({100*no_code_count/max(total,1):.1f}%)")
    print(f"  available                : {avail_count:,}  ({100*avail_count/max(total,1):.1f}%)")
    print(f"  not_available            : {total-avail_count:,}  ({100*(total-avail_count)/max(total,1):.1f}%)")
    print(f"  pure CV (no retrieval)   : {pure_cv_count:,}  ({100*pure_cv_count/max(total,1):.1f}%)")
    print(f"  ── Output ─────────────────────────────────────────────")
    print(f"  Unique descriptions      : {len(desc_freq):,}")
    print(f"  Leaf files written       : {buckets_written}")
    print(f"  Output folder            : {os.path.abspath(DATASET_DIR)}/")
    print(f"  ── Performance ────────────────────────────────────────")
    print(f"  Total wall time          : {total_time:.1f}s")
    print()


def run_preprocessing(input_file, dataset_dir) -> "Path":
    """
    Importable entry point for rank.py and test_pruning.py.
    Clears dataset_dir, runs all 3 passes, builds the Dataset/ tree.
    """
    import shutil
    from pathlib import Path as _Path

    input_file  = _Path(input_file)
    dataset_dir = _Path(dataset_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"candidates.jsonl not found: {input_file}")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily override the module-level globals so run() uses the right paths
    global INPUT_FILE, DATASET_DIR
    _orig_input, _orig_dataset = INPUT_FILE, DATASET_DIR
    INPUT_FILE  = str(input_file)
    DATASET_DIR = str(dataset_dir)
    try:
        run()
    finally:
        INPUT_FILE  = _orig_input
        DATASET_DIR = _orig_dataset

    return dataset_dir


if __name__ == "__main__":
    run()