"""
constants.py — Central configuration for RedRob pipeline.
All weights, thresholds, paths, and lookup tables live here.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "decompressed"
DATA_DIR = PROJECT_ROOT / "data"
JD_JSON_PATH = DATA_DIR / "jd.json"

# Model sub-paths (after setup.sh decompresses)
# Three models only: MiniLM sentence-transformer, FlashRank, Fraud KB
SENTENCE_TRANSFORMER_DIR = MODELS_DIR / "sentence_transformer"
FLASHRANK_DIR = MODELS_DIR / "ms-marco-MiniLM-L-12-v2"   # actual dir from tar extraction
FRAUD_KB_PATH = MODELS_DIR / "fraud_kb" / "fraud_kb.db"

# Embedding model name (downloaded by build_kb.py)
ST_MODEL_NAME = "all-MiniLM-L6-v2"
FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
EMBED_DIM = 384

# ──────────────────────────────────────────────────────────────────────────────
# GATE / THRESHOLDS
# ──────────────────────────────────────────────────────────────────────────────
GATE_TOP_FRACTION = 0.50      # top 50% by l1c_score
GATE_RANDOM_FRACTION = 0.25   # + random 25% from bottom half
GATE_TOTAL = GATE_TOP_FRACTION + GATE_RANDOM_FRACTION  # 75%

FOLDER_PRUNE_THRESHOLD = 0.15  # folder token-overlap cutoff
FOLDER_DISPATCH_STAGGER_SEC = 0   # 0 = all folder threads start simultaneously
MIN_FOLDERS_KEPT = 1           # never drop everything

HARD_POOL_CAP = 60_000         # hard cap before expensive layers

# ──────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS (feed into FIS membership; also used for fallback weighted sum)
# ──────────────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "L1C": 0.35,  # explicit skill match score
    "L4": 0.25,   # semantic work-to-JD relevance
    "L6": 0.10,   # behavioral
    # L3 and L1b are penalty multipliers, not weighted scores
}

# Soft penalty magnitudes
SENIORITY_PENALTY = 0.85       # multiply score if under-qualified (L3)
DOWNLEVEL_SENIORITY_PENALTY = 0.85  # multiply score if candidate held a higher title (L3)

# ──────────────────────────────────────────────────────────────────────────────
# BATCH SIZES
# ──────────────────────────────────────────────────────────────────────────────
L1C_MIN_SKILL_MATCH = 0.0    # per-folder gate inside l1c_skill_match (0.0 = off)
L4_BATCH_SIZE = 64
FLASHRANK_TOP_N = 50           # polish only top 50
FLASHRANK_FIS_WEIGHT = 0.25    # blend weight: final = W*flashrank + (1-W)*fis  → FIS weight = 0.75
OUTPUT_TOP_N = 100             # final CSV rows

# ──────────────────────────────────────────────────────────────────────────────
# L7 TIER THRESHOLDS — "very good" candidate criteria
# A candidate meeting ALL of these (with no L1b flags and no L3 penalty)
# is promoted to tier "very_good" and guaranteed placement in the top 10.
# ──────────────────────────────────────────────────────────────────────────────
L7_VERY_GOOD_L1C_MIN = 0.65       # skill match score
L7_VERY_GOOD_L4_MIN = 0.60        # semantic work relevance
L7_VERY_GOOD_L6_MIN = 0.60        # behavioral signals
L7_VERY_GOOD_EXP_MIN = 7.0        # years of experience
L7_VERY_GOOD_COMPANY_AGE_MIN = 10.0  # oldest company must be ≥ 10 yrs old (0 = no data, skip)

# ──────────────────────────────────────────────────────────────────────────────
# FOLDER NAME ABBREVIATION EXPANSION (pruning)
# ──────────────────────────────────────────────────────────────────────────────
FOLDER_ABBREVIATIONS = {
    "swe": "software engineer",
    "sde": "software development engineer",
    "ml": "machine learning",
    "mlops": "machine learning operations ml ops",
    "ai": "artificial intelligence",
    "ds": "data scientist data science",
    "de": "data engineer",
    "hr": "human resources recruiter",
    "pm": "product manager",
    "qa": "quality assurance testing",
    "devops": "development operations devops",
    "cloud": "cloud computing infrastructure",
    "fe": "frontend front end",
    "be": "backend back end",
    "fullstack": "full stack",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "sre": "site reliability engineer",
    "ux": "user experience design",
    "ui": "user interface design",
    "sec": "security",
}

# ──────────────────────────────────────────────────────────────────────────────
# SKILL ALIASES (L2 text normalization / semantic_neighbors fallback)
# ──────────────────────────────────────────────────────────────────────────────
SKILL_ALIASES = {
    "llm": ["large language models", "large language model"],
    "ml": ["machine learning"],
    "nlp": ["natural language processing"],
    "dl": ["deep learning"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "cv": ["computer vision"],
    "k8s": ["kubernetes"],
    "js": ["javascript"],
    "ts": ["typescript"],
    "pg": ["postgresql", "postgres"],
}

# ──────────────────────────────────────────────────────────────────────────────
# SENIORITY LEVELS (L3)
# ──────────────────────────────────────────────────────────────────────────────
"""SENIORITY_KEYWORDS = {
    "intern": 0,
    "junior": 1,
    "entry": 1,
    "associate": 2,
    "mid": 3,
    "intermediate": 3,
    "senior": 5,
    "lead": 7,
    "staff": 8,
    "principal": 9,
    "director": 10,
}"""

# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED FOLDER PRUNING RULES
# ──────────────────────────────────────────────────────────────────────────────

# Recognised spellings of Bangalore (JD location field)
BANGALORE_ALIASES: frozenset = frozenset({
    "bangalore", "bengaluru", "blr", "bangaluru", "bengalore",
    "bangalore city", "bangalore urban",
})

# Folder-path substrings that mean "candidates NOT from the target city".
# Checked against normalised (lowercase, _ → space) path components.
INDIA_OUTSIDE_PATTERNS: tuple = (
    "india outside", "india_outside", "outside india", "outside_india",
    "rest of india", "rest_of_india", "non bangalore", "non_bangalore",
    "other cities", "other_cities", "pan india", "pan_india",
    "other locations", "other_locations",
)

# Keywords that classify a JD role as "engineering".
# Matched against the normalised job_title + required skills text.
ENGINEERING_ROLE_KEYWORDS: frozenset = frozenset({
    "engineer", "engineering", "developer", "dev", "sde", "swe",
    "backend", "frontend", "fullstack", "full stack",
    "data", "ml", "ai", "aiml", "machine learning", "deep learning",
    "cloud", "devops", "mlops", "platform", "infrastructure", "sre",
    "python", "java", "golang", "scala", "rust", "typescript",
    "nlp", "computer vision", "data scientist", "data science", "software",
})

# Folder-path substrings that mean "non-engineering candidates".
# If the JD is engineering, folders matching any of these are hard-dropped.
OTHER_ROLE_FOLDER_PATTERNS: tuple = (
    "other role", "other_role", "non engineering", "non_engineering",
    "non tech", "non_tech", "business", "sales", "marketing",
    "human resources", "finance", "legal", "operations",
    "customer success", "customer_success",
)

# Candidate-level status values that mean the profile is inactive.
# Checked (lowercased) against any of: status, candidate_status, active_status.
INACTIVE_CANDIDATE_STATUSES: frozenset = frozenset({
    "inactive", "not active", "not_active", "closed", "unavailable",
    "on hold", "hold", "deactivated", "blacklisted", "rejected",
    "not available", "passive", "archived",
})

# Folder names (normalised: lowercase, _ / - → space) where the ENTIRE FOLDER
# is a bucket of inactive candidates.  The branch is hard-dropped in Phase 1
# before any candidate is loaded.
INACTIVE_FOLDER_NAMES: frozenset = frozenset({
    "inactive", "not active", "closed", "unavailable",
    "deactivated", "blacklisted", "rejected", "archived",
})

# ── New tree: root-level code/no-code bucket identifiers ─────────────────────
# Used by Phase 1a to prune the entire branch when the JD's coding requirement
# contradicts the folder's root-level bucket.

# Normalised root folder names that mean "no coding required" candidates.
NO_CODE_FOLDER_ROOTS: frozenset = frozenset({
    "no code", "no_code", "nocode",
    "non code", "non_code",
    "non tech", "non_tech",
    "no coding", "non coding", "non_coding",
})

# Normalised root folder names that mean "coding / technical" candidates.
CODE_FOLDER_ROOTS: frozenset = frozenset({
    "code", "coding", "tech", "technical",
})

# Folder names that confirm a branch is openly active/eligible.
# Used for trace logging clarity only.
ACTIVE_FOLDER_NAMES: frozenset = frozenset({
    "active", "open", "available", "eligible", "moderate",
})

# Default years-to-level mapping for candidate experience
def years_to_seniority(years: float) -> int:
    if years < 1:   return 0
    if years < 2:   return 1
    if years < 3:   return 2
    if years < 5:   return 3
    if years < 7:   return 5
    if years < 9:   return 7
    if years < 11:  return 8
    return 9
