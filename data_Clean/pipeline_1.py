# # pipeline.py
# # ─────────────────────────────────────────────────────────────────────────────
# # PURPOSE
# #   Single-pass pipeline.
# #   Reads candidates.jsonl → cleans every candidate (fixes + flags)
# #   → computes global description frequency in the same pass
# #   → stamps each candidate with possible_fabrication + fabrication_bandwidth
# #   → builds the Dataset/ folder tree
# #   All in one run, no intermediate files, no second read.
# #
# # FABRICATION FLAGS (added to every candidate at the END of processing)
# #   fabrication_bandwidth  : int   — how many times this candidate's job
# #                                    descriptions appear across the full
# #                                    dataset (sum over all career entries)
# #   possible_fabrication   : bool  — True if fabrication_bandwidth > 100
# #
# # WHY A SECOND PASS OVER cleaned[] IS NEEDED
# #   Description frequencies can only be known after ALL candidates are read.
# #   So the flow is:
# #     Pass 1 (file read)  → clean each candidate + build freq hash table
# #     Pass 2 (in-memory)  → stamp fabrication flags onto each cleaned candidate
# #     Pass 3 (in-memory)  → build and write the Dataset/ tree
# #   Passes 2 and 3 are over the already-loaded list — no extra file I/O.
# #
# # USAGE
# #   python pipeline.py
# # ─────────────────────────────────────────────────────────────────────────────

# import json
# import os
# import time
# import hashlib
# from datetime import date, datetime
# from collections import defaultdict

# # ── Config ────────────────────────────────────────────────────────────────────
# INPUT_FILE  = "candidates.jsonl"
# DATASET_DIR = "Dataset"
# TODAY       = date(2026, 6, 17)

# FABRICATION_THRESHOLD = 100   # freq > this → possible_fabrication = True


# # ══════════════════════════════════════════════════════════════════════════════
# #  CONSTANTS
# # ══════════════════════════════════════════════════════════════════════════════

# POSTGRAD_DEGREES  = {"m.tech", "m.e.", "m.s.", "m.sc", "m.sc.", "ph.d", "mba", "m.b.a"}
# UNDERGRAD_DEGREES = {"b.tech", "b.e.", "b.sc", "b.sc.", "b.a.", "be", "btech", "b.eng"}
# ENG_DEGREE_SET    = {"b.tech", "m.tech", "b.e.", "m.e.", "be", "btech", "mtech", "b.eng"}
# INVALID_ENG_FIELDS = {"mba", "m.b.a", "commerce"}

# AI_SKILLS = {
#     "machine learning", "deep learning", "nlp", "computer vision",
#     "object detection", "image classification", "rag", "llms",
#     "fine-tuning llms", "vector search", "embeddings", "faiss",
#     "pinecone", "qdrant", "weaviate", "sentence transformers",
#     "hugging face transformers", "langchain", "llamaindex",
#     "diffusion models", "gans", "yolo", "cnn", "rnn", "lstm",
#     "transformers", "bert", "gpt", "reinforcement learning",
#     "recommendation systems", "information retrieval",
#     "semantic search", "prompt engineering", "mlops", "mlflow",
#     "weights & biases", "kubeflow", "bentoml",
#     "speech recognition", "tts", "asr", "forecasting",
#     "time series", "statistical modeling", "learning to rank",
#     "bm25", "opensearch", "haystack", "qlora", "pgvector",
#     "feature engineering", "data science", "opencv",
#     "scikit-learn", "xgboost", "lightgbm", "prophet",
#     "tensorflow", "pytorch", "keras"
# }

# AI_CAREER_KEYWORDS = {
#     "ml engineer", "machine learning", "data scientist", "ai specialist",
#     "ai engineer", "nlp engineer", "computer vision", "research engineer",
#     "deep learning", "junior ml", "senior ml", "ai research",
#     "applied scientist", "mlops engineer"
# }

# AI_INDUSTRIES_EXACT = {
#     "ai/ml", "ai", "ml", "machine learning", "artificial intelligence",
#     "edtech ai", "healthtech ai", "deep learning", "nlp"
# }

# DE_DOMAIN_SKILLS = {
#     "apache spark", "spark", "kafka", "airflow", "dbt", "hadoop",
#     "hive", "presto", "trino", "databricks", "snowflake", "redshift",
#     "bigquery", "data warehouse", "data lake", "etl", "elt",
#     "pipeline", "flink", "nifi", "luigi", "prefect", "dagster",
#     "glue", "fivetran", "stitch", "talend", "informatica",
#     "delta lake", "iceberg", "parquet", "avro", "pyspark"
# }

# AI_TITLE_KEYWORDS = {
#     "ml engineer", "machine learning", "data scientist", "ai engineer",
#     "ai specialist", "nlp engineer", "computer vision engineer",
#     "deep learning engineer", "research scientist", "ai research",
#     "applied scientist", "mlops engineer"
# }

# DE_TITLE_KEYWORDS = {
#     "data engineer", "analytics engineer", "etl developer",
#     "data pipeline", "data architect", "big data", "spark engineer",
#     "kafka engineer", "databricks engineer"
# }

# SWE_TITLE_KEYWORDS = {
#     "software engineer", "software developer", "backend engineer",
#     "frontend engineer", "full stack", "fullstack", "sde",
#     "platform engineer", "devops engineer", "site reliability",
#     "cloud engineer", "systems engineer"
# }

# INDIA_PREFERRED_CITIES = {
#     "noida", "pune", "bangalore", "bengaluru", "hyderabad",
#     "chennai", "mumbai", "delhi", "gurugram", "gurgaon",
#     "navi mumbai", "thane", "kolkata"
# }

# _SCORE_FIELDS = [
#     "skill_career_domain_mismatch",
#     "education_overlap",
#     "reverse_degree_order",
#     "second_undergrad_after_first",
#     "education_career_gap_flag",
#     "active_before_signup",
#     "duplicate_job_descriptions",
#     "all_descriptions_identical",
#     "low_engagement_flag",
#     "possible_honeypot",
#     "_any_invalid_degree_field",
# ]


# # ══════════════════════════════════════════════════════════════════════════════
# #  SHARED HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def _s(val) -> str:
#     return (val or "").lower().strip()


# def _parse_date(s) -> date | None:
#     if not s:
#         return None
#     try:
#         return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
#     except Exception:
#         return None


# def _has_ai_career(career: list) -> bool:
#     for job in career:
#         title = _s(job.get("title"))
#         ind   = _s(job.get("industry"))
#         if any(kw in title for kw in AI_CAREER_KEYWORDS):
#             return True
#         if ind in AI_INDUSTRIES_EXACT:
#             return True
#         if "machine learning" in ind or "artificial intelligence" in ind:
#             return True
#     return False


# def _md5(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()


# # ══════════════════════════════════════════════════════════════════════════════
# #  FIXES  (F1, F2, F3)
# # ══════════════════════════════════════════════════════════════════════════════

# def _fix_salary(sig: dict) -> None:
#     sal = sig.setdefault("expected_salary_range_inr_lpa", {})
#     mn, mx = sal.get("min"), sal.get("max")
#     if mn is not None and mx is not None and mn > mx:
#         sal["min"], sal["max"] = mx, mn
#         sal["salary_was_inverted"] = True
#     else:
#         sal["salary_was_inverted"] = False


# def _fix_github(sig: dict) -> None:
#     score = sig.get("github_activity_score")
#     if score == -1 or score is None:
#         sig["github_activity_score"] = None
#         sig["github_not_linked"]     = True
#     else:
#         sig["github_not_linked"] = False


# def _fix_offer(sig: dict) -> None:
#     if sig.get("offer_acceptance_rate") == -1:
#         sig["offer_acceptance_rate"] = None
#         sig["no_offer_history"]      = True
#     else:
#         sig["no_offer_history"] = False


# def apply_fixes(c: dict) -> None:
#     sig = c.setdefault("redrob_signals", {})
#     _fix_salary(sig)
#     _fix_github(sig)
#     _fix_offer(sig)
#     sal = sig.get("expected_salary_range_inr_lpa", {})
#     c["salary_was_inverted"] = sal.get("salary_was_inverted", False)
#     c["github_not_linked"]   = sig.get("github_not_linked",   False)
#     c["no_offer_history"]    = sig.get("no_offer_history",    False)


# # ══════════════════════════════════════════════════════════════════════════════
# #  FLAGS
# # ══════════════════════════════════════════════════════════════════════════════

# def fl3_skill_career_domain_mismatch(c: dict) -> bool:
#     skills   = c.get("skills", [])
#     career   = c.get("career_history", [])
#     ai_count = sum(1 for sk in skills if _s(sk.get("name")) in AI_SKILLS)
#     if ai_count < 3:
#         return False
#     return not _has_ai_career(career)


# def fl5_education_overlap(edu_list: list) -> tuple[bool, list]:
#     pairs = []
#     for i in range(len(edu_list)):
#         for j in range(i + 1, len(edu_list)):
#             s1 = edu_list[i].get("start_year")
#             e1 = edu_list[i].get("end_year")
#             s2 = edu_list[j].get("start_year")
#             e2 = edu_list[j].get("end_year")
#             if None in (s1, e1, s2, e2):
#                 continue
#             if max(s1, s2) < min(e1, e2):
#                 pairs.append([i, j])
#     return bool(pairs), pairs


# def fl6_reverse_degree_order(edu_list: list) -> bool:
#     pg_ends, ug_starts = [], []
#     for e in edu_list:
#         d  = _s(e.get("degree"))
#         ey = e.get("end_year")
#         sy = e.get("start_year")
#         if d in POSTGRAD_DEGREES  and ey is not None: pg_ends.append(ey)
#         if d in UNDERGRAD_DEGREES and sy is not None: ug_starts.append(sy)
#     if not pg_ends or not ug_starts:
#         return False
#     return min(pg_ends) <= max(ug_starts)


# def fl7_second_undergrad_after_first(edu_list: list) -> bool:
#     ugs = sorted(
#         [e for e in edu_list
#          if _s(e.get("degree")) in UNDERGRAD_DEGREES
#          and e.get("start_year") is not None
#          and e.get("end_year")   is not None],
#         key=lambda e: e["start_year"]
#     )
#     if len(ugs) < 2:
#         return False
#     return ugs[1]["start_year"] >= ugs[0]["end_year"] + 2


# def fl8_education_career_gap(c: dict) -> tuple[bool, float]:
#     edu_ends = [
#         e["end_year"] for e in c.get("education", [])
#         if e.get("end_year") is not None
#     ]
#     career_starts = [
#         _parse_date(j.get("start_date")).year
#         for j in c.get("career_history", [])
#         if _parse_date(j.get("start_date"))
#     ]
#     if not edu_ends or not career_starts:
#         return False, 0.0
#     gap = float(min(career_starts) - max(edu_ends))
#     flagged = gap > 1
#     return flagged, round(gap, 1)


# def fl11_active_before_signup(c: dict) -> tuple[bool, int]:
#     sig    = c.get("redrob_signals", {})
#     signup = _parse_date(sig.get("signup_date"))
#     last   = _parse_date(sig.get("last_active_date"))
#     if signup is None or last is None:
#         return False, 0
#     if last < signup:
#         return True, (signup - last).days
#     return False, 0


# def fl16_invalid_degree_field(edu: dict) -> bool:
#     deg   = _s(edu.get("degree"))
#     field = _s(edu.get("field_of_study"))
#     if deg in ENG_DEGREE_SET and field in INVALID_ENG_FIELDS:
#         return True
#     if deg in {"m.sc", "m.sc.", "m.s.", "m.s"} and field in INVALID_ENG_FIELDS:
#         return True
#     return False


# def fl18_fl19_duplicate_descriptions(career: list) -> tuple[bool, bool, list]:
#     if len(career) < 2:
#         return False, False, []
#     descs = [(j.get("description") or "").strip() for j in career]
#     pairs = [
#         [i, j] for i in range(len(descs))
#         for j in range(i + 1, len(descs))
#         if descs[i] and descs[i] == descs[j]
#     ]
#     non_empty = [d for d in descs if d]
#     all_ident  = len(non_empty) > 1 and len(set(non_empty)) == 1
#     return bool(pairs), all_ident, pairs


# def fl20_low_engagement(c: dict) -> bool:
#     sig  = c.get("redrob_signals", {})
#     rate = sig.get("recruiter_response_rate")
#     hrs  = sig.get("avg_response_time_hours")
#     if rate is None or hrs is None:
#         return False
#     return rate < 0.10 and hrs > 200


# def fl25_possible_honeypot(c: dict) -> bool:
#     structural = sum([
#         bool(c.get("education_overlap")),
#         bool(c.get("reverse_degree_order")),
#         bool(c.get("second_undergrad_after_first")),
#         bool(c.get("all_descriptions_identical")),
#         bool(c.get("duplicate_job_descriptions")),
#     ])
#     return structural >= 3


# def fl26_honeypot_score(c: dict) -> int:
#     return sum(1 for f in _SCORE_FIELDS if c.get(f, False))


# # ══════════════════════════════════════════════════════════════════════════════
# #  MASTER CLEAN FUNCTION
# # ══════════════════════════════════════════════════════════════════════════════

# def process_candidate(c: dict) -> dict:
#     edu_list = c.get("education", [])
#     career   = c.get("career_history", [])

#     apply_fixes(c)

#     c["skill_career_domain_mismatch"] = fl3_skill_career_domain_mismatch(c)

#     ov_flag, ov_pairs = fl5_education_overlap(edu_list)
#     c["education_overlap"]             = ov_flag
#     c["overlapping_education_indices"] = ov_pairs

#     c["reverse_degree_order"]         = fl6_reverse_degree_order(edu_list)
#     c["second_undergrad_after_first"] = fl7_second_undergrad_after_first(edu_list)

#     gap_flag, gap_val = fl8_education_career_gap(c)
#     c["education_career_gap_flag"]    = gap_flag
#     c["education_career_gap_years"]   = gap_val

#     abs_flag, abs_days = fl11_active_before_signup(c)
#     c["active_before_signup"]         = abs_flag
#     c["signup_active_gap_days"]       = abs_days

#     dup_any, all_ident, dup_idx = fl18_fl19_duplicate_descriptions(career)
#     c["duplicate_job_descriptions"]    = dup_any
#     c["all_descriptions_identical"]    = all_ident
#     c["duplicate_description_indices"] = dup_idx

#     c["low_engagement_flag"] = fl20_low_engagement(c)

#     any_invalid = False
#     for edu in edu_list:
#         val = fl16_invalid_degree_field(edu)
#         edu["invalid_degree_field_combination"] = val
#         if val:
#             any_invalid = True
#     c["_any_invalid_degree_field"] = any_invalid

#     c["possible_honeypot"] = fl25_possible_honeypot(c)
#     c["honeypot_score"]    = fl26_honeypot_score(c)

#     # fabrication flags are stamped LATER (after freq table is complete)
#     # placeholders so the field always exists in the dict
#     c["fabrication_bandwidth"] = 0
#     c["possible_fabrication"]  = False

#     return c


# # ══════════════════════════════════════════════════════════════════════════════
# #  DESCRIPTION FREQUENCY  (hash table built during read pass)
# # ══════════════════════════════════════════════════════════════════════════════

# def collect_desc_freq(freq: dict, c: dict) -> None:
#     """
#     Called once per candidate during the file read loop.
#     Counts every description occurrence globally (including duplicates
#     within the same candidate's career).
#     Key = MD5(description).  Value = running count.
#     """
#     for job in c.get("career_history", []):
#         desc = (job.get("description") or "").strip()
#         if not desc:
#             continue
#         h = _md5(desc)
#         freq[h] = freq.get(h, 0) + 1


# def stamp_fabrication(c: dict, freq: dict) -> None:
#     """
#     Called in a second in-memory pass after ALL candidates are read
#     and the freq table is complete.

#     fabrication_bandwidth:
#         Sum of global frequencies of every career description this
#         candidate carries.  Tells you how many times their job
#         descriptions collectively appear across the entire dataset.

#     possible_fabrication:
#         True if fabrication_bandwidth > FABRICATION_THRESHOLD (100).
#         Signals the candidate's career text is likely copy-pasted
#         from a small template pool shared by many other candidates.
#     """
#     total = 0
#     for job in c.get("career_history", []):
#         desc = (job.get("description") or "").strip()
#         if desc:
#             total += freq.get(_md5(desc), 0)

#     c["fabrication_bandwidth"] = total
#     c["possible_fabrication"]  = total > FABRICATION_THRESHOLD


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLASSIFICATION  (4-dimension tree routing)
# # ══════════════════════════════════════════════════════════════════════════════

# def _country_bucket(c: dict) -> str:
#     profile  = c.get("profile", {})
#     country  = _s(profile.get("country"))
#     location = _s(profile.get("location"))
#     if country in ("india", "in"):
#         city = location.split(",")[0].strip()
#         if any(pref in city for pref in INDIA_PREFERRED_CITIES):
#             return "india_preferred"
#         return "india_other"
#     return "outside_india"


# def _experience_band(c: dict) -> str:
#     yoe = c.get("profile", {}).get("years_of_experience")
#     if yoe is None: return "0_to_4"
#     if yoe < 5:     return "0_to_4"
#     if yoe <= 9:    return "5_to_9"
#     return "10_plus"


# def _domain_bucket(c: dict) -> str:
#     title       = _s(c.get("profile", {}).get("current_title"))
#     industry    = _s(c.get("profile", {}).get("current_industry"))
#     skill_names = {_s(sk.get("name")) for sk in c.get("skills", [])}
#     ai_overlap  = len(skill_names & AI_SKILLS)
#     de_overlap  = len(skill_names & DE_DOMAIN_SKILLS)

#     has_ai_title  = any(kw in title for kw in AI_TITLE_KEYWORDS)
#     has_de_title  = any(kw in title for kw in DE_TITLE_KEYWORDS)
#     has_swe_title = any(kw in title for kw in SWE_TITLE_KEYWORDS)
#     has_ai_ind    = (
#         industry in AI_INDUSTRIES_EXACT
#         or "machine learning" in industry
#         or "artificial intelligence" in industry
#     )

#     if has_ai_title or has_ai_ind:
#         return "ai_ml"
#     if ai_overlap >= 3 and _has_ai_career(c.get("career_history", [])):
#         return "ai_ml"
#     if has_de_title or de_overlap >= 3:
#         return "data_engineering"
#     if has_swe_title:
#         return "software_engineering"
#     return "other"


# def _availability_bucket(c: dict) -> str:
#     last = _parse_date(c.get("redrob_signals", {}).get("last_active_date"))
#     if last is None:
#         return "inactive"
#     days = (TODAY - last).days
#     if days <= 30:  return "active"
#     if days <= 90:  return "moderate"
#     return "inactive"


# def classify(c: dict) -> dict:
#     return {
#         "country":      _country_bucket(c),
#         "experience":   _experience_band(c),
#         "domain":       _domain_bucket(c),
#         "availability": _availability_bucket(c),
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# #  TREE BUILDER
# # ══════════════════════════════════════════════════════════════════════════════

# def _leaf_path(dataset_dir: str, buckets: dict) -> str:
#     return os.path.join(
#         dataset_dir,
#         buckets["country"],
#         buckets["experience"],
#         buckets["domain"],
#         buckets["availability"],
#         "candidates.json"
#     )


# def build_tree(all_candidates: list, dataset_dir: str) -> int:
#     grouped: dict[str, list] = defaultdict(list)
#     for c in all_candidates:
#         path = _leaf_path(dataset_dir, classify(c))
#         grouped[path].append(c)

#     for path, candidates in grouped.items():
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "w", encoding="utf-8") as fh:
#             json.dump(candidates, fh, ensure_ascii=False, indent=2)

#     return len(grouped)


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def run() -> None:
#     if not os.path.exists(INPUT_FILE):
#         print(f"❌  File not found: {INPUT_FILE}")
#         print(f"    Make sure candidates.jsonl is in the same folder.")
#         return

#     wall_start = time.time()

#     # freq table: md5 → global occurrence count
#     desc_freq: dict[str, int] = {}

#     # ── Pass 1: Read file → clean + collect description frequencies ───────────
#     print(f"📂  Reading and cleaning {INPUT_FILE} ...")
#     cleaned   = []
#     skipped   = 0
#     honeypots = 0

#     with open(INPUT_FILE, "r", encoding="utf-8") as fh:
#         for i, line in enumerate(fh, 1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 raw = json.loads(line)
#             except json.JSONDecodeError as e:
#                 print(f"    ⚠  Skipped line {i}: {e}")
#                 skipped += 1
#                 continue

#             # collect frequencies from raw before any mutation
#             collect_desc_freq(desc_freq, raw)

#             c = process_candidate(raw)
#             cleaned.append(c)
#             if c.get("possible_honeypot"):
#                 honeypots += 1

#             if i % 10_000 == 0:
#                 elapsed = time.time() - wall_start
#                 print(f"    ... {i:,} processed  ({elapsed:.1f}s elapsed)")

#     read_clean_time = time.time() - wall_start
#     print(f"    ✅ {len(cleaned):,} candidates cleaned  |  "
#           f"{skipped} skipped  |  "
#           f"{honeypots:,} honeypots  |  "
#           f"{read_clean_time:.1f}s")

#     # ── Pass 2 (in-memory): stamp fabrication flags onto every candidate ──────
#     print(f"\n🔖  Stamping fabrication flags ...")
#     fab_start  = time.time()
#     fabricated = 0
#     for c in cleaned:
#         stamp_fabrication(c, desc_freq)
#         if c["possible_fabrication"]:
#             fabricated += 1
#     print(f"    ✅ Done  |  possible_fabrication=True : {fabricated:,}  |  "
#           f"{time.time()-fab_start:.1f}s")

#     # ── Pass 3 (in-memory): build Dataset/ tree ───────────────────────────────
#     print(f"\n🌳  Building Dataset/ tree ...")
#     tree_start      = time.time()
#     buckets_written = build_tree(cleaned, DATASET_DIR)
#     print(f"    ✅ {buckets_written} leaf buckets written  |  "
#           f"{time.time()-tree_start:.1f}s")

#     total_time = time.time() - wall_start

#     print()
#     print("=" * 58)
#     print("  PIPELINE COMPLETE")
#     print("=" * 58)
#     print(f"  Candidates loaded        : {len(cleaned) + skipped:,}")
#     print(f"  Candidates cleaned       : {len(cleaned):,}")
#     print(f"  Skipped (malformed)      : {skipped}")
#     print(f"  Honeypots flagged        : {honeypots:,}  "
#           f"({100 * honeypots / max(len(cleaned), 1):.1f}%)")
#     print(f"  Possible fabrication     : {fabricated:,}  "
#           f"({100 * fabricated / max(len(cleaned), 1):.1f}%)")
#     print(f"  Unique descriptions      : {len(desc_freq):,}")
#     print(f"  Leaf buckets written     : {buckets_written}")
#     print(f"  Output folder            : {os.path.abspath(DATASET_DIR)}/")
#     print(f"  Total wall time          : {total_time:.1f}s")
#     print()


# if __name__ == "__main__":
#     run()


# pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Single-pass pipeline.
#   Reads candidates.jsonl → cleans → stamps fabrication flags
#   → builds the Dataset/ folder tree
#
# TREE STRUCTURE
#   Dataset/
#   └── {availability}        active | moderate | inactive
#       └── {country}         india_preferred | india_other | outside_india
#           └── {experience}  0_to_4 | 5_to_9 | 10_plus
#               └── {domain}  tech | data_and_ai | devops_and_cloud |
#                              product_and_design | operations | business |
#                              marketing | finance | hr_and_people |
#                              engineering_non_tech | other
#                   └── {role_type}.json   (e.g. python_developer.json,
#                                           ml_engineer.json, hr_manager.json)
#
# USAGE
#   python pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

# import json
# import os
# import time
# import hashlib
# import re
# from datetime import date, datetime
# from collections import defaultdict

# # ── Config ────────────────────────────────────────────────────────────────────
# INPUT_FILE            = "candidates.jsonl"
# DATASET_DIR           = "Dataset"
# TODAY                 = date(2026, 6, 17)
# FABRICATION_THRESHOLD = 100


# # ══════════════════════════════════════════════════════════════════════════════
# #  CONSTANTS
# # ══════════════════════════════════════════════════════════════════════════════

# POSTGRAD_DEGREES  = {"m.tech", "m.e.", "m.s.", "m.sc", "m.sc.", "ph.d", "mba", "m.b.a"}
# UNDERGRAD_DEGREES = {"b.tech", "b.e.", "b.sc", "b.sc.", "b.a.", "be", "btech", "b.eng"}
# ENG_DEGREE_SET    = {"b.tech", "m.tech", "b.e.", "m.e.", "be", "btech", "mtech", "b.eng"}
# INVALID_ENG_FIELDS = {"mba", "m.b.a", "commerce"}

# AI_SKILLS = {
#     "machine learning", "deep learning", "nlp", "computer vision",
#     "object detection", "image classification", "rag", "llms",
#     "fine-tuning llms", "vector search", "embeddings", "faiss",
#     "pinecone", "qdrant", "weaviate", "sentence transformers",
#     "hugging face transformers", "langchain", "llamaindex",
#     "diffusion models", "gans", "yolo", "cnn", "rnn", "lstm",
#     "transformers", "bert", "gpt", "reinforcement learning",
#     "recommendation systems", "information retrieval",
#     "semantic search", "prompt engineering", "mlops", "mlflow",
#     "weights & biases", "kubeflow", "bentoml",
#     "speech recognition", "tts", "asr", "forecasting",
#     "time series", "statistical modeling", "learning to rank",
#     "bm25", "opensearch", "haystack", "qlora", "pgvector",
#     "feature engineering", "data science", "opencv",
#     "scikit-learn", "xgboost", "lightgbm", "prophet",
#     "tensorflow", "pytorch", "keras"
# }

# AI_CAREER_KEYWORDS = {
#     "ml engineer", "machine learning", "data scientist", "ai specialist",
#     "ai engineer", "nlp engineer", "computer vision", "research engineer",
#     "deep learning", "junior ml", "senior ml", "ai research",
#     "applied scientist", "mlops engineer"
# }

# AI_INDUSTRIES_EXACT = {
#     "ai/ml", "ai", "ml", "machine learning", "artificial intelligence",
#     "edtech ai", "healthtech ai", "deep learning", "nlp"
# }

# DE_DOMAIN_SKILLS = {
#     "apache spark", "spark", "kafka", "airflow", "dbt", "hadoop",
#     "hive", "presto", "trino", "databricks", "snowflake", "redshift",
#     "bigquery", "data warehouse", "data lake", "etl", "elt",
#     "pipeline", "flink", "nifi", "luigi", "prefect", "dagster",
#     "glue", "fivetran", "stitch", "talend", "informatica",
#     "delta lake", "iceberg", "parquet", "avro", "pyspark"
# }

# INDIA_PREFERRED_CITIES = {
#     "noida", "pune", "bangalore", "bengaluru", "hyderabad",
#     "chennai", "mumbai", "delhi", "gurugram", "gurgaon",
#     "navi mumbai", "thane", "kolkata"
# }

# _SCORE_FIELDS = [
#     "skill_career_domain_mismatch", "education_overlap",
#     "reverse_degree_order", "second_undergrad_after_first",
#     "education_career_gap_flag", "active_before_signup",
#     "duplicate_job_descriptions", "all_descriptions_identical",
#     "low_engagement_flag", "possible_honeypot", "_any_invalid_degree_field",
# ]

# # ══════════════════════════════════════════════════════════════════════════════
# #  DOMAIN + ROLE TAXONOMY
# #
# #  Domain  : the functional area (what department they belong to)
# #  Role    : specific job type within that domain (becomes the filename)
# #
# #  Rules:
# #  - Checked in priority order — first match wins
# #  - Title is lowercased and stripped before matching
# #  - Match uses substring check ("ml engineer" in title)
# #  - Role slug is snake_case, used as the .json filename
# # ══════════════════════════════════════════════════════════════════════════════

# # Each entry: (domain, role_slug, [title_keywords_that_match])
# # Checked top-to-bottom — first match wins per candidate
# DOMAIN_ROLE_RULES = [

#     # ── DATA & AI ─────────────────────────────────────────────────────────────
#     ("data_and_ai", "ml_engineer",
#         ["ml engineer", "machine learning engineer", "junior ml", "senior ml"]),
#     ("data_and_ai", "data_scientist",
#         ["data scientist", "applied scientist"]),
#     ("data_and_ai", "ai_engineer",
#         ["ai engineer", "ai specialist", "ai research"]),
#     ("data_and_ai", "nlp_engineer",
#         ["nlp engineer"]),
#     ("data_and_ai", "computer_vision_engineer",
#         ["computer vision engineer", "computer vision"]),
#     ("data_and_ai", "deep_learning_engineer",
#         ["deep learning engineer"]),
#     ("data_and_ai", "research_engineer",
#         ["research engineer", "research scientist"]),
#     ("data_and_ai", "mlops_engineer",
#         ["mlops engineer", "mlops"]),
#     ("data_and_ai", "data_engineer",
#         ["data engineer", "analytics engineer"]),
#     ("data_and_ai", "data_architect",
#         ["data architect"]),
#     ("data_and_ai", "etl_developer",
#         ["etl developer", "spark engineer", "kafka engineer", "databricks engineer"]),
#     ("data_and_ai", "data_analyst",
#         ["data analyst", "business intelligence", "bi analyst"]),

#     # ── DEVOPS & CLOUD ────────────────────────────────────────────────────────
#     ("devops_and_cloud", "devops_engineer",
#         ["devops engineer"]),
#     ("devops_and_cloud", "cloud_engineer",
#         ["cloud engineer"]),
#     ("devops_and_cloud", "platform_engineer",
#         ["platform engineer"]),
#     ("devops_and_cloud", "site_reliability_engineer",
#         ["site reliability", "sre"]),
#     ("devops_and_cloud", "systems_engineer",
#         ["systems engineer"]),
#     ("devops_and_cloud", "network_engineer",
#         ["network engineer"]),
#     ("devops_and_cloud", "security_engineer",
#         ["security engineer", "cybersecurity"]),

#     # ── TECH (Software Engineering) ───────────────────────────────────────────
#     ("tech", "backend_engineer",
#         ["backend engineer", "backend developer"]),
#     ("tech", "frontend_engineer",
#         ["frontend engineer", "frontend developer"]),
#     ("tech", "full_stack_developer",
#         ["full stack", "fullstack"]),
#     ("tech", "mobile_developer",
#         ["mobile developer", "android developer", "ios developer",
#          "mobile engineer", "android engineer", "ios engineer"]),
#     ("tech", "software_engineer",
#         ["software engineer", "software developer", "sde", "sde-"]),
#     ("tech", "qa_engineer",
#         ["qa engineer", "quality assurance", "test engineer",
#          "automation engineer", "sdet"]),
#     ("tech", "net_developer",
#         [".net developer", ".net engineer"]),
#     ("tech", "embedded_engineer",
#         ["embedded engineer", "firmware engineer"]),

#     # ── PRODUCT & DESIGN ──────────────────────────────────────────────────────
#     ("product_and_design", "product_manager",
#         ["product manager", "associate product manager", "senior product manager"]),
#     ("product_and_design", "product_designer",
#         ["product designer"]),
#     ("product_and_design", "ux_designer",
#         ["ux designer", "user experience", "ux researcher"]),
#     ("product_and_design", "ui_designer",
#         ["ui designer", "user interface"]),
#     ("product_and_design", "graphic_designer",
#         ["graphic designer"]),
#     ("product_and_design", "brand_designer",
#         ["brand designer", "creative director", "visual designer"]),

#     # ── OPERATIONS ────────────────────────────────────────────────────────────
#     ("operations", "operations_manager",
#         ["operations manager", "ops manager"]),
#     ("operations", "supply_chain_manager",
#         ["supply chain", "procurement manager", "logistics manager"]),
#     ("operations", "project_manager",
#         ["project manager"]),
#     ("operations", "program_manager",
#         ["program manager"]),
#     ("operations", "scrum_master",
#         ["scrum master", "agile coach"]),
#     ("operations", "customer_support",
#         ["customer support", "customer success", "support engineer",
#          "customer service"]),

#     # ── BUSINESS ──────────────────────────────────────────────────────────────
#     ("business", "business_analyst",
#         ["business analyst"]),
#     ("business", "sales_executive",
#         ["sales executive", "sales manager", "account executive",
#          "account manager", "business development"]),
#     ("business", "consultant",
#         ["consultant", "strategy consultant", "management consultant"]),

#     # ── MARKETING ─────────────────────────────────────────────────────────────
#     ("marketing", "marketing_manager",
#         ["marketing manager"]),
#     ("marketing", "digital_marketing",
#         ["digital marketing", "performance marketing", "growth marketer",
#          "paid marketing"]),
#     ("marketing", "seo_specialist",
#         ["seo specialist", "seo manager", "seo"]),
#     ("marketing", "content_writer",
#         ["content writer", "content strategist", "copywriter",
#          "technical writer"]),
#     ("marketing", "social_media_manager",
#         ["social media manager", "social media"]),

#     # ── FINANCE ───────────────────────────────────────────────────────────────
#     ("finance", "finance_manager",
#         ["finance manager", "financial controller", "cfo"]),
#     ("finance", "financial_analyst",
#         ["financial analyst", "investment analyst"]),
#     ("finance", "accountant",
#         ["accountant", "senior accountant", "staff accountant"]),
#     ("finance", "tax_consultant",
#         ["tax consultant", "tax manager", "tax analyst"]),
#     ("finance", "auditor",
#         ["auditor", "internal auditor", "external auditor"]),

#     # ── HR & PEOPLE ───────────────────────────────────────────────────────────
#     ("hr_and_people", "hr_manager",
#         ["hr manager", "human resources manager", "people manager"]),
#     ("hr_and_people", "recruiter",
#         ["recruiter", "talent acquisition", "technical recruiter",
#          "sourcing specialist"]),
#     ("hr_and_people", "learning_and_development",
#         ["learning and development", "l&d", "training manager"]),
#     ("hr_and_people", "hr_business_partner",
#         ["hr business partner", "hrbp"]),

#     # ── ENGINEERING (non-tech / physical) ────────────────────────────────────
#     ("engineering_non_tech", "mechanical_engineer",
#         ["mechanical engineer"]),
#     ("engineering_non_tech", "civil_engineer",
#         ["civil engineer"]),
#     ("engineering_non_tech", "electrical_engineer",
#         ["electrical engineer"]),
#     ("engineering_non_tech", "chemical_engineer",
#         ["chemical engineer"]),
#     ("engineering_non_tech", "structural_engineer",
#         ["structural engineer"]),
#     ("engineering_non_tech", "manufacturing_engineer",
#         ["manufacturing engineer"]),
#     ("engineering_non_tech", "quality_engineer",
#         ["quality engineer", "quality manager", "qc engineer"]),
#     ("engineering_non_tech", "biomedical_engineer",
#         ["biomedical engineer"]),
# ]

# # Fallback slug when no rule matches
# OTHER_DOMAIN    = "other"
# OTHER_ROLE_SLUG = "unclassified"


# # ══════════════════════════════════════════════════════════════════════════════
# #  SHARED HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def _s(val) -> str:
#     return (val or "").lower().strip()


# def _parse_date(s) -> date | None:
#     if not s:
#         return None
#     try:
#         return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
#     except Exception:
#         return None


# def _has_ai_career(career: list) -> bool:
#     for job in career:
#         title = _s(job.get("title"))
#         ind   = _s(job.get("industry"))
#         if any(kw in title for kw in AI_CAREER_KEYWORDS):
#             return True
#         if ind in AI_INDUSTRIES_EXACT:
#             return True
#         if "machine learning" in ind or "artificial intelligence" in ind:
#             return True
#     return False


# def _md5(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()


# # ══════════════════════════════════════════════════════════════════════════════
# #  FIXES  (F1, F2, F3)
# # ══════════════════════════════════════════════════════════════════════════════

# def _fix_salary(sig):
#     sal = sig.setdefault("expected_salary_range_inr_lpa", {})
#     mn, mx = sal.get("min"), sal.get("max")
#     if mn is not None and mx is not None and mn > mx:
#         sal["min"], sal["max"] = mx, mn
#         sal["salary_was_inverted"] = True
#     else:
#         sal["salary_was_inverted"] = False


# def _fix_github(sig):
#     score = sig.get("github_activity_score")
#     if score == -1 or score is None:
#         sig["github_activity_score"] = None
#         sig["github_not_linked"]     = True
#     else:
#         sig["github_not_linked"] = False


# def _fix_offer(sig):
#     if sig.get("offer_acceptance_rate") == -1:
#         sig["offer_acceptance_rate"] = None
#         sig["no_offer_history"]      = True
#     else:
#         sig["no_offer_history"] = False


# def apply_fixes(c):
#     sig = c.setdefault("redrob_signals", {})
#     _fix_salary(sig)
#     _fix_github(sig)
#     _fix_offer(sig)
#     sal = sig.get("expected_salary_range_inr_lpa", {})
#     c["salary_was_inverted"] = sal.get("salary_was_inverted", False)
#     c["github_not_linked"]   = sig.get("github_not_linked",   False)
#     c["no_offer_history"]    = sig.get("no_offer_history",    False)


# # ══════════════════════════════════════════════════════════════════════════════
# #  FLAGS
# # ══════════════════════════════════════════════════════════════════════════════

# def fl3_skill_career_domain_mismatch(c):
#     skills   = c.get("skills", [])
#     career   = c.get("career_history", [])
#     ai_count = sum(1 for sk in skills if _s(sk.get("name")) in AI_SKILLS)
#     if ai_count < 3:
#         return False
#     return not _has_ai_career(career)


# def fl5_education_overlap(edu_list):
#     pairs = []
#     for i in range(len(edu_list)):
#         for j in range(i + 1, len(edu_list)):
#             s1 = edu_list[i].get("start_year")
#             e1 = edu_list[i].get("end_year")
#             s2 = edu_list[j].get("start_year")
#             e2 = edu_list[j].get("end_year")
#             if None in (s1, e1, s2, e2):
#                 continue
#             if max(s1, s2) < min(e1, e2):
#                 pairs.append([i, j])
#     return bool(pairs), pairs


# def fl6_reverse_degree_order(edu_list):
#     pg_ends, ug_starts = [], []
#     for e in edu_list:
#         d  = _s(e.get("degree"))
#         ey = e.get("end_year")
#         sy = e.get("start_year")
#         if d in POSTGRAD_DEGREES  and ey is not None: pg_ends.append(ey)
#         if d in UNDERGRAD_DEGREES and sy is not None: ug_starts.append(sy)
#     if not pg_ends or not ug_starts:
#         return False
#     return min(pg_ends) <= max(ug_starts)


# def fl7_second_undergrad_after_first(edu_list):
#     ugs = sorted(
#         [e for e in edu_list
#          if _s(e.get("degree")) in UNDERGRAD_DEGREES
#          and e.get("start_year") is not None
#          and e.get("end_year")   is not None],
#         key=lambda e: e["start_year"]
#     )
#     if len(ugs) < 2:
#         return False
#     return ugs[1]["start_year"] >= ugs[0]["end_year"] + 2


# def fl8_education_career_gap(c):
#     edu_ends = [
#         e["end_year"] for e in c.get("education", [])
#         if e.get("end_year") is not None
#     ]
#     career_starts = [
#         _parse_date(j.get("start_date")).year
#         for j in c.get("career_history", [])
#         if _parse_date(j.get("start_date"))
#     ]
#     if not edu_ends or not career_starts:
#         return False, 0.0
#     gap = float(min(career_starts) - max(edu_ends))
#     return gap > 1, round(gap, 1)


# def fl11_active_before_signup(c):
#     sig    = c.get("redrob_signals", {})
#     signup = _parse_date(sig.get("signup_date"))
#     last   = _parse_date(sig.get("last_active_date"))
#     if signup is None or last is None:
#         return False, 0
#     if last < signup:
#         return True, (signup - last).days
#     return False, 0


# def fl16_invalid_degree_field(edu):
#     deg   = _s(edu.get("degree"))
#     field = _s(edu.get("field_of_study"))
#     if deg in ENG_DEGREE_SET and field in INVALID_ENG_FIELDS:
#         return True
#     if deg in {"m.sc", "m.sc.", "m.s.", "m.s"} and field in INVALID_ENG_FIELDS:
#         return True
#     return False


# def fl18_fl19_duplicate_descriptions(career):
#     if len(career) < 2:
#         return False, False, []
#     descs = [(j.get("description") or "").strip() for j in career]
#     pairs = [
#         [i, j] for i in range(len(descs))
#         for j in range(i + 1, len(descs))
#         if descs[i] and descs[i] == descs[j]
#     ]
#     non_empty = [d for d in descs if d]
#     all_ident  = len(non_empty) > 1 and len(set(non_empty)) == 1
#     return bool(pairs), all_ident, pairs


# def fl20_low_engagement(c):
#     sig  = c.get("redrob_signals", {})
#     rate = sig.get("recruiter_response_rate")
#     hrs  = sig.get("avg_response_time_hours")
#     if rate is None or hrs is None:
#         return False
#     return rate < 0.10 and hrs > 200


# def fl25_possible_honeypot(c):
#     structural = sum([
#         bool(c.get("education_overlap")),
#         bool(c.get("reverse_degree_order")),
#         bool(c.get("second_undergrad_after_first")),
#         bool(c.get("all_descriptions_identical")),
#         bool(c.get("duplicate_job_descriptions")),
#     ])
#     return structural >= 3


# def fl26_honeypot_score(c):
#     return sum(1 for f in _SCORE_FIELDS if c.get(f, False))


# # ══════════════════════════════════════════════════════════════════════════════
# #  MASTER CLEAN FUNCTION
# # ══════════════════════════════════════════════════════════════════════════════

# def process_candidate(c):
#     edu_list = c.get("education", [])
#     career   = c.get("career_history", [])

#     apply_fixes(c)

#     c["skill_career_domain_mismatch"] = fl3_skill_career_domain_mismatch(c)

#     ov_flag, ov_pairs = fl5_education_overlap(edu_list)
#     c["education_overlap"]             = ov_flag
#     c["overlapping_education_indices"] = ov_pairs

#     c["reverse_degree_order"]         = fl6_reverse_degree_order(edu_list)
#     c["second_undergrad_after_first"] = fl7_second_undergrad_after_first(edu_list)

#     gap_flag, gap_val = fl8_education_career_gap(c)
#     c["education_career_gap_flag"]    = gap_flag
#     c["education_career_gap_years"]   = gap_val

#     abs_flag, abs_days = fl11_active_before_signup(c)
#     c["active_before_signup"]         = abs_flag
#     c["signup_active_gap_days"]       = abs_days

#     dup_any, all_ident, dup_idx = fl18_fl19_duplicate_descriptions(career)
#     c["duplicate_job_descriptions"]    = dup_any
#     c["all_descriptions_identical"]    = all_ident
#     c["duplicate_description_indices"] = dup_idx

#     c["low_engagement_flag"] = fl20_low_engagement(c)

#     any_invalid = False
#     for edu in edu_list:
#         val = fl16_invalid_degree_field(edu)
#         edu["invalid_degree_field_combination"] = val
#         if val:
#             any_invalid = True
#     c["_any_invalid_degree_field"] = any_invalid

#     c["possible_honeypot"] = fl25_possible_honeypot(c)
#     c["honeypot_score"]    = fl26_honeypot_score(c)

#     # fabrication — stamped later after full freq table is built
#     c["fabrication_bandwidth"] = 0
#     c["possible_fabrication"]  = False

#     return c


# # ══════════════════════════════════════════════════════════════════════════════
# #  DESCRIPTION FREQUENCY
# # ══════════════════════════════════════════════════════════════════════════════

# def collect_desc_freq(freq, c):
#     for job in c.get("career_history", []):
#         desc = (job.get("description") or "").strip()
#         if desc:
#             h = _md5(desc)
#             freq[h] = freq.get(h, 0) + 1


# def stamp_fabrication(c, freq):
#     total = sum(
#         freq.get(_md5((job.get("description") or "").strip()), 0)
#         for job in c.get("career_history", [])
#         if (job.get("description") or "").strip()
#     )
#     c["fabrication_bandwidth"] = total
#     c["possible_fabrication"]  = total > FABRICATION_THRESHOLD


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLASSIFICATION  (5-dimension tree routing)
# # ══════════════════════════════════════════════════════════════════════════════

# def _availability_bucket(c):
#     last = _parse_date(c.get("redrob_signals", {}).get("last_active_date"))
#     if last is None:
#         return "inactive"
#     days = (TODAY - last).days
#     if days <= 30:  return "active"
#     if days <= 90:  return "moderate"
#     return "inactive"


# def _country_bucket(c):
#     profile  = c.get("profile", {})
#     country  = _s(profile.get("country"))
#     location = _s(profile.get("location"))
#     if country in ("india", "in"):
#         city = location.split(",")[0].strip()
#         if any(pref in city for pref in INDIA_PREFERRED_CITIES):
#             return "india_preferred"
#         return "india_other"
#     return "outside_india"


# def _experience_band(c):
#     yoe = c.get("profile", {}).get("years_of_experience")
#     if yoe is None: return "0_to_4"
#     if yoe < 5:     return "0_to_4"
#     if yoe <= 9:    return "5_to_9"
#     return "10_plus"


# def _domain_and_role(c):
#     """
#     Returns (domain_slug, role_slug) by matching current_title against
#     DOMAIN_ROLE_RULES in priority order.
#     First match wins.
#     Falls back to (other, unclassified) if nothing matches.
#     """
#     title = _s(c.get("profile", {}).get("current_title", ""))
#     for domain, role_slug, keywords in DOMAIN_ROLE_RULES:
#         if any(kw in title for kw in keywords):
#             return domain, role_slug
#     return OTHER_DOMAIN, OTHER_ROLE_SLUG


# def classify(c):
#     domain, role = _domain_and_role(c)
#     return {
#         "availability": _availability_bucket(c),
#         "country":      _country_bucket(c),
#         "experience":   _experience_band(c),
#         "domain":       domain,
#         "role":         role,
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# #  TREE BUILDER
# #
# #  Leaf path:
# #    Dataset/{availability}/{country}/{experience}/{domain}/{role}.json
# #
# #  Each leaf file is a JSON array of all candidates in that bucket.
# # ══════════════════════════════════════════════════════════════════════════════

# def _leaf_path(dataset_dir, buckets):
#     return os.path.join(
#         dataset_dir,
#         buckets["availability"],
#         buckets["country"],
#         buckets["experience"],
#         buckets["domain"],
#         buckets["role"] + ".json"      # role becomes the filename
#     )


# def build_tree(all_candidates, dataset_dir):
#     grouped = defaultdict(list)
#     for c in all_candidates:
#         path = _leaf_path(dataset_dir, classify(c))
#         grouped[path].append(c)

#     for path, candidates in grouped.items():
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "w", encoding="utf-8") as fh:
#             json.dump(candidates, fh, ensure_ascii=False, indent=2)

#     return len(grouped)


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def run():
#     if not os.path.exists(INPUT_FILE):
#         print(f"❌  File not found: {INPUT_FILE}")
#         return

#     wall_start = time.time()
#     desc_freq  = {}

#     # ── Pass 1: read + clean + collect description frequencies ────────────────
#     print(f"📂  Reading and cleaning {INPUT_FILE} ...")
#     cleaned   = []
#     skipped   = 0
#     honeypots = 0

#     with open(INPUT_FILE, "r", encoding="utf-8") as fh:
#         for i, line in enumerate(fh, 1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 raw = json.loads(line)
#             except json.JSONDecodeError as e:
#                 print(f"    ⚠  Skipped line {i}: {e}")
#                 skipped += 1
#                 continue

#             collect_desc_freq(desc_freq, raw)
#             c = process_candidate(raw)
#             cleaned.append(c)
#             if c.get("possible_honeypot"):
#                 honeypots += 1

#             if i % 10_000 == 0:
#                 elapsed = time.time() - wall_start
#                 print(f"    ... {i:,} processed  ({elapsed:.1f}s elapsed)")

#     print(f"    ✅ {len(cleaned):,} cleaned  |  {skipped} skipped  |  "
#           f"{honeypots:,} honeypots  |  {time.time()-wall_start:.1f}s")

#     # ── Pass 2: stamp fabrication flags ───────────────────────────────────────
#     print(f"\n🔖  Stamping fabrication flags ...")
#     t = time.time()
#     fabricated = 0
#     for c in cleaned:
#         stamp_fabrication(c, desc_freq)
#         if c["possible_fabrication"]:
#             fabricated += 1
#     print(f"    ✅ possible_fabrication=True : {fabricated:,}  |  {time.time()-t:.1f}s")

#     # ── Pass 3: build tree ────────────────────────────────────────────────────
#     print(f"\n🌳  Building Dataset/ tree ...")
#     t = time.time()
#     buckets_written = build_tree(cleaned, DATASET_DIR)
#     print(f"    ✅ {buckets_written} leaf files written  |  {time.time()-t:.1f}s")

#     total_time = time.time() - wall_start

#     print()
#     print("=" * 58)
#     print("  PIPELINE COMPLETE")
#     print("=" * 58)
#     print(f"  Candidates loaded        : {len(cleaned) + skipped:,}")
#     print(f"  Candidates cleaned       : {len(cleaned):,}")
#     print(f"  Skipped (malformed)      : {skipped}")
#     print(f"  Honeypots flagged        : {honeypots:,}  "
#           f"({100*honeypots/max(len(cleaned),1):.1f}%)")
#     print(f"  Possible fabrication     : {fabricated:,}  "
#           f"({100*fabricated/max(len(cleaned),1):.1f}%)")
#     print(f"  Unique descriptions      : {len(desc_freq):,}")
#     print(f"  Leaf files written       : {buckets_written}")
#     print(f"  Output folder            : {os.path.abspath(DATASET_DIR)}/")
#     print(f"  Total wall time          : {total_time:.1f}s")
#     print()


# if __name__ == "__main__":
#     run()








#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################









# pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Single-pass pipeline.
#   Reads candidates.jsonl → cleans (fixes + flags) → stamps fabrication
#   → classifies into 6-dimension tree → writes Dataset/ folder tree
#
# TREE STRUCTURE
#   Dataset/
#   └── {code_status}        code | no_code
#       └── {availability}   active_available | inactive
#           └── {relocation} relocation_ready | standard
#               └── {experience} 0_to_3 | 4_to_9 | 10_plus
#                   └── {domain}  engineering | devops_and_cloud |
#                                  product_and_design | operations |
#                                  business | marketing | finance |
#                                  hr_and_people | non_tech_engineering | other
#                       └── {role}.json
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
#   python pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import time
import hashlib
from datetime import date, datetime
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE            = "candidates.jsonl"
DATASET_DIR           = "Dataset"
TODAY                 = date(2026, 6, 17)
FABRICATION_THRESHOLD = 100


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

DE_DOMAIN_SKILLS = {
    "apache spark", "spark", "kafka", "airflow", "dbt", "hadoop",
    "hive", "presto", "trino", "databricks", "snowflake", "redshift",
    "bigquery", "data warehouse", "data lake", "etl", "elt",
    "pipeline", "flink", "nifi", "luigi", "prefect", "dagster",
    "glue", "fivetran", "stitch", "talend", "informatica",
    "delta lake", "iceberg", "parquet", "avro", "pyspark"
}

INDIA_PREFERRED_CITIES = {
    "noida", "pune", "bangalore", "bengaluru", "hyderabad",
    "chennai", "mumbai", "delhi", "gurugram", "gurgaon",
    "navi mumbai", "thane", "kolkata"
}

_SCORE_FIELDS = [
    "skill_career_domain_mismatch", "education_overlap",
    "reverse_degree_order", "second_undergrad_after_first",
    "education_career_gap_flag", "active_before_signup",
    "duplicate_job_descriptions", "all_descriptions_identical",
    "low_engagement_flag", "possible_honeypot", "_any_invalid_degree_field",
]

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

def fl18_fl19_duplicate_descriptions(career):
    if len(career) < 2: return False, False, []
    descs = [(j.get("description") or "").strip() for j in career]
    pairs = [
        [i, j] for i in range(len(descs))
        for j in range(i + 1, len(descs))
        if descs[i] and descs[i] == descs[j]
    ]
    non_empty = [d for d in descs if d]
    all_ident  = len(non_empty) > 1 and len(set(non_empty)) == 1
    return bool(pairs), all_ident, pairs

def fl20_low_engagement(c):
    sig  = c.get("redrob_signals", {})
    rate = sig.get("recruiter_response_rate")
    hrs  = sig.get("avg_response_time_hours")
    if rate is None or hrs is None: return False
    return rate < 0.10 and hrs > 200

def fl25_possible_honeypot(c):
    structural = sum([
        bool(c.get("education_overlap")),
        bool(c.get("reverse_degree_order")),
        bool(c.get("second_undergrad_after_first")),
        bool(c.get("all_descriptions_identical")),
        bool(c.get("duplicate_job_descriptions")),
    ])
    return structural >= 3

def fl26_honeypot_score(c):
    return sum(1 for f in _SCORE_FIELDS if c.get(f, False))


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

    dup_any, all_ident, dup_idx = fl18_fl19_duplicate_descriptions(career)
    c["duplicate_job_descriptions"]    = dup_any
    c["all_descriptions_identical"]    = all_ident
    c["duplicate_description_indices"] = dup_idx

    c["low_engagement_flag"] = fl20_low_engagement(c)

    any_invalid = False
    for edu in edu_list:
        val = fl16_invalid_degree_field(edu)
        edu["invalid_degree_field_combination"] = val
        if val: any_invalid = True
    c["_any_invalid_degree_field"] = any_invalid

    c["possible_honeypot"] = fl25_possible_honeypot(c)
    c["honeypot_score"]    = fl26_honeypot_score(c)

    # fabrication — stamped after full freq table built
    c["fabrication_bandwidth"] = 0
    c["possible_fabrication"]  = False

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

def stamp_fabrication(c, freq):
    total = sum(
        freq.get(_md5((job.get("description") or "").strip()), 0)
        for job in c.get("career_history", [])
        if (job.get("description") or "").strip()
    )
    c["fabrication_bandwidth"] = total
    c["possible_fabrication"]  = total > FABRICATION_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION  (6-dimension tree routing)
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
    active_available: open_to_work=True OR last active within 90 days
    inactive:         open_to_work=False AND last active > 90 days
    """
    sig          = c.get("redrob_signals", {})
    open_to_work = sig.get("open_to_work_flag", False)
    last         = _parse_date(sig.get("last_active_date"))

    if open_to_work:
        return "active_available"
    if last is not None and (TODAY - last).days <= 90:
        return "active_available"
    return "inactive"


def _relocation_bucket(c) -> str:
    """
    relocation_ready: willing_to_relocate=True AND notice_period_days <= 30
    standard:         everything else
    """
    sig    = c.get("redrob_signals", {})
    willing = sig.get("willing_to_relocate", False)
    notice  = sig.get("notice_period_days") or 999
    if willing and notice <= 30:
        return "relocation_ready"
    return "standard"


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
        "relocation":   _relocation_bucket(c),
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
        b["relocation"],
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
    honeypots = 0

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
            if c.get("possible_honeypot"): honeypots += 1

            if i % 10_000 == 0:
                elapsed = time.time() - wall_start
                print(f"    ... {i:,} processed  ({elapsed:.1f}s elapsed)")

    t1 = time.time() - wall_start
    print(f"    ✅ {len(cleaned):,} cleaned  |  {skipped} skipped  |  "
          f"{honeypots:,} honeypots  |  {t1:.1f}s")

    # ── Pass 2: stamp fabrication flags ───────────────────────────────────────
    print(f"\n🔖  Stamping fabrication flags ...")
    t = time.time()
    fabricated = 0
    for c in cleaned:
        stamp_fabrication(c, desc_freq)
        if c["possible_fabrication"]: fabricated += 1
    print(f"    ✅ possible_fabrication=True : {fabricated:,}  |  {time.time()-t:.1f}s")

    # ── Pass 3: build tree ────────────────────────────────────────────────────
    print(f"\n🌳  Building Dataset/ tree ...")
    t = time.time()
    buckets_written = build_tree(cleaned, DATASET_DIR)
    print(f"    ✅ {buckets_written} leaf files written  |  {time.time()-t:.1f}s")

    # ── Compute summary counts ─────────────────────────────────────────────────
    total         = len(cleaned)
    code_count    = sum(1 for c in cleaned if _code_status(c)        == "code")
    no_code_count = total - code_count
    active_count  = sum(1 for c in cleaned if _availability_bucket(c) == "active_available")
    reloc_count   = sum(1 for c in cleaned if _relocation_bucket(c)   == "relocation_ready")
    pure_cv_count = sum(1 for c in cleaned if _is_pure_cv(c))
    total_time    = time.time() - wall_start

    print()
    print("=" * 62)
    print("  PIPELINE COMPLETE")
    print("=" * 62)
    print(f"  Candidates loaded        : {total + skipped:,}")
    print(f"  Candidates cleaned       : {total:,}")
    print(f"  Skipped (malformed)      : {skipped}")
    print(f"  ── Quality ────────────────────────────────────────────")
    print(f"  Honeypots (possible)     : {honeypots:,}  ({100*honeypots/max(total,1):.1f}%)")
    print(f"  Possible fabrication     : {fabricated:,}  ({100*fabricated/max(total,1):.1f}%)")
    print(f"  ── Classification ─────────────────────────────────────")
    print(f"  code                     : {code_count:,}  ({100*code_count/max(total,1):.1f}%)")
    print(f"  no_code                  : {no_code_count:,}  ({100*no_code_count/max(total,1):.1f}%)")
    print(f"  active_available         : {active_count:,}  ({100*active_count/max(total,1):.1f}%)")
    print(f"  inactive                 : {total-active_count:,}  ({100*(total-active_count)/max(total,1):.1f}%)")
    print(f"  relocation_ready         : {reloc_count:,}  ({100*reloc_count/max(total,1):.1f}%)")
    print(f"  standard                 : {total-reloc_count:,}  ({100*(total-reloc_count)/max(total,1):.1f}%)")
    print(f"  pure CV (no retrieval)   : {pure_cv_count:,}  ({100*pure_cv_count/max(total,1):.1f}%)")
    print(f"  ── Output ─────────────────────────────────────────────")
    print(f"  Unique descriptions      : {len(desc_freq):,}")
    print(f"  Leaf files written       : {buckets_written}")
    print(f"  Output folder            : {os.path.abspath(DATASET_DIR)}/")
    print(f"  ── Performance ────────────────────────────────────────")
    print(f"  Total wall time          : {total_time:.1f}s")
    print()


if __name__ == "__main__":
    run()