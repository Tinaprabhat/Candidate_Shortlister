#!/usr/bin/env python3
"""
build_kb.py — One-time artifact builder (run BEFORE submission).

This downloads, quantizes, and compresses every model the pipeline needs,
plus builds the SQLite fraud knowledge base. It may exceed the 5-minute
ranking budget — that is fine. This is pre-computation, NOT the ranking step.

Outputs land in:
  models/decompressed/   (ready-to-load artifacts)
  models/compressed/     (.tar.gz archives committed to Git)

Usage:
    python build_kb.py --all
    python build_kb.py --sentence-transformer --flashrank   # selective

After running, commit models/compressed/*.tar.gz (via Git LFS).
On a fresh clone, setup.sh decompresses them back into models/decompressed/.
"""

import os
import sys
import json
import math
import time
import tarfile
import hashlib
import sqlite3
import argparse
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_kb")

ROOT = Path(__file__).parent
DECOMP = ROOT / "models" / "decompressed"
COMP = ROOT / "models" / "compressed"
DECOMP.mkdir(parents=True, exist_ok=True)
COMP.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. SENTENCE-TRANSFORMER (bi-encoder, L2/L4) → ONNX INT8
# ──────────────────────────────────────────────────────────────────────────────
def build_sentence_transformer():
    logger.info("Building sentence-transformer (ONNX INT8)...")
    out_dir = DECOMP / "sentence_transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from sentence_transformers import SentenceTransformer
        # Download + save in ST format (works on CPU offline after this).
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        model.save(str(out_dir))
        # Optional ONNX INT8 export via optimum (if installed)
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            from optimum.onnxruntime import ORTQuantizer
            onnx_dir = out_dir / "onnx"
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", export=True)
            ort_model.save_pretrained(onnx_dir)
            quantizer = ORTQuantizer.from_pretrained(onnx_dir)
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=onnx_dir, quantization_config=qconfig)
            logger.info("  ONNX INT8 export complete")
        except Exception as e:
            logger.warning(f"  ONNX INT8 export skipped ({e}); ST format saved (still CPU-fast)")
        _compress("sentence_transformer")
        return True
    except Exception as e:
        logger.error(f"sentence-transformer build failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 2. FLASHRANK (cross-encoder, L7) → cached locally
# ──────────────────────────────────────────────────────────────────────────────
def build_flashrank():
    logger.info("Building FlashRank cross-encoder...")
    out_dir = DECOMP / "flashrank_onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from flashrank import Ranker
        # Ranker downloads the ONNX model into cache_dir; we point it at our dir.
        Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(out_dir.parent))
        _compress("flashrank_onnx")
        return True
    except Exception as e:
        logger.error(f"FlashRank build failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 3. spaCy (L5 helper)
# ──────────────────────────────────────────────────────────────────────────────
def build_spacy():
    logger.info("Building spaCy model...")
    out_dir = DECOMP / "spacy_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import spacy
        from spacy.cli import download as spacy_download
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy_download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        nlp.to_disk(out_dir / "en_core_web_sm")
        _compress("spacy_model")
        return True
    except Exception as e:
        logger.error(f"spaCy build failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 4. KenLM (L5 perplexity) — 5-gram model from curated tech-resume corpus
# ──────────────────────────────────────────────────────────────────────────────

# ~250 diverse tech-resume sentences covering action verbs, tools, roles, and
# architecture patterns.  Variety matters more than quantity for sparse n-gram
# coverage, so sentences deliberately mix different sentence structures.
_TECH_CORPUS = [
    # ── engineering & implementation ─────────────────────────────────────────
    "developed machine learning pipeline using python and pytorch for real-time model inference",
    "implemented retrieval augmented generation system using faiss vector database and openai embeddings",
    "designed and built microservices architecture using docker and kubernetes on aws eks",
    "built distributed data processing pipeline using apache spark and kafka streaming",
    "developed rest api backend using python fastapi and postgresql relational database",
    "implemented continuous integration and deployment pipeline using github actions and docker",
    "built real-time recommendation engine using deep learning models and redis caching layer",
    "designed scalable data ingestion pipeline processing millions of events per day using kafka",
    "developed semantic search system using sentence transformers and faiss index for low latency",
    "implemented bi-encoder and cross-encoder reranking pipeline for candidate matching at scale",
    "built fine-tuning infrastructure for large language models using huggingface transformers library",
    "designed end-to-end machine learning training and inference platform on google cloud vertex ai",
    "developed python microservice for document parsing and embedding generation using spacy nlp",
    "implemented model serving infrastructure using kubernetes and prometheus for monitoring",
    "built automated testing framework for machine learning models with evaluation metrics tracking",
    "designed distributed caching layer using redis and memcached to reduce database query latency",
    "developed graphql api layer on top of postgresql and mongodb databases using node and typescript",
    "implemented data validation and preprocessing pipeline using python pandas and great expectations",
    "built anomaly detection system using isolation forest and autoencoder neural network models",
    "designed high throughput api gateway handling one hundred thousand requests per second",
    "developed llm-powered document summarization and information extraction system using langchain",
    "implemented vector similarity search using pinecone and weaviate for semantic document retrieval",
    "built real-time fraud detection system using gradient boosting and streaming feature engineering",
    "designed multi-tenant saas platform on aws with rds postgresql and elasticache redis",
    "developed mobile backend api using golang and grpc for high performance low latency responses",
    "implemented automated model retraining pipeline using airflow and mlflow experiment tracking",
    "built sparse and dense hybrid retrieval system combining bm25 and neural embedding models",
    "designed kafka-based event streaming architecture for real-time analytics and data processing",
    "developed fine-tuned bert model for named entity recognition achieving ninety-two percent f1 score",
    "implemented rag pipeline with reranking using cross-encoder to improve answer relevance scores",
    # ── infrastructure & devops ───────────────────────────────────────────────
    "deployed containerized applications using docker compose and kubernetes helm charts on gcp",
    "built ci cd pipeline using jenkins github actions and terraform for infrastructure as code",
    "automated infrastructure provisioning using terraform and ansible across aws and azure cloud",
    "implemented observability stack using prometheus grafana and datadog for production monitoring",
    "designed blue green deployment strategy to achieve zero downtime releases for backend services",
    "built kubernetes operator to automate scaling and lifecycle management of machine learning workloads",
    "migrated monolithic application to microservices architecture reducing deployment time by sixty percent",
    "implemented gitops workflow using argocd for continuous deployment of kubernetes applications",
    "built data lake architecture on aws s3 with glue etl and athena for sql query access",
    "designed multi-region active-active deployment on aws eks with route53 health-based routing",
    "automated database schema migrations using flyway integrated into ci cd deployment pipeline",
    "implemented distributed tracing using opentelemetry and jaeger across microservices architecture",
    "built cost optimization system for aws infrastructure reducing monthly cloud spend by thirty percent",
    "designed service mesh using istio for traffic management security and observability on kubernetes",
    "implemented log aggregation pipeline using elasticsearch logstash and kibana elk stack",
    # ── ml and ai research ───────────────────────────────────────────────────
    "researched and implemented novel attention mechanism improving transformer model performance",
    "fine-tuned large language model using lora and qlora achieving state-of-the-art results on benchmark",
    "designed reinforcement learning from human feedback pipeline for llm alignment and safety",
    "developed multi-modal model combining vision transformer and language model for image captioning",
    "implemented knowledge distillation to compress large neural network into smaller efficient model",
    "evaluated and benchmarked embedding models for semantic search using ndcg and mrr metrics",
    "researched retrieval augmented generation approaches and published findings at acl conference",
    "built automated hyperparameter optimization system using optuna and ray tune frameworks",
    "designed data annotation pipeline and labeling workflow for ten million training examples",
    "implemented model quantization using int8 and fp16 precision to reduce inference latency by half",
    "developed custom loss function and training objective for ranking and relevance scoring tasks",
    "analyzed model bias and implemented fairness-aware training using adversarial debiasing methods",
    "built offline evaluation framework for recommendation system using held-out test data and ab testing",
    "implemented continual learning system to update model weights incrementally without catastrophic forgetting",
    "designed neural architecture search pipeline to automate model design for edge deployment constraints",
    "developed data-efficient few-shot learning model achieving strong performance with limited labeled data",
    "built explainability tools using shap and lime to interpret machine learning model predictions",
    "implemented active learning loop to prioritize annotation of most informative training examples",
    "researched dense passage retrieval approaches for open-domain question answering systems",
    "designed multi-task learning architecture sharing representations across related nlp tasks",
    # ── databases & backend ───────────────────────────────────────────────────
    "optimized postgresql database queries reducing average response latency from two hundred to ten milliseconds",
    "designed database schema and indexing strategy for high-write transactional workload on postgres",
    "implemented read replicas and connection pooling using pgbouncer to scale postgresql database",
    "built full-text search functionality using elasticsearch with custom analyzers and relevance tuning",
    "designed event sourcing and cqrs architecture using postgresql and redis for audit trail",
    "implemented database sharding strategy to horizontally scale mongodb across multiple nodes",
    "built time-series data storage and query layer using influxdb and grafana visualization",
    "optimized elasticsearch index settings and query performance for billion-document search corpus",
    "designed change data capture pipeline using debezium and kafka for real-time database replication",
    "implemented redis-based session management and rate limiting for high-traffic api endpoints",
    "built graph database layer using neo4j for relationship traversal in recommendation system",
    "designed snowflake data warehouse schema with materialized views for analytics dashboard",
    "implemented database connection pooling and query caching to reduce load on primary database",
    "built cassandra cluster for high-availability time-series data storage across multiple data centers",
    "optimized snowflake sql queries and clustering keys reducing compute cost by forty percent",
    # ── leadership & collaboration ────────────────────────────────────────────
    "led team of eight engineers to deliver large-scale distributed platform on schedule and budget",
    "mentored junior and mid-level engineers through code review pair programming and technical guidance",
    "collaborated with product managers and designers to define technical requirements and roadmap",
    "drove alignment across cross-functional teams including engineering data science and operations",
    "led technical design review process establishing coding standards and architecture guidelines",
    "coordinated with stakeholders to gather requirements prioritize features and communicate progress",
    "spearheaded migration from on-premise infrastructure to aws cloud saving two million annually",
    "managed engineering team hiring process conducting technical interviews and evaluating candidates",
    "facilitated agile sprint planning retrospectives and daily standups for distributed engineering team",
    "established machine learning platform team and defined roadmap for internal ml tooling",
    "led incident response and postmortem process improving mean time to resolution by fifty percent",
    "drove adoption of engineering best practices including testing documentation and code review",
    "mentored three junior engineers who were promoted to senior level within eighteen months",
    "collaborated with research team to transfer llm fine-tuning methods from research to production",
    "led technical due diligence for two acquisitions evaluating codebase scalability and architecture",
    # ── product & delivery ────────────────────────────────────────────────────
    "shipped production machine learning features serving ten million users with ninety-nine percent uptime",
    "delivered end-to-end candidate ranking system reducing time to shortlist by seventy percent",
    "launched semantic job matching platform processing five hundred thousand job applications daily",
    "built and shipped ios and android mobile application using react native and typescript",
    "delivered internal developer platform reducing new service setup time from days to minutes",
    "shipped real-time analytics dashboard using react and websocket for live data visualization",
    "launched a b testing infrastructure to evaluate product changes across millions of users",
    "delivered data pipeline processing and indexing one hundred million documents for search",
    "shipped automated resume screening tool improving recruiter efficiency by three hundred percent",
    "launched personalized email recommendation engine increasing user engagement by twenty-five percent",
    # ── security & reliability ────────────────────────────────────────────────
    "implemented oauth2 and jwt authentication system for secure api access and user sessions",
    "designed zero-trust security architecture with mutual tls between internal microservices",
    "conducted security audit and penetration testing identifying and remediating critical vulnerabilities",
    "implemented rate limiting ddos protection and waf rules for production api endpoints",
    "built disaster recovery and backup strategy achieving recovery point objective of one hour",
    "designed chaos engineering experiments using gremlin to validate system resilience assumptions",
    "implemented secret management using hashicorp vault and aws secrets manager for credentials",
    "built compliance monitoring system tracking data residency and privacy requirements across services",
    "designed multi-region failover with rto of five minutes for mission-critical payment service",
    "implemented encryption at rest and in transit for all user data using aes-256 and tls",
    # ── data engineering & analytics ─────────────────────────────────────────
    "built real-time etl pipeline ingesting and transforming data from fifty different source systems",
    "designed star schema data warehouse with dimension and fact tables for business intelligence",
    "implemented stream processing using apache flink for real-time aggregation and alerting",
    "built automated data quality monitoring system detecting schema drift and null value anomalies",
    "developed feature store using feast for sharing machine learning features across teams",
    "designed dbt transformation layer on top of snowflake for analytics engineering workflows",
    "built data catalog and lineage tracking system to document and discover internal datasets",
    "implemented delta lake lakehouse architecture on databricks for unified batch and streaming",
    "designed partitioning and compaction strategy for parquet files on aws s3 data lake",
    "built reporting dashboard using apache superset with custom sql metrics for business stakeholders",
    "developed real-time feature engineering pipeline using kafka streams for online model serving",
    "implemented data masking and anonymization pipeline for gdpr compliance across data warehouse",
    "built automated pipeline to ingest public job posting data and extract structured skill information",
    "designed multi-hop knowledge graph pipeline linking candidates skills roles and organizations",
    "implemented incremental data processing using watermarks and late-arrival handling in flink",
    # ── nlp & search ─────────────────────────────────────────────────────────
    "developed custom named entity recognition model for extracting skills from unstructured resume text",
    "implemented query understanding and intent classification for enterprise search using bert",
    "built synonym expansion and spell correction layer for improving search recall at query time",
    "designed learning to rank pipeline using listwise ranking loss and offline click-through data",
    "implemented document clustering using k-means and topic modeling with latent dirichlet allocation",
    "developed sentiment analysis model for analyzing candidate reviews and employer feedback",
    "built multilingual text processing pipeline supporting english hindi and spanish documents",
    "implemented coreference resolution and entity linking for structured information extraction",
    "designed hybrid search ranking combining keyword matching and semantic similarity with learned weights",
    "developed question answering system using dense passage retrieval and generative reader models",
    "implemented real-time autocomplete using trie data structure and prefix search on elasticsearch",
    "built resume parser extracting structured work experience education and skills using spacy nlp",
    "designed entity normalization pipeline mapping skill aliases and abbreviations to canonical forms",
    "implemented keyphrase extraction using unsupervised graph-based ranking and supervised tagger",
    "developed text augmentation techniques using back-translation and synonym replacement for training",
    # ── systems & architecture ────────────────────────────────────────────────
    "designed event-driven architecture using pub-sub messaging to decouple services and enable scaling",
    "built api versioning strategy and backward-compatible schema evolution for public rest api",
    "implemented circuit breaker and retry logic using resilience4j to handle downstream service failures",
    "designed caching hierarchy using cdn edge cache redis and in-process cache for read-heavy workloads",
    "built internal service discovery and load balancing using consul and envoy proxy sidecar",
    "implemented async job processing using celery and rabbitmq for background task execution",
    "designed multi-tenant data isolation using row-level security in postgresql and kubernetes namespaces",
    "built streaming aggregation service computing rolling window metrics over high-velocity event stream",
    "designed modular plugin architecture allowing third-party integrations without core code changes",
    "implemented websocket server for real-time bidirectional communication supporting fifty thousand connections",
    "built content delivery network integration and cache invalidation strategy for global user base",
    "designed workflow orchestration system using airflow dagster for complex data pipeline scheduling",
    "implemented saga pattern for distributed transaction management across microservices boundaries",
    "built capacity planning and auto-scaling system based on predicted traffic load using time-series forecasting",
    "designed api gateway with request routing authentication caching and rate limiting for all services",
    # ── experience & skills descriptions ─────────────────────────────────────
    "five years of experience building and deploying machine learning systems in production environments",
    "strong expertise in python java and golang with deep knowledge of distributed systems design",
    "proficient in pytorch tensorflow and huggingface transformers for training and fine-tuning models",
    "extensive experience with aws gcp and azure cloud platforms and container orchestration using kubernetes",
    "hands-on experience designing high availability systems with strong understanding of cap theorem",
    "strong background in natural language processing computer vision and large language model research",
    "working knowledge of postgresql mongodb redis elasticsearch and snowflake data storage systems",
    "experience leading engineering teams of five to fifteen people across multiple time zones",
    "strong skills in data structures algorithms system design and software engineering best practices",
    "proficient in sql and experience with big data processing frameworks spark hadoop and flink",
    "deep expertise in search ranking and recommendation systems with hands-on production experience",
    "experience with agile development processes including sprint planning code review and retrospectives",
    "strong communication skills with experience presenting technical findings to non-technical stakeholders",
    "bachelor degree in computer science from iit with additional masters in machine learning from stanford",
    "open source contributor to pytorch huggingface and scikit-learn with multiple accepted pull requests",
    "experience building real-time systems with sub-millisecond latency requirements at high throughput",
    "strong understanding of llm alignment prompt engineering retrieval augmented generation and fine-tuning",
    "experience designing and operating data pipelines processing terabytes of data daily",
    "hands-on experience with mlflow weights and biases for experiment tracking and model registry",
    "proficient in react typescript and node for full-stack web application development and deployment",
]


def _try_lmplz(corpus_text: str, arpa_path: Path, order: int) -> bool:
    """Attempt to build the ARPA using the lmplz binary; return True on success."""
    # Search for lmplz in PATH and common kenlm package locations
    candidates = [shutil.which("lmplz")]
    try:
        import kenlm
        for sub in ("bin/lmplz", "build/bin/lmplz"):
            candidates.append(str(Path(kenlm.__file__).parent / sub))
    except ImportError:
        pass
    import sys as _sys
    for prefix in (_sys.prefix, _sys.exec_prefix):
        candidates.append(str(Path(prefix) / "bin" / "lmplz"))

    lmplz_bin = next((c for c in candidates if c and Path(c).exists()), None)
    if not lmplz_bin:
        logger.info("  lmplz binary not found; using Python n-gram builder")
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8", newline="\n"
    ) as tf:
        tf.write(corpus_text)
        corpus_file = tf.name

    try:
        proc = subprocess.run(
            [lmplz_bin, "-o", str(order), "--text", corpus_file, "--arpa", str(arpa_path)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0 and arpa_path.exists():
            logger.info(
                f"  lmplz built {order}-gram ARPA "
                f"({arpa_path.stat().st_size // 1024} KB)"
            )
            return True
        logger.warning(f"  lmplz failed (rc={proc.returncode}): {proc.stderr[:300]}")
        return False
    except Exception as exc:
        logger.warning(f"  lmplz error: {exc}")
        return False
    finally:
        Path(corpus_file).unlink(missing_ok=True)


def _build_arpa_python(sentences: list, order: int, path: Path) -> None:
    """Pure-Python n-gram ARPA builder using interpolated absolute discounting.

    Builds a valid KenLM-loadable ARPA file at *order* (e.g. 5).  Correctness
    guarantee: for every history h, sum_w P(w|h) <= 1 and the ARPA backoff
    weights account for the remaining mass.
    """
    D = 0.75  # absolute discount (Kneser-Ney convention)

    # ── tokenise ─────────────────────────────────────────────────────────────
    tokenized = []
    for s in sentences:
        words = s.strip().lower().split()
        if words:
            # Pad with (order-1) BOS tokens so every word has a full history
            tokenized.append(["<s>"] * (order - 1) + words + ["</s>"])

    # ── raw n-gram counts ─────────────────────────────────────────────────────
    raw: dict[int, dict] = {}
    for n in range(1, order + 1):
        c: dict = defaultdict(int)
        for words in tokenized:
            for i in range(len(words) - n + 1):
                gram = tuple(words[i : i + n])
                if n > 1 and gram[0] == "</s>":
                    continue
                c[gram] += 1
        raw[n] = dict(c)

    # ── context totals (sum of counts over all extensions of a prefix) ────────
    ctx: dict = defaultdict(int)
    ctx[()] = sum(raw[1].values())
    for n in range(2, order + 1):
        for gram, count in raw[n].items():
            ctx[gram[:-1]] += count

    # ── followers: (order, prefix) -> {word -> count} ─────────────────────────
    foll: dict = {}
    for n in range(2, order + 1):
        d: dict = defaultdict(dict)
        for gram, count in raw[n].items():
            d[gram[:-1]][gram[-1]] = count
        for h, succ in d.items():
            foll[(n, h)] = succ

    # ── log probs ────────────────────────────────────────────────────────────
    lp: dict = {}           # gram_tuple -> log10(P(last_word | prefix))
    N = ctx[()]

    # Unigrams — ML estimate; special handling for <s> and <unk>
    for gram, count in raw[1].items():
        lp[gram] = math.log10(max(count / N, 1e-12))
    lp[("<s>",)] = -99.0    # sentence-start marker cannot be generated
    lp[("<unk>",)] = math.log10(max(1.0 / (N + len(raw[1]) + 1), 1e-12))

    # Bigrams through highest order — absolute discounting + interpolation
    for n in range(2, order + 1):
        for gram, count in raw[n].items():
            h = gram[:-1]
            c_h = ctx.get(h, 0)
            if c_h == 0:
                continue
            succ_h = foll.get((n, h), {})
            lambda_h = D * len(succ_h) / c_h
            p_disc = max(count - D, 0.0) / c_h
            # Lower-order probability: back off one position in the history
            backed = gram[1:]                        # (n-1)-gram
            p_lower = 10.0 ** lp.get(
                backed,
                lp.get(gram[-1:], lp.get(("<unk>",), -7.0)),
            )
            lp[gram] = math.log10(max(p_disc + lambda_h * p_lower, 1e-12))

    # ── backoff weights for orders 1 … order-1 ───────────────────────────────
    bow: dict = {}   # prefix_tuple -> log10(backoff_weight)
    for n in range(1, order):
        for key, succ_dict in foll.items():
            if key[0] != n + 1:
                continue
            h = key[1]                              # prefix of length n
            seen_w = set(succ_dict.keys())

            # A = prob mass already assigned to words seen after h
            A = sum(10.0 ** lp.get(h + (w,), -15.0) for w in seen_w)
            # B = lower-order prob mass for those same words
            h_back = h[1:]                          # one shorter history
            B = sum(
                10.0 ** lp.get(
                    h_back + (w,),
                    lp.get((w,), lp.get(("<unk>",), -7.0)),
                )
                for w in seen_w
            )
            num = max(1.0 - A, 0.0)
            den = max(1.0 - B, 1e-10)
            bow[h] = math.log10(max(num / den, 1e-10))

    # ── collect grams per order ───────────────────────────────────────────────
    grams_by_order: dict = {}
    for n in range(1, order + 1):
        g = list(raw[n].keys())
        if n == 1 and ("<unk>",) not in raw[1]:
            g.append(("<unk>",))
        grams_by_order[n] = sorted(g, key=lambda x: " ".join(x))

    # ── write ARPA ────────────────────────────────────────────────────────────
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(grams_by_order[n])}\n")
        f.write("\n")
        for n in range(1, order + 1):
            f.write(f"\\{n}-grams:\n")
            for gram in grams_by_order[n]:
                text = " ".join(gram)
                prob = lp.get(gram, -7.0)
                if n < order:
                    b = bow.get(gram, 0.0)
                    f.write(f"{prob:.4f}\t{text}\t{b:.4f}\n")
                else:
                    f.write(f"{prob:.4f}\t{text}\n")
            f.write("\n")
        f.write("\\end\\\n")


def build_kenlm(order: int = 5):
    """Build a {order}-gram KenLM ARPA model from the curated tech-resume corpus.

    Tries the ``lmplz`` binary first (faster, standard KN smoothing).  Falls
    back to the pure-Python absolute-discounting builder when lmplz is absent
    (typical on Windows pip installs).
    """
    logger.info(f"Building KenLM {order}-gram model from tech-resume corpus...")
    out_dir = DECOMP / "kenlm_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    arpa = out_dir / "model.arpa"

    try:
        corpus_text = "\n".join(_TECH_CORPUS)
        total_tokens = sum(len(s.split()) for s in _TECH_CORPUS)
        logger.info(
            f"  Corpus: {len(_TECH_CORPUS)} sentences, ~{total_tokens} tokens"
        )

        # ── attempt lmplz (preferred) ────────────────────────────────────────
        built = _try_lmplz(corpus_text, arpa, order)

        # ── fall back to Python builder ───────────────────────────────────────
        if not built:
            logger.info(f"  Building {order}-gram ARPA in Python...")
            _build_arpa_python(_TECH_CORPUS, order, arpa)

        # ── sanity-check the model loads and has sensible perplexity ─────────
        try:
            import kenlm
            model = kenlm.Model(str(arpa))
            probe = model.perplexity("developed machine learning system using python")
            logger.info(
                f"  {arpa.name}: {order}-gram, "
                f"{arpa.stat().st_size // 1024} KB, "
                f"probe perplexity={probe:.1f}"
            )
            if probe > 1e8:
                logger.warning(
                    "  Probe perplexity is very high — model vocabulary may be too small"
                )
        except Exception as exc:
            logger.warning(f"  KenLM load check failed: {exc}")

        _compress("kenlm_model")
        return True

    except Exception as exc:
        logger.error(f"KenLM build failed: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 5. FRAUD KB (L1) — SQLite from public sources / curated lists
# ──────────────────────────────────────────────────────────────────────────────
FICTIONAL = [
    ("dunder mifflin", "TV fiction (The Office)"),
    ("hooli", "TV fiction (Silicon Valley)"),
    ("pied piper", "TV fiction (Silicon Valley)"),
    ("acme corp", "cartoon fiction"),
    ("acme corporation", "cartoon fiction"),
    ("initech", "film fiction (Office Space)"),
    ("globex", "TV fiction (Simpsons)"),
    ("soylent corp", "film fiction"),
    ("umbrella corporation", "game fiction (Resident Evil)"),
    ("stark industries", "comic fiction (Marvel)"),
    ("wayne enterprises", "comic fiction (DC)"),
    ("cyberdyne systems", "film fiction (Terminator)"),
    ("weyland-yutani", "film fiction (Alien)"),
    ("tyrell corporation", "film fiction (Blade Runner)"),
    ("oscorp", "comic fiction (Marvel)"),
    ("aperture science", "game fiction (Portal)"),
    ("black mesa", "game fiction (Half-Life)"),
    ("vault-tec", "game fiction (Fallout)"),
    ("wonka industries", "film fiction"),
    ("massive dynamic", "TV fiction (Fringe)"),
]

# A tiny seed of real company founding years. In production, populate this from
# MCA data.gov.in / DPIIT / public registries via build scripts.
COMPANY_FOUNDING_SEED = [
    ("google", 1998), ("microsoft", 1975), ("amazon", 1994), ("apple", 1976),
    ("meta", 2004), ("facebook", 2004), ("netflix", 1997), ("tesla", 2003),
    ("openai", 2015), ("anthropic", 2021), ("nvidia", 1993), ("infosys", 1981),
    ("tcs", 1968), ("wipro", 1945), ("flipkart", 2007), ("zomato", 2008),
    ("paytm", 2010), ("ola", 2010), ("swiggy", 2014), ("razorpay", 2014),
]

SKILL_ALIASES = [
    ("large language models", "llm", "ml"),
    ("machine learning", "ml", "ml"),
    ("natural language processing", "nlp", "nlp"),
    ("deep learning", "dl", "ml"),
    ("retrieval augmented generation", "rag", "nlp"),
    ("python", "python3", "lang"),
    ("kubernetes", "k8s", "infra"),
]


def build_fraud_kb():
    """Build fraud KB using the enhanced multi-source builder (rebuild_fraud_kb.py).

    Sources included:
      - Indian Companies   : MCA data.gov.in
      - Indian Startups    : DPIIT Startup India
      - Indian Universities: AICTE + UGC
      - Global Companies   : Kaggle PDL 7M
      - Global Universities: WHED UNESCO
      - Research Venues    : ArXiv bulk metadata
    """
    logger.info("Building fraud KB (SQLite) — enhanced multi-source build...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rebuild_fraud_kb", ROOT / "rebuild_fraud_kb.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok = mod.build()
        if ok:
            logger.info("  Fraud KB (enhanced) built successfully")
        return ok
    except Exception as e:
        logger.error(f"Enhanced fraud KB build failed: {e}; falling back to seed-only build")
        # Minimal fallback so the pipeline isn't broken
        out_dir = DECOMP / "fraud_kb"
        out_dir.mkdir(parents=True, exist_ok=True)
        db_path = out_dir / "fraud_kb.db"
        if db_path.exists():
            db_path.unlink()
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE fictional_companies (company_name TEXT PRIMARY KEY, reason TEXT)")
        conn.execute("CREATE TABLE company_founding_dates (company_name TEXT PRIMARY KEY, founding_year INT)")
        conn.execute("CREATE TABLE skill_aliases (skill_canonical TEXT, alias TEXT, category TEXT)")
        conn.executemany("INSERT OR IGNORE INTO fictional_companies VALUES (?,?)", FICTIONAL)
        conn.executemany("INSERT OR IGNORE INTO company_founding_dates VALUES (?,?)", COMPANY_FOUNDING_SEED)
        conn.executemany("INSERT INTO skill_aliases VALUES (?,?,?)", SKILL_ALIASES)
        conn.execute("CREATE INDEX idx_fic ON fictional_companies(company_name)")
        conn.execute("CREATE INDEX idx_found ON company_founding_dates(company_name)")
        conn.execute("CREATE INDEX idx_alias ON skill_aliases(skill_canonical)")
        conn.commit()
        conn.close()
        logger.info(f"  Fraud KB (fallback): {len(FICTIONAL)} fictional, "
                    f"{len(COMPANY_FOUNDING_SEED)} real companies")
        _compress("fraud_kb")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# COMPRESSION + CHECKSUMS
# ──────────────────────────────────────────────────────────────────────────────
def _compress(name: str):
    src = DECOMP / name
    if not src.exists():
        logger.warning(f"  nothing to compress for {name}")
        return
    tar_path = COMP / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src, arcname=name)
    size_mb = tar_path.stat().st_size / 1e6
    logger.info(f"  compressed → {tar_path.name} ({size_mb:.1f} MB)")
    _update_checksum(tar_path)


def _update_checksum(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    checksum_file = COMP / "checksums.sha256"
    lines = {}
    if checksum_file.exists():
        for line in checksum_file.read_text().splitlines():
            if "  " in line:
                cs, fn = line.split("  ", 1)
                lines[fn] = cs
    lines[path.name] = h.hexdigest()
    checksum_file.write_text("".join(f"{cs}  {fn}\n" for fn, cs in sorted(lines.items())))


def main():
    ap = argparse.ArgumentParser(description="Build all RedRob model artifacts")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sentence-transformer", action="store_true")
    ap.add_argument("--flashrank", action="store_true")
    ap.add_argument("--spacy", action="store_true")
    ap.add_argument("--kenlm", action="store_true")
    ap.add_argument("--fraud-kb", action="store_true")
    args = ap.parse_args()

    if not any(vars(args).values()):
        ap.error("Pass --all or specific flags (e.g. --fraud-kb)")

    t0 = time.time()
    results = {}
    if args.all or args.fraud_kb:            results["fraud_kb"] = build_fraud_kb()
    if args.all or args.sentence_transformer: results["sentence_transformer"] = build_sentence_transformer()
    if args.all or args.flashrank:           results["flashrank"] = build_flashrank()
    if args.all or args.spacy:               results["spacy"] = build_spacy()
    if args.all or args.kenlm:               results["kenlm"] = build_kenlm()

    print("\n" + "=" * 50)
    print("BUILD SUMMARY")
    print("=" * 50)
    for name, ok_ in results.items():
        print(f"  {'OK' if ok_ else 'FAIL'} {name}")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Compressed artifacts in: {COMP}")
    print("Next: commit models/compressed/*.tar.gz (Git LFS), then run setup_check.py")


if __name__ == "__main__":
    main()
