"""
dictionary.py — Skill synonym/alias dictionary for RedRob L1c NLP skill matching.

Structure:
    SKILL_SYNONYMS[canonical_name] = [list_of_equivalent_terms]

All canonical keys and synonyms are lowercase. Matching is case-insensitive.
Use expand_skill(term) to get all recognized forms for a given input term.

v2 Changes:
- Added JD-critical canonicals: embeddings-based retrieval systems, vector databases,
  hybrid search infrastructure, evaluation frameworks for ranking systems
- Expanded: information retrieval, mlops, semantic search, embeddings
- Added: learning to rank, hybrid search, retrieval systems, reranking
- Added: vector search tools (Qdrant, Milvus, Weaviate, pgvector, LanceDB, Vespa)
- Added: search infrastructure (BM25, Haystack, Solr, OpenSearch, Elasticsearch)
- Added: IR evaluation metrics (NDCG, MAP, MRR, BEIR, TREC, nDCG@k)
- Added: embedding models (sentence-transformers, bi-encoder, cross-encoder, HNSW, ScaNN)
- Added: MLOps tools (MLflow, Kubeflow, W&B, DVC, feature store, model registry)
- Added: AI/LLM ecosystem (LlamaIndex, LangChain, OpenAI, Anthropic, Cohere)
- Added: modern ML (diffusion models, multimodal, vision-language, RLHF, DPO)
- Added: data science tools (Polars, DuckDB, Arrow, Dask)
- Added: SRE/Platform engineering, observability tools
- Expanded: cloud services (AWS Bedrock, Azure OpenAI, Vertex AI)
- Added: domain-specific (RecSys, search relevance, candidate ranking, ATS)
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# MASTER DICTIONARY
# canonical → list of equivalent terms (do NOT repeat the canonical in the list)
# ──────────────────────────────────────────────────────────────────────────────
SKILL_SYNONYMS: dict[str, list[str]] = {

    # ══════════════════════════════════════════════════════════════════════════
    # JD-CRITICAL CANONICALS (Senior AI Engineer — Retrieval/Ranking/Search)
    # These map the exact JD skill phrases to all real-world tool/term variants
    # ══════════════════════════════════════════════════════════════════════════

    "embeddings-based retrieval systems": [
        # Tools
        "faiss", "facebook ai similarity search", "faiss index", "faiss library",
        "sentence transformers", "sentence-transformers", "sentence transformer",
        "pgvector", "pg vector", "postgres vector",
        "annoy", "hnswlib", "hnsw", "scann", "nmslib",
        # Concepts
        "dense retrieval", "dense vector retrieval", "neural retrieval",
        "bi-encoder", "bi encoder", "biencoder", "dual encoder", "dual-encoder",
        "cross-encoder", "cross encoder", "crossencoder",
        "embedding retrieval", "vector retrieval", "ann search",
        "approximate nearest neighbor", "approximate nearest neighbour",
        "semantic retrieval", "embedding search", "vector similarity search",
        "knn search", "k-nearest neighbor search", "dense passage retrieval",
        "dpr", "retrieval system", "embedding based search",
        "neural search", "passage retrieval", "document retrieval",
        "text retrieval", "nearest neighbor search",
        # Production phrasing variants (LLM often outputs these)
        "production embeddings", "embeddings in production",
        "production with embedding", "embedding system",
        "retrieval with embeddings", "embedding-based search",
        "vector-based retrieval", "production retrieval system",
        "production vector search",
    ],

    "vector databases": [
        # Dedicated vector DBs
        "qdrant", "qdrant vector",
        "milvus", "milvus vector",
        "weaviate", "weaviate vector",
        "pinecone", "pinecone vector",
        "chroma", "chromadb", "chroma vector",
        "lancedb", "lance db",
        "marqo",
        "vespa", "vespa.ai",
        "zilliz",
        "vald",
        # Vector extensions
        "pgvector", "pg vector", "postgres vector",
        "redis vector", "redis vss", "redis search",
        "opensearch knn", "opensearch vector",
        "elasticsearch vector", "elastic vector search",
        # Concepts
        "vector store", "vector db", "vectordb",
        "vector search engine", "vector index",
        "similarity search database", "ann database",
        "embedding database", "embedding store",
    ],

    "hybrid search infrastructure": [
        # Sparse retrieval tools
        "bm25", "bm 25", "okapi bm25", "bm25 ranking",
        "tf-idf", "tfidf", "tf idf", "term frequency",
        "sparse retrieval", "keyword search", "inverted index",
        # Search engines
        "elasticsearch", "elastic search", "elastic",
        "opensearch", "open search",
        "solr", "apache solr",
        "haystack", "deepset haystack",
        "typesense",
        "meilisearch", "meili search",
        "algolia",
        # Hybrid concepts
        "hybrid search", "hybrid ranking", "hybrid retrieval",
        "dense sparse fusion", "dense+sparse",
        "reciprocal rank fusion", "rrf",
        "colbert", "colbertv2", "late interaction",
        "splade",
        "reranking", "re-ranking", "reranker",
        "cross-encoder reranking", "two-stage retrieval",
        "retrieval pipeline", "search pipeline",
        "query expansion", "query rewriting",
        "lexical search", "full text search", "fts",
    ],

    "evaluation frameworks for ranking systems": [
        # IR Metrics
        "ndcg", "ndcg@10", "ndcg@k", "normalized discounted cumulative gain",
        "map", "mean average precision",
        "mrr", "mean reciprocal rank",
        "precision@k", "recall@k", "p@10", "p@5",
        "hit rate", "hit@k",
        "f1 score retrieval", "average precision",
        # Benchmarks
        "beir", "beir benchmark", "beir evaluation",
        "trec", "trec eval", "trec benchmark",
        "msmarco", "ms marco", "squad retrieval",
        "hotpotqa", "natural questions nq",
        # Evaluation concepts
        "ranking evaluation", "retrieval evaluation",
        "ir metrics", "ir evaluation", "search quality",
        "relevance evaluation", "relevance judgment",
        "offline evaluation", "online evaluation",
        "a/b testing search", "interleaving", "counterfactual evaluation",
        "ranking quality", "search relevance evaluation",
        "recall evaluation", "precision evaluation",
        "evaluation harness", "lm eval", "llm evaluation",
    ],

    "information retrieval": [
        "ir", "information retrieval system",
        "search systems", "search engine", "search engines",
        "ranking systems", "search ranking", "result ranking",
        "learning to rank", "ltr", "learntorank", "learn to rank",
        "search relevance", "relevance ranking",
        "query understanding", "query processing", "query analysis",
        "document ranking", "passage ranking",
        "search infrastructure", "retrieval infrastructure",
        "question answering", "qa system", "open domain qa",
        "knowledge retrieval", "fact retrieval",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # PROGRAMMING LANGUAGES
    # ══════════════════════════════════════════════════════════════════════════

    "python": [
        "python3", "python 3", "py", "python scripting", "python programming",
        "python development", "python backend", "python engineer",
        "cpython", "ipython",
    ],
    "java": [
        "java 8", "java 11", "java 17", "core java", "java ee", "j2ee",
        "java programming", "java development", "java backend",
    ],
    "javascript": [
        "js", "java script", "ecmascript", "es6", "es2015",
        "es2020", "es2022", "vanilla js", "node.js", "nodejs",
    ],
    "typescript": ["ts", "type script", "typescript development"],
    "golang": ["go", "go lang", "go programming", "google go", "go development"],
    "rust": ["rust lang", "rust programming", "rust systems"],
    "c++": ["cpp", "c plus plus", "cplusplus", "c/c++", "c++ programming"],
    "c#": ["csharp", "c sharp", ".net c#", "c sharp dotnet"],
    "c": ["c language", "ansi c", "c programming"],
    "ruby": ["ruby programming", "ruby development"],
    "php": ["php7", "php8", "php programming"],
    "kotlin": ["kotlin android", "kotlin jvm", "kotlin backend"],
    "swift": ["swift ios", "swift programming"],
    "scala": ["scala programming", "akka scala", "scala spark"],
    "r": ["r programming", "r language", "rstats", "r statistical", "r data analysis"],
    "matlab": ["mat lab", "matlab programming", "matlab modeling"],
    "julia": ["julia language", "julia programming", "julia ml"],
    "bash": [
        "shell script", "bash scripting", "shell scripting",
        "sh", "zsh", "bash script", "linux scripting",
    ],
    "perl": ["perl scripting"],
    "groovy": ["apache groovy"],
    "dart": ["dart language", "flutter dart"],
    "assembly": ["asm", "assembly language"],
    "cobol": [],
    "fortran": [],
    "haskell": ["functional programming haskell"],
    "elixir": ["elixir phoenix"],
    "clojure": [],
    "f#": ["fsharp", "f sharp"],
    "objective-c": ["objc", "objective c"],
    "vba": ["visual basic for applications", "excel vba"],
    "powershell": ["power shell", "ps scripting"],
    "lua": [],
    "solidity": ["solidity smart contracts", "ethereum solidity"],

    # ══════════════════════════════════════════════════════════════════════════
    # ML / AI CONCEPTS
    # ══════════════════════════════════════════════════════════════════════════

    "machine learning": [
        "ml", "machine-learning", "statistical learning",
        "predictive modeling", "statistical modeling",
        "ml engineering", "applied ml", "applied machine learning",
        "ml research", "machine learning engineer",
    ],
    "deep learning": [
        "dl", "deep-learning", "neural networks", "neural network",
        "ann", "dnn", "deep neural network", "deep neural networks",
        "deep learning engineering",
    ],
    "natural language processing": [
        "nlp", "natural language understanding", "nlu", "text mining",
        "computational linguistics", "text analytics", "language ai",
        "text processing", "language processing",
    ],
    "computer vision": [
        "cv", "image recognition", "image classification", "object detection",
        "visual ai", "image processing", "video analytics",
        "vision ai", "vision model",
    ],
    "reinforcement learning": [
        "rl", "rl agent", "reward learning", "policy gradient",
        "q-learning", "ppo", "dqn", "actor critic",
    ],
    "generative ai": [
        "gen ai", "genai", "generative artificial intelligence",
        "ai generation", "generative model",
    ],
    "large language model": [
        "llm", "llms", "large language models", "language model",
        "language models", "foundation model", "foundation models",
        "llm engineering", "llm development", "llm fine-tuning",
        # Production / application variants
        "llm application", "llm deployment", "llm inference",
        "llm api", "llm orchestration", "llm pipeline",
        "llm integration", "llm evaluation", "llm ops",
        # Open-source model families (commonly listed on resumes)
        "llama", "llama2", "llama3", "llama 2", "llama 3",
        "mistral", "mixtral", "falcon", "qwen", "phi", "phi-3",
        "gemma", "gemma2", "vicuna", "alpaca", "wizard",
        # Commercial model families
        "gpt-4", "gpt4", "gpt-3.5", "claude model", "gemini",
        "palm", "palm2",
    ],
    "retrieval augmented generation": [
        "rag", "retrieval-augmented generation", "retrieval augmented",
        "rag pipeline", "rag system", "rag architecture",
        "rag application", "rag chatbot", "knowledge-augmented generation",
        "grounded generation",
        # Cross-encoder / reranking overlap (RAG pipelines commonly include reranking)
        "rag with reranking", "rag reranking", "retrieval and reranking",
        "retrieval reranking pipeline", "augmented generation",
        "context retrieval", "knowledge retrieval generation",
    ],
    "transformer": [
        "transformers", "transformer architecture", "attention mechanism",
        "self-attention", "multi-head attention", "encoder decoder",
        "transformer model", "attention model",
    ],
    "bert": [
        "bidirectional encoder representations", "bert model",
        "bert embeddings", "roberta", "distilbert", "albert",
        "deberta", "electra", "xlm-roberta",
    ],
    "gpt": [
        "gpt-2", "gpt-3", "gpt-4", "gpt4", "gpt3", "chatgpt",
        "openai gpt", "gpt model", "gpt-3.5",
    ],
    "embeddings": [
        "vector embeddings", "word embeddings", "text embeddings",
        "sentence embeddings", "word vectors", "dense vectors",
        "semantic vectors", "embedding model", "embedding layer",
        "contextual embeddings", "static embeddings",
        # Similarity / matching variants
        "similarity matching", "vector similarity matching", "embedding similarity",
        "semantic similarity matching", "vector similarity",
        "similarity search", "embedding based", "vector based",
        "embedding generation", "text embedding", "vector representation",
    ],
    "semantic search": [
        "dense retrieval", "neural search", "vector search",
        "semantic retrieval", "meaning-based search",
        "semantic similarity", "semantic matching",
        # Overlap with embeddings and retrieval
        "embedding search", "vector similarity search",
        "similarity-based search", "similarity based search",
        "semantic text matching", "dense search",
        "neural semantic search", "semantic retrieval system",
    ],
    "learning to rank": [
        "ltr", "learntorank", "learn to rank",
        "ranking model", "ranknet", "lambdamart", "lambdarank",
        "pointwise ranking", "pairwise ranking", "listwise ranking",
        "xgboost ranking", "lightgbm ranking",
    ],
    "hybrid search": [
        "sparse dense fusion", "dense sparse retrieval",
        "hybrid retrieval", "bm25 + vector", "keyword + semantic",
        "multi-stage retrieval", "two-stage retrieval",
        "first stage retrieval", "second stage ranking",
    ],
    "prompt engineering": [
        "prompt design", "chain-of-thought", "few-shot prompting",
        "zero-shot", "cot prompting", "prompt optimization",
        "system prompt", "prompt template",
    ],
    "fine-tuning": [
        "finetuning", "fine tuning", "model fine-tuning",
        "lora", "qlora", "peft", "instruction tuning",
        "rlhf", "supervised fine-tuning", "sft",
        "dpo", "direct preference optimization",
        "rpo", "continued pretraining",
        # Full names of the abbreviations
        "low rank adaptation", "low-rank adaptation",
        "parameter efficient fine tuning", "parameter efficient finetuning",
        "quantized lora", "quantized low rank adaptation",
        "reinforcement learning from human feedback",
        # Broader fine-tuning techniques
        "model adaptation", "domain adaptation", "task-specific fine-tuning",
        "adapter tuning", "prefix tuning", "prompt tuning",
        "llm fine-tuning", "llm finetuning", "language model fine-tuning",
        "custom model training", "transfer learning fine-tuning",
        # Common related terms
        "model alignment", "instruction following", "chat fine-tuning",
        "axolotl", "unsloth", "llama factory", "trl", "transformers trainer",
    ],
    "multimodal": [
        "multimodal model", "vision language model", "vlm",
        "image text model", "clip", "dall-e", "stable diffusion",
        "llava", "flamingo", "blip",
    ],
    "diffusion models": [
        "diffusion model", "stable diffusion", "ddpm",
        "denoising diffusion", "latent diffusion",
    ],
    "knowledge graph": [
        "kg", "knowledge base", "ontology",
        "knowledge representation", "graph rag",
    ],
    "recommendation system": [
        "recommender system", "rec sys", "recsys",
        "collaborative filtering", "content-based filtering",
        "hybrid recommender", "matrix factorization",
        "two-tower model", "candidate retrieval",
    ],
    "time series": [
        "time-series", "time series analysis", "forecasting",
        "sequence modeling", "temporal modeling",
        "time series forecasting", "anomaly detection time series",
    ],
    "anomaly detection": [
        "outlier detection", "fraud detection ml",
        "novelty detection", "out of distribution detection",
    ],
    "classification": [
        "binary classification", "multi-class classification",
        "text classification", "image classification",
        "multilabel classification",
    ],
    "regression": [
        "linear regression", "logistic regression",
        "regression analysis", "regression model",
    ],
    "clustering": [
        "k-means", "dbscan", "hdbscan", "unsupervised clustering",
        "cluster analysis", "topic modeling",
    ],
    "feature engineering": [
        "feature extraction", "feature selection",
        "dimensionality reduction", "pca", "feature store",
    ],
    "model deployment": [
        "model serving", "ml deployment", "production ml",
        "mlops deployment", "model inference", "inference serving",
        "model api", "inference endpoint",
    ],
    "data augmentation": [
        "augmentation", "synthetic data", "oversampling",
        "undersampling", "smote",
    ],
    "object detection": [
        "yolo", "yolov8", "faster rcnn", "ssd object detection",
        "detection model", "detr",
    ],
    "speech recognition": [
        "asr", "automatic speech recognition", "speech to text",
        "stt", "whisper", "wav2vec",
    ],
    "text to speech": ["tts", "speech synthesis", "voice synthesis"],
    "a/b testing": [
        "ab testing", "a/b test", "ab test",
        "a/b test interpretation", "ab test interpretation",
        "split testing", "experimentation",
        "hypothesis testing", "online experimentation",
        "interleaving test", "bandit algorithms",
    ],
    "causal inference": [
        "causality", "causal analysis", "treatment effect",
        "propensity score", "instrumental variables",
    ],
    "knowledge distillation": [
        "model distillation", "teacher student", "model compression",
        "model pruning", "quantization",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # ML FRAMEWORKS & LIBRARIES
    # ══════════════════════════════════════════════════════════════════════════

    "tensorflow": [
        "tf", "tensor flow", "tensorflow 2", "tf2",
        "keras tensorflow", "tensorflow keras",
    ],
    "pytorch": [
        "torch", "py torch", "pytorch lightning",
        "torch nn", "pytorch training",
    ],
    "keras": ["keras api", "keras deep learning"],
    "scikit-learn": ["sklearn", "scikit learn", "scikitlearn"],
    "xgboost": ["xgb", "extreme gradient boosting"],
    "lightgbm": ["lgbm", "light gbm", "light gradient boosting"],
    "catboost": ["cat boost"],
    "hugging face": [
        "huggingface", "hf transformers", "hugging face transformers",
        "transformers library", "hf hub", "huggingface hub",
        "hugging face hub", "hf pipeline",
    ],
    "langchain": [
        "lang chain", "langchain framework", "langchain ai",
        "langchain python",
    ],
    "llamaindex": [
        "llama index", "llama-index", "gpt-index",
        "llama index framework",
    ],
    "openai api": [
        "openai", "chatgpt api", "gpt api", "openai sdk",
        "openai client", "openai embeddings",
    ],
    "anthropic": [
        "claude api", "claude ai", "anthropic api",
        "anthropic sdk",
    ],
    "cohere": ["cohere api", "cohere embeddings", "cohere rerank"],
    "spacy": ["spacy nlp", "spacy library", "spacy en"],
    "nltk": ["natural language toolkit", "nltk library"],
    "gensim": ["gensim nlp", "gensim word2vec"],
    "faiss": [
        "facebook ai similarity search", "faiss index",
        "faiss library", "faiss gpu",
    ],
    "annoy": ["approximate nearest neighbor oh yeah", "annoy index"],
    "hnswlib": ["hnsw", "hierarchical nsw", "hnsw index"],
    "scann": ["google scann", "scann index"],
    "weaviate": ["weaviate vector", "weaviate client"],
    "pinecone": ["pinecone vector", "pinecone client", "pinecone index"],
    "chroma": ["chromadb", "chroma vector", "chroma db"],
    "milvus": ["milvus vector", "milvus db", "pymilvus"],
    "qdrant": ["qdrant vector", "qdrant client", "qdrant cloud", "quadrant vector db"],
    "pgvector": ["pg vector", "postgres vector", "pgvector extension"],
    "lancedb": ["lance db", "lance vector"],
    "vespa": ["vespa.ai", "vespa search"],
    "opensearch": [
        "open search", "opensearch knn", "opensearch vector",
        "amazon opensearch",
    ],
    "elasticsearch": [
        "elastic search", "elastic", "es",
        "elastic stack", "elk", "elasticsearch client",
    ],
    "haystack": [
        "deepset haystack", "haystack framework",
        "haystack pipeline", "haystack ai",
    ],
    "pandas": [
        "pd", "pandas dataframe", "pandas library",
        "pandas data analysis",
    ],
    "numpy": ["np", "numpy arrays", "numpy library"],
    "scipy": ["scipy library", "scipy optimization"],
    "polars": ["polars dataframe", "polars library"],
    "dask": ["dask dataframe", "dask parallel"],
    "arrow": ["apache arrow", "pyarrow"],
    "duckdb": ["duck db", "duckdb analytics"],
    "matplotlib": ["matplotlib pyplot", "plt", "matplotlib plotting"],
    "seaborn": ["seaborn visualization", "seaborn plotting"],
    "plotly": ["plotly dash", "plotly visualization"],
    "opencv": ["cv2", "open cv", "open-cv", "opencv python"],
    "pillow": ["pil", "python imaging library"],
    "streamlit": ["streamlit app", "streamlit ui"],
    "gradio": ["gradio ui", "gradio demo"],
    "sentence-bert": [
        "sbert", "sentence bert", "all-minilm",
        "all-mpnet", "e5 model", "bge model",
        "instructor embeddings", "gte embeddings",
    ],
    "flashrank": ["flash rank", "flashrank reranker", "cross encoder reranker"],
    "reranking": [
        "re-ranking", "reranker", "cross-encoder reranking",
        "passage reranking", "bge reranker", "cohere rerank",
        "monot5", "monobert",
        # Cross-encoder is the most common reranking implementation
        "cross encoders", "cross-encoders", "cross encoder model",
        "neural reranking", "two-stage reranking", "semantic reranking",
        "rankgpt", "rank gpt", "llm reranking", "rag reranking",
        # Flashrank is the reranker used in this pipeline
        "flash rank", "flashrank reranker",
        # RAG adjacent — candidates with RAG experience almost always know reranking
        "rag pipeline reranking", "retrieval reranking",
    ],
    "colbert": ["colbertv2", "late interaction retrieval", "col bert"],
    "bm25": [
        "bm 25", "okapi bm25", "bm25 scoring",
        "bm25 retrieval", "bm25s", "rank-bm25",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # MLOPS / MODEL LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════════

    "mlops": [
        "ml ops", "machine learning operations",
        "ml pipeline", "ml infrastructure", "ml platform",
        "model lifecycle", "ml engineering",
    ],
    "mlflow": [
        "ml flow", "mlflow tracking", "mlflow registry",
        "mlflow experiments",
    ],
    "kubeflow": [
        "kube flow", "kubeflow pipelines",
        "kubeflow training", "kubeflow serving",
    ],
    "wandb": [
        "weights and biases", "weights & biases",
        "wandb logging", "wandb tracking",
    ],
    "dvc": [
        "data version control", "dvc pipeline",
        "dvc tracking",
    ],
    "prefect": ["prefect workflow", "prefect orchestration"],
    "bentoml": ["bento ml", "bento serving", "bento server"],
    "ray": [
        "ray serve", "ray tune", "ray train",
        "ray cluster", "ray distributed",
    ],
    "triton": ["triton inference server", "nvidia triton", "triton serving"],
    "onnx": [
        "onnx runtime", "open neural network exchange",
        "onnx model", "onnxruntime",
    ],
    "tensorrt": ["tensor rt", "nvidia tensorrt", "trt"],
    "feature store": [
        "feast", "tecton", "hopsworks",
        "feature platform", "ml feature store",
    ],
    "model registry": [
        "model store", "model versioning", "artifact registry",
        "mlflow registry",
    ],
    "model monitoring": [
        "drift detection", "data drift", "model drift",
        "concept drift", "evidently", "whylogs",
    ],
    "experiment tracking": [
        "experiment management", "run tracking",
        "hyperparameter tracking", "trial tracking",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # CLOUD PLATFORMS
    # ══════════════════════════════════════════════════════════════════════════

    "aws": [
        "amazon web services", "amazon cloud", "aws cloud",
        "amazon aws", "aws services",
    ],
    "azure": [
        "microsoft azure", "azure cloud", "azure ml",
        "azure services", "ms azure",
    ],
    "gcp": [
        "google cloud", "google cloud platform",
        "google cloud services", "google cloud compute",
    ],

    # AWS Services
    "sagemaker": ["aws sagemaker", "amazon sagemaker", "sagemaker studio"],
    "aws bedrock": ["amazon bedrock", "bedrock llm", "bedrock api"],
    "aws lambda": ["lambda function", "serverless lambda"],
    "dynamodb": ["dynamo db", "amazon dynamodb"],
    "amazon rds": ["rds", "relational database service"],
    "redshift": ["amazon redshift", "aws redshift"],
    "amazon kinesis": ["kinesis streaming", "kinesis data streams"],
    "aws glue": ["glue etl", "aws glue catalog"],
    "amazon s3": ["s3", "s3 bucket", "simple storage service"],
    "amazon ec2": ["ec2", "ec2 instance"],
    "aws ecs": ["ecs", "elastic container service"],
    "aws eks": ["eks", "elastic kubernetes service"],

    # Azure Services
    "azure ml": ["azure machine learning", "azure ml studio"],
    "azure openai": ["azure openai service", "azure gpt", "aoai"],
    "azure devops": ["ado", "azure pipelines"],
    "azure functions": ["azure serverless"],
    "azure blob storage": ["blob storage", "azure storage"],
    "azure databricks": ["azure spark", "databricks azure"],

    # GCP Services
    "bigquery": ["big query", "google bigquery", "bq"],
    "vertex ai": [
        "google vertex", "gcp vertex", "vertex ai platform",
        "vertex ml",
    ],
    "google cloud run": ["cloud run"],
    "dataflow": ["google dataflow", "apache beam gcp"],

    # ══════════════════════════════════════════════════════════════════════════
    # DEVOPS / INFRASTRUCTURE
    # ══════════════════════════════════════════════════════════════════════════

    "docker": [
        "containers", "containerization", "docker container",
        "docker image", "dockerfile", "docker compose",
    ],
    "kubernetes": [
        "k8s", "kube", "container orchestration",
        "k8s cluster", "kubernetes cluster",
    ],
    "terraform": [
        "infrastructure as code", "iac", "terraform iac",
        "terraform hcl",
    ],
    "ansible": ["ansible playbook", "ansible automation"],
    "jenkins": ["ci/cd jenkins", "jenkins pipeline"],
    "github actions": [
        "gh actions", "github ci", "github workflows",
        "github actions workflow",
    ],
    "gitlab ci": ["gitlab pipelines", "gitlab cd", "gitlab ci/cd"],
    "ci/cd": [
        "continuous integration", "continuous deployment",
        "continuous delivery", "cicd", "ci cd",
        "build pipeline", "deployment pipeline",
    ],
    "linux": [
        "unix", "ubuntu", "centos", "debian", "rhel",
        "linux administration", "linux server", "linux shell",
    ],
    "nginx": ["nginx web server", "nginx proxy", "nginx load balancer"],
    "prometheus": ["prometheus monitoring", "prometheus metrics"],
    "grafana": ["grafana dashboard", "grafana monitoring"],
    "datadog": ["datadog monitoring", "datadog apm"],
    "opentelemetry": ["otel", "open telemetry", "distributed tracing"],
    "elk stack": [
        "elasticsearch logstash kibana", "elk",
        "elastic stack", "kibana",
    ],
    "helm": ["helm charts", "helm kubernetes"],
    "argocd": ["argo cd", "gitops argocd"],
    "serverless": ["serverless framework", "faas", "function as a service"],
    "vault": ["hashicorp vault", "secrets management"],

    # ══════════════════════════════════════════════════════════════════════════
    # DATABASES
    # ══════════════════════════════════════════════════════════════════════════

    "postgresql": [
        "postgres", "pg", "psql", "postgresql database",
        "pgsql",
    ],
    "mysql": ["my sql", "mysql database"],
    "sqlite": ["sqlite3", "sqlite database"],
    "mongodb": ["mongo", "mongo db", "mongodb atlas"],
    "redis": ["redis cache", "redis db", "redis cluster"],
    "cassandra": ["apache cassandra", "datastax cassandra"],
    "neo4j": ["graph database", "neo4j graph", "cypher"],
    "snowflake": [
        "snowflake data warehouse", "snowflake db",
        "snowflake cloud",
    ],
    "databricks": [
        "data bricks", "databricks platform",
        "databricks lakehouse", "databricks spark",
    ],
    "clickhouse": ["click house", "clickhouse analytics"],
    "influxdb": ["time series db influx", "influx database"],

    # ══════════════════════════════════════════════════════════════════════════
    # SQL / DATA QUERYING
    # ══════════════════════════════════════════════════════════════════════════

    "sql": [
        "structured query language", "database querying",
        "query optimization", "sql queries", "sql scripting",
    ],
    "nosql": ["no sql", "non-relational database", "document database"],
    "graphql": [
        "graph ql", "graph query language", "apollo graphql",
        "graphql api", "graphql schema",
    ],
    "dbt": ["data build tool", "dbt core", "dbt cloud"],

    # ══════════════════════════════════════════════════════════════════════════
    # DATA ENGINEERING / BIG DATA
    # ══════════════════════════════════════════════════════════════════════════

    "apache spark": [
        "spark", "pyspark", "spark sql",
        "spark streaming", "apache spark ml",
    ],
    "apache kafka": [
        "kafka", "kafka streaming", "event streaming",
        "kafka cluster", "kafka producer", "kafka consumer",
    ],
    "apache hadoop": ["hadoop", "hdfs", "mapreduce", "hadoop ecosystem"],
    "apache flink": ["flink", "stream processing flink"],
    "apache hive": ["hive", "hiveql", "hive metastore"],
    "apache beam": ["beam pipeline", "beam dataflow"],
    "apache airflow": [
        "airflow", "airflow dag", "workflow orchestration",
        "airflow pipeline",
    ],
    "etl": [
        "extract transform load", "data pipeline", "data pipelines",
        "elt", "data ingestion",
    ],
    "data warehouse": [
        "data warehousing", "dwh", "edw",
        "enterprise data warehouse",
    ],
    "data lake": [
        "data lakehouse", "lakehouse", "delta lake",
        "apache iceberg", "delta table",
    ],
    "data modeling": [
        "dimensional modeling", "data schema design",
        "entity relationship",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # WEB FRAMEWORKS / BACKEND
    # ══════════════════════════════════════════════════════════════════════════

    "django": [
        "django rest framework", "drf", "django framework",
        "django python",
    ],
    "flask": ["flask python", "flask api", "flask framework"],
    "fastapi": ["fast api", "fastapi python", "fastapi framework"],
    "spring boot": [
        "spring", "spring framework", "spring mvc",
        "spring cloud",
    ],
    "express.js": ["expressjs", "express node", "express framework"],
    "nestjs": ["nest.js", "nest js"],
    "asp.net": [
        "asp net", "dotnet", ".net core",
        "dotnet core", "aspnetcore",
    ],
    "grpc": ["protocol buffers", "protobuf", "grpc service"],
    "rest api": [
        "rest", "restful api", "restful", "rest apis",
        "rest services", "http api", "restful services", "web api",
    ],
    "rabbitmq": ["rabbit mq", "message queue", "amqp"],

    # ══════════════════════════════════════════════════════════════════════════
    # FRONTEND
    # ══════════════════════════════════════════════════════════════════════════

    "react": ["reactjs", "react.js", "react hooks", "react library"],
    "angular": [
        "angularjs", "angular 2+", "angular framework",
        "angular cli",
    ],
    "vue": ["vuejs", "vue.js", "vue 3", "vue framework"],
    "nextjs": ["next.js", "next js", "next.js framework"],
    "html": ["html5", "hypertext markup language"],
    "css": ["css3", "cascading style sheets"],
    "tailwind": ["tailwind css"],
    "redux": ["redux toolkit", "state management redux"],

    # ══════════════════════════════════════════════════════════════════════════
    # VERSION CONTROL & COLLABORATION
    # ══════════════════════════════════════════════════════════════════════════

    "git": [
        "github", "gitlab", "bitbucket", "version control",
        "git version control", "git workflow",
    ],
    "jira": ["atlassian jira", "project tracking jira"],
    "swagger": ["openapi", "api documentation", "swagger ui"],

    # ══════════════════════════════════════════════════════════════════════════
    # ARCHITECTURE / SYSTEM DESIGN
    # ══════════════════════════════════════════════════════════════════════════

    "system design": [
        "system architecture", "software architecture",
        "high level design", "hld", "low level design", "lld",
        "distributed system design", "scalable architecture",
    ],
    "microservices": [
        "microservice architecture", "microservices architecture",
        "service oriented architecture", "soa",
    ],
    "distributed systems": [
        "distributed computing", "distributed architecture",
        "consensus algorithms", "raft", "paxos",
    ],
    "event-driven architecture": [
        "eda", "event driven", "event sourcing", "cqrs",
    ],
    "design patterns": [
        "software design patterns", "oop patterns",
        "solid principles", "clean architecture",
    ],
    "sre": [
        "site reliability engineering", "site reliability engineer",
        "reliability engineering", "on-call", "incident management",
    ],
    "devops": [
        "development operations", "site reliability",
        "platform engineering", "devsecops",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # STATISTICS / MATH
    # ══════════════════════════════════════════════════════════════════════════

    "statistics": [
        "statistical modeling", "probability", "hypothesis testing",
        "bayesian statistics", "inferential statistics",
        "descriptive statistics",
    ],
    "linear algebra": [
        "matrix operations", "vector spaces", "eigenvalues",
        "matrix decomposition", "svd",
    ],
    "optimization": [
        "gradient descent", "sgd", "adam optimizer",
        "convex optimization", "mathematical optimization",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # SOFT SKILLS / METHODOLOGIES
    # ══════════════════════════════════════════════════════════════════════════

    "software engineering": [
        "software development", "swe", "software engineer",
        "software design", "clean code", "code review",
        "software craftsmanship",
    ],
    "project management": [
        "pm", "project planning", "project coordination",
        "project delivery", "program management",
    ],
    "agile": [
        "agile methodology", "agile development", "scrum agile",
        "kanban", "agile practices",
    ],
    "scrum": [
        "scrum master", "sprint planning", "agile scrum",
        "scrum framework",
    ],
    "leadership": [
        "team leadership", "people management",
        "engineering management", "team lead", "people leader",
    ],
    "data analysis": [
        "data analytics", "business analytics", "analytical skills",
        "statistical analysis", "data interpretation",
    ],
    "technical writing": [
        "documentation", "technical documentation",
        "api docs", "runbook",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # BUSINESS / DOMAIN TOOLS
    # ══════════════════════════════════════════════════════════════════════════

    "excel": [
        "microsoft excel", "ms excel", "spreadsheet",
        "vlookup", "pivot table", "excel formulas",
    ],
    "tableau": [
        "tableau desktop", "tableau server",
        "data visualization tableau",
    ],
    "power bi": ["powerbi", "microsoft power bi", "power bi desktop"],
    "looker": ["looker studio", "google looker", "looker bi"],
    "seo": ["search engine optimization", "organic search", "seo strategy"],
    "sales": [
        "business development", "bd", "sales development",
        "revenue growth", "b2b sales",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN-SPECIFIC: SEARCH / RANKING / RECSYS
    # ══════════════════════════════════════════════════════════════════════════

    "search relevance": [
        "relevance engineering", "search quality",
        "result quality", "search ux",
        "query relevance", "document relevance",
    ],
    "candidate ranking": [
        "talent ranking", "resume ranking", "cv ranking",
        "applicant ranking", "jd matching", "job matching",
    ],
    "ranking pipeline": [
        "ranking system", "ranking infrastructure",
        "scoring pipeline", "ranking model pipeline",
    ],
    "query understanding": [
        "query analysis", "query expansion", "query classification",
        "intent detection", "query parsing",
    ],
    "index serving": [
        "search index", "inverted index", "forward index",
        "index pipeline",
    ],
    "mlops for search": [
        "search ml platform", "retrieval infrastructure",
        "ml serving for search",
    ],

    # ══════════════════════════════════════════════════════════════════════════
    # COMPLIANCE / SECURITY
    # ══════════════════════════════════════════════════════════════════════════

    "oauth": ["oauth2", "openid connect", "oidc", "oauth 2.0"],
    "ssl/tls": ["https", "ssl", "tls", "tls encryption"],
    "compliance": ["regulatory compliance", "gdpr", "hipaa", "sox"],

    # ══════════════════════════════════════════════════════════════════════════
    # EMERGING / NICHE
    # ══════════════════════════════════════════════════════════════════════════

    "blockchain": [
        "smart contracts", "ethereum", "web3",
        "defi", "distributed ledger",
    ],
    "iot": [
        "internet of things", "embedded systems",
        "firmware", "edge computing", "edge ai",
    ],
    "quantum computing": ["quantum algorithms", "qiskit", "quantum ml"],
}


# ──────────────────────────────────────────────────────────────────────────────
# JD REQUIREMENTS MATCHER
# skill_tokens: matched against candidate skills[].name (normalized, lowercased)
# desc_terms:   matched against career_history[].description (lowercased substring)
# ──────────────────────────────────────────────────────────────────────────────
JD_REQUIREMENTS = {

    "embeddings_retrieval": {
        "skill_tokens": {
            "sentence transformers", "faiss", "embeddings", "semantic search",
            "vector search", "dense retrieval", "bi-encoder", "cross-encoder",
            "bge", "e5", "openai embeddings", "cohere embeddings",
            "all-minilm", "mpnet", "dpr", "colbert", "annoy", "hnswlib",
            "embedding", "text embeddings", "neural retrieval",
        },
        "desc_terms": {
            "embedding", "semantic search", "dense retrieval", "sentence-transformer",
            "bi-encoder", "faiss", "annoy", "hnsw", "retrieval system",
            "vector index", "embedding drift", "index refresh",
        },
    },

    "vector_db_hybrid_search": {
        "skill_tokens": {
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "pgvector", "chroma", "lancedb", "vespa",
            "hybrid search", "vector database", "redis", "typesense",
            "zilliz", "marqo", "turbopuffer",
        },
        "desc_terms": {
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "pgvector", "vector database", "hybrid search",
            "sparse dense", "bm25 embedding", "keyword semantic",
        },
    },

    "python": {
        "skill_tokens": {
            "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
            "numpy", "pandas", "fastapi", "flask", "django", "pydantic",
            "hugging face", "hugging face transformers", "transformers",
            "langchain", "llamaindex", "haystack", "ray", "dask",
        },
        "desc_terms": {
            "python", "pytorch", "tensorflow", ".py", "pip install",
            "sklearn", "pandas", "numpy", "jupyter",
        },
    },

    "ranking_eval_frameworks": {
        "skill_tokens": {
            "learning to rank", "information retrieval", "ndcg", "mrr", "map",
            "a/b testing", "ranking", "bm25", "reranking", "flashrank",
            "evaluation", "offline evaluation", "relevance evaluation",
            "trec", "beir", "msmarco",
        },
        "desc_terms": {
            "ndcg", "mrr", "map", "recall@", "precision@", "a/b test",
            "ranking evaluation", "evaluation framework", "offline eval",
            "online eval", "learning to rank", "relevance label",
            "click-through", "human judgment", "benchmark", "eval harness",
            "held-out eval", "ranking metric",
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# LAYER KEYWORD VOCABULARIES (used by src/layers.py)
# ──────────────────────────────────────────────────────────────────────────────

# In-code fictional company blacklist (backup to SQLite KB) — L1
FICTIONAL_COMPANIES = {
    "dunder mifflin", "hooli", "acme corp", "acme corporation", "initech",
    "pied piper", "vehement capital", "globex", "soylent corp", "umbrella corp",
    "umbrella corporation", "stark industries", "wayne enterprises", "wonka industries",
    "cyberdyne systems", "weyland-yutani", "tyrell corporation", "oscorp",
    "massive dynamic", "aperture science", "black mesa", "vault-tec",
}

# Ordered longest-first so "pvt ltd" is stripped before "ltd" — L1
_COMPANY_SUFFIXES = (
    "pvt ltd", "private limited", "pvt. ltd.", "pvt. ltd", "pvt ltd.",
    "corporation", "incorporated", "limited liability company",
    "corp.", "corp", "inc.", "inc", "ltd.", "ltd", "llc", "llp",
    "plc", "gmbh", "s.a.", "s.a", "nv", "bv", "ag", "kg",
)

# Proficiency-level keywords → numeric weight — L1c
_PROFICIENCY_MAP: dict[str, float] = {
    "beginner": 0.1, "intermediate": 0.2, "advanced": 0.3, "expert": 0.4
}

# ── Keyword vocabularies for cols 11–13 (L2) ─────────────────────────────────
_PRODUCTION_KW: dict[str, float] = {
    "deployed": 0.20, "deploy": 0.15, "deployment": 0.15,
    "shipped": 0.18, "ship": 0.15, "shipping": 0.15,
    "production": 0.20,
    "real users": 0.25, "live users": 0.20, "end users": 0.15,
    "at scale": 0.15, "large scale": 0.15,
    "a/b test": 0.15, "ab test": 0.15, "ab testing": 0.15,
    "rollout": 0.12, "launched": 0.15, "launch": 0.12,
    "serving": 0.12,
    "qps": 0.20,
    "p95 latency": 0.22,
    "p99 latency": 0.22,
    "ranking pipeline": 0.22,
    "retrieval system": 0.22,
    "search at scale": 0.25,
    "online serving": 0.20,
}

_ARCHITECTURE_KW: dict[str, float] = {
    "designed system": 0.20, "system design": 0.18,
    "architected": 0.20, "architecture": 0.12,
    "distributed system": 0.15, "distributed": 0.10,
    "microservices": 0.12, "microservice": 0.12,
    "scalable": 0.10, "scalability": 0.10,
    "high availability": 0.12, "fault tolerant": 0.12,
    "low latency": 0.10, "throughput": 0.10,
    "system architecture": 0.18, "pipeline design": 0.12,
}

_TESTING_EVAL_KW: dict[str, float] = {
    "ndcg": 0.25, "mrr": 0.25,
    "mean average precision": 0.25,
    "a/b test": 0.20, "ab test": 0.20, "ab testing": 0.20,
    "recall@k": 0.20, "recall@": 0.18,
    "benchmark": 0.15,
    "precision@": 0.15, "precision": 0.12,
    "evaluation": 0.10, "metrics": 0.10,
    "beir": 0.20, "trec": 0.20,
    "offline evaluation": 0.15, "online evaluation": 0.15,
    "f1 score": 0.12,
}

# ── Tools vocabulary for col 18 — weighted by stack relevance (L2) ───────────
# CV-specific tools (opencv, torchvision, detectron2, yolo, albumentations …)
# are deliberately excluded.  Categories and weights:
#   Embedding tools          → 1.0–0.90  (highest: core IR capability)
#   Vector DB / ANN indexes  → 0.85–0.70  (high)
#   Ranking / reranking      → 0.90–0.80  (high)
#   NLP & LLM models         → 0.70–0.50  (medium)
#   Deployment / MLOps       → 0.50–0.40  (medium)
#   AI orchestration         → 0.30–0.25  (low)
#   Cloud ML platforms       → 0.20       (lowest)

# Embedding tools — highest weight
_EMBEDDING_TOOLS: dict[str, float] = {
    "sentence-transformers": 1.00,
    "fastembed":             1.00,
    "huggingface":           0.90,   # HF Hub / Transformers ecosystem
    "cohere":                0.90,   # Cohere embed + rerank API
}

# Vector database / ANN index tools — high weight
_VECTOR_DB_TOOLS: dict[str, float] = {
    "faiss":         0.85,
    "qdrant":        0.85,
    "milvus":        0.85,
    "weaviate":      0.85,
    "pinecone":      0.85,
    "chromadb":      0.85,
    "pgvector":      0.85,
    "lancedb":       0.85,
    "vespa":         0.85,
    "elasticsearch": 0.80,
    "opensearch":    0.80,
    "solr":          0.70,
}

# Ranking / reranking tools — high weight
_RANKING_TOOLS: dict[str, float] = {
    "flashrank":     0.90,
    "colbert":       0.90,
    "cross-encoder": 0.85,
    "bm25":          0.80,
    "reranker":      0.80,
}

# NLP and LLM model tools — medium weight
_NLP_MODEL_TOOLS: dict[str, float] = {
    "bert":        0.70,
    "llama":       0.70,
    "spacy":       0.65,
    "mistral":     0.65,
    "openai":      0.65,   # GPT / embeddings API
    "gensim":      0.60,
    "pytorch":     0.60,
    "anthropic":   0.60,
    "nltk":        0.55,
    "tensorflow":  0.55,
    "keras":       0.50,
    "onnx":        0.50,
    "tensorrt":    0.50,
}

# Deployment / serving / MLOps tools — medium weight
_DEPLOYMENT_TOOLS: dict[str, float] = {
    "docker":     0.50,
    "kubernetes": 0.50,
    "triton":     0.50,
    "bentoml":    0.50,
    "ray":        0.45,
    "mlflow":     0.45,
    "kubeflow":   0.45,
    "wandb":      0.40,
    "dvc":        0.40,
}

# AI orchestration frameworks — low weight
_ORCHESTRATION_TOOLS: dict[str, float] = {
    "langchain":  0.30,
    "llamaindex": 0.30,
    "haystack":   0.30,
    "crewai":     0.25,
}

# Cloud ML platforms — lowest weight
_CLOUD_TOOLS: dict[str, float] = {
    "sagemaker":  0.20,
    "bigquery":   0.20,
    "databricks": 0.20,
    "snowflake":  0.20,
    "redshift":   0.20,
}

# Merged lookup: tool_name → weight  (used for tools_score computation) — L2
_TOOLS_WEIGHTED: dict[str, float] = {
    **_EMBEDDING_TOOLS,
    **_VECTOR_DB_TOOLS,
    **_RANKING_TOOLS,
    **_NLP_MODEL_TOOLS,
    **_DEPLOYMENT_TOOLS,
    **_ORCHESTRATION_TOOLS,
    **_CLOUD_TOOLS,
}

# IR domain terms for condition d scoring (0/0.5/1.0 based on hit count) — L2
_IR_DOMAIN_TERMS: frozenset = frozenset({
    "bm25", "faiss", "rerank", "retrieval", "ranking",
    "vector search", "embedding",
})

# ── Consulting firms / industries for col 19 (L2) ────────────────────────────
_CONSULTING_FIRMS: frozenset = frozenset({
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "mindtree", "hexaware", "ltimindtree", "lti mindtree",
    "l&t infotech", "kpmg", "deloitte", "pwc", "ernst & young", "ey",
    "mckinsey", "boston consulting group", "bcg", "bain",
    "niit technologies", "zensar", "cyient", "persistent systems",
})

_CONSULTING_INDUSTRIES: frozenset = frozenset({
    "it services", "consulting", "professional services", "outsourcing",
    "it consulting", "management consulting", "technology services",
    "bpo", "kpo", "ites", "information technology services",
})

# ── Research publication keywords for col 17 (L2) ────────────────────────────
_RESEARCH_KW: frozenset = frozenset({
    "paper", "published", "arxiv", "proceedings", "neurips", "nips",
    "icml", "iclr", "cvpr", "acl", "emnlp", "journal", "preprint",
    "citation", "dissertation", "thesis", "conference paper", "research paper",
    "peer reviewed", "peer-reviewed",
})

# ── JD-relevant title keywords (NLP / IR / Applied AI — NOT CV / Speech) — L3 ─
# Titles matching these indicate the candidate's role is closely aligned with
# the JD target (Senior AI Engineer, IR/NLP stack).  Used in L3 for +0.02 bonus.
# CV-specific titles (computer vision, vision scientist, etc.) are intentionally
# excluded — they signal a different domain.
_JD_REQ_TITLES: frozenset = frozenset({
    # Core AI / ML engineering
    "ai engineer", "senior ai engineer", "staff ai engineer", "principal ai engineer",
    "ml engineer", "machine learning engineer", "senior ml engineer",
    "principal ml engineer", "staff ml engineer",
    # Applied AI / Applied Science
    "applied scientist", "applied ai engineer", "applied ai",
    "applied machine learning engineer", "applied research scientist",
    # NLP / Information Retrieval / Search
    "nlp engineer", "nlp scientist", "natural language processing engineer",
    "search engineer", "senior search engineer", "search scientist",
    "ranking engineer", "relevance engineer", "information retrieval engineer",
    # Research Engineering (AI/ML-focused, not domain-locked; "ai research engineer"
    # excluded — too broad, fires on CV/speech candidates)
    "research engineer", "senior research engineer",
    "ml research engineer",
    # AI / ML leadership
    "ai tech lead", "ml tech lead", "ai lead", "ml lead",
    "ai architect", "ml architect",
})

# CV / Speech domain skills and title keywords — candidates whose profile is
# dominated by these are hard-penalized by the donts layer regardless of
# embedding similarity, which is too noisy at 384-dim to resolve domains — L4
_CV_SPEECH_SKILLS: frozenset = frozenset({
    "asr", "tts", "speech recognition", "automatic speech recognition",
    "text to speech", "text-to-speech", "computer vision", "opencv",
    "yolo", "object detection", "image classification", "image segmentation",
    "detectron", "torchvision", "face recognition", "face detection",
    "optical character recognition", "ocr",
})
_CV_SPEECH_TITLE_KW: frozenset = frozenset({
    "computer vision", "cv engineer", "vision scientist", "vision engineer",
    "speech recognition", "asr engineer", "tts engineer", "speech engineer",
    "image processing", "visual recognition",
})


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

# Pre-build a reverse index: synonym → canonical  (populated once at import)
_SYNONYM_TO_CANONICAL: dict[str, str] = {}
for _canonical, _synonyms in SKILL_SYNONYMS.items():
    for _syn in _synonyms:
        _SYNONYM_TO_CANONICAL[_syn.lower()] = _canonical


def expand_skill(term: str) -> list[str]:
    """
    Return all recognized forms (canonical + synonyms) for a skill term.

    Lookup order:
      1. Exact canonical match
      2. Synonym reverse-index lookup
      3. Fallback: return [term.lower()] (not in dictionary)

    Examples:
        expand_skill("Qdrant")     → ["vector databases", "qdrant", "qdrant vector", ...]
        expand_skill("BM25")       → ["hybrid search infrastructure", "bm25", ...]
        expand_skill("FAISS")      → ["embeddings-based retrieval systems", "faiss", ...]
        expand_skill("NDCG")       → ["evaluation frameworks for ranking systems", ...]
        expand_skill("LTR")        → ["learning to rank", "ltr", ...]
        expand_skill("LLM")        → ["large language model", "llm", "llms", ...]
        expand_skill("Python")     → ["python", "python3", "python 3", "py", ...]
    """
    t = term.strip().lower()
    if not t:
        return []
    # 1. Direct canonical match
    if t in SKILL_SYNONYMS:
        return [t] + [s.lower() for s in SKILL_SYNONYMS[t]]
    # 2. Reverse-index lookup
    canonical = _SYNONYM_TO_CANONICAL.get(t)
    if canonical:
        return [canonical] + [s.lower() for s in SKILL_SYNONYMS[canonical]]
    # 3. Not in dictionary
    return [t]


def skill_match_score(
    candidate_skills: list[str],
    candidate_text: str,
    jd_required: list[str],
    jd_bonus: list[str] | None = None,
) -> dict:
    """
    Score a candidate's skill match against JD required and bonus skills.

    Args:
        candidate_skills: List of skill strings from candidate profile
        candidate_text:   Full candidate text (skills + work descriptions)
        jd_required:      List of required skills from jd.json
        jd_bonus:         List of bonus skills from jd.json (optional)

    Returns:
        dict with keys:
            score:           float 0.0–1.0 (weighted: required 0.8 + bonus 0.2)
            required_matched: list of matched required skills
            required_missing: list of unmatched required skills
            bonus_matched:    list of matched bonus skills
            match_ratio:      float (matched / total required)
    """
    if jd_bonus is None:
        jd_bonus = []

    candidate_text_lower = candidate_text.lower()
    candidate_skills_lower = [s.lower() for s in candidate_skills]

    def _is_matched(jd_skill: str) -> bool:
        jd_aliases = set(expand_skill(jd_skill))
        # Bidirectional: expand candidate skill too, check alias overlap
        for cand_skill in candidate_skills_lower:
            cand_aliases = set(expand_skill(cand_skill))
            if jd_aliases & cand_aliases:
                return True
        # Also check raw text against all JD aliases
        for alias in jd_aliases:
            if alias in candidate_text_lower:
                return True
        return False

    required_matched = [s for s in jd_required if _is_matched(s)]
    required_missing = [s for s in jd_required if s not in required_matched]
    bonus_matched = [s for s in jd_bonus if _is_matched(s)]

    req_score = len(required_matched) / len(jd_required) if jd_required else 0.0
    bonus_score = len(bonus_matched) / len(jd_bonus) if jd_bonus else 0.0
    final_score = req_score * 0.8 + bonus_score * 0.2

    return {
        "score": round(final_score, 4),
        "required_matched": required_matched,
        "required_missing": required_missing,
        "bonus_matched": bonus_matched,
        "match_ratio": round(req_score, 4),
    }