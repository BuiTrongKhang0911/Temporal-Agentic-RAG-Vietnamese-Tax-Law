TEMPORAL-AWARE AGENTIC RAG FOR VIETNAMESE TAX-LAW QUESTION ANSWERING
==================================================================

1. PROJECT CONTENTS
-------------------

- source_code/backend: FastAPI backend, Agent 1-5 workflow, Structured
  Retriever, temporal filtering, evidence verification, and calculation tools.
- source_code/frontend: browser-based chat interface.
- corpus_construction: Step 1-3 version-aware corpus construction pipeline,
  47 legal documents, per-document artifacts, corpus manifest, and the Qdrant
  snapshot download link.
- experiments/retriever: 100-question Retriever test set, evaluation runners,
  and reported results.
- experiments/end_to_end: 300-question end-to-end test set, baseline and
  proposed-system runners, evaluation code, and reported results.
- experiments/runtime: frozen and self-contained runtime used only to
  reproduce the experiments reported in the thesis.

Production uses Gemini for all five agents. The Qwen implementation under
experiments/runtime is retained only to reproduce the reported model
comparison and is not imported by source_code. For that experiment, Qwen 3.5
27B is hosted with Ollama on an external GPU platform. The Ollama endpoint is
exposed through ngrok, and the experiment runner calls it through the ngrok
API URL configured in .env.

2. SYSTEM REQUIREMENTS
----------------------

- Windows 10/11 or another operating system capable of running Python 3.12.
- Python 3.12.
- Qdrant Server v1.18.2 available at localhost:6333 by default. Use this exact
  server version when restoring the supplied snapshot to avoid snapshot
  compatibility errors.
- A valid Google Gemini API key.
- Internet access on the first run to download Hugging Face models, unless the
  models already exist in the local cache.
- An NVIDIA CUDA-compatible GPU is optional. The reranker falls back to CPU.

3. INSTALLATION AND ENVIRONMENT CONFIGURATION
---------------------------------------------

Run all commands from the repository root.

Install dependencies:

    python -m pip install -r requirements.txt

Create .env from .env.example. For the production application, corpus
construction, and Gemini experiments, the user only needs to replace:

    GOOGLE_API_KEY=your_google_api_key_here

The remaining Gemini and corpus model fields already contain the submitted
defaults:

    E2E_PROPOSED_GEMINI_MODEL=models/gemini-3.1-flash-lite
    E2E_BASELINE_GEMINI_MODEL=models/gemini-3.1-flash-lite
    E2E_JUDGE_GEMINI_MODEL=models/gemini-3.5-flash-lite
    CORPUS_ANALYSIS_GEMINI_MODEL=models/gemini-3.1-flash-lite
    SAC_TIER1_GEMINI_MODEL=models/gemini-3.1-flash-lite
    SAC_TIER2_GEMINI_MODEL=models/gemini-3.1-flash-lite
Production requests do not use E2E_LLM_CALL_DELAY_SECONDS; this delay applies
only to experiment runners. Retrieval and application settings not listed in
.env use the defaults defined in source_code/backend/config.py.

The Qwen configuration is optional. To reproduce the Qwen experiment, run
Qwen 3.5 27B with Ollama on a GPU-providing platform, expose the Ollama API
through ngrok, and replace the URL:

    QWEN_OLLAMA_URL=https://your-ngrok-domain.ngrok-free.app

QWEN_OLLAMA_TOKEN is optional. Set it to the same Bearer token configured by
the external proxy (for example, API_TOKEN in the supplied Kaggle proxy
script). Leave it empty when connecting to an Ollama endpoint without that
authentication layer:

    QWEN_OLLAMA_TOKEN=

The remaining QWEN_* fields in .env.example already match the submitted
Qwen 3.5 27B configuration.

Do not commit .env because it contains the API key.

4. CORPUS SETUP
---------------

Recommended method: restore the ready-built Qdrant snapshot.

The supplied snapshot was created with Qdrant Server v1.18.2. Start Qdrant
with the same version before restoring it. For Docker, the corresponding image
is qdrant/qdrant:v1.18.2.

1. Open corpus_construction/qdrant_snapshot/Snapshot-GoogleDrive-Link.txt.
2. Download the snapshot and place it in corpus_construction/qdrant_snapshot.
3. Start Qdrant.
4. Run:

    cd corpus_construction
    python restore_snapshot.py
    cd ..

Use the following command only when intentionally replacing an existing
legal_documents collection:

    python corpus_construction/restore_snapshot.py --force

To validate the supplied artifacts without loading models or writing Qdrant:

    python corpus_construction/step3_embedding_qdrant_versioned.py --validate-only

To rebuild the corpus from the source documents:

    python corpus_construction/step1_markdown_hierarchy.py
    python corpus_construction/step2_sac_generation.py
    python corpus_construction/step3_embedding_qdrant_versioned.py

The normal Step 3 command writes to QDRANT_COLLECTION. For a test ingestion,
set QDRANT_COLLECTION to a separate collection such as legal_documents_test.

5. RUNNING THE PRODUCTION APPLICATION
-------------------------------------

Ensure that Qdrant is running and the legal_documents collection is available.
Then run:

    python -m source_code.backend.run

Open the browser at:

    http://127.0.0.1:8000

The backend health endpoint is:

    http://127.0.0.1:8000/api/health

6. RUNNING THE RETRIEVER EXPERIMENTS
------------------------------------

Three-case smoke tests:

    python -m experiments.retriever.run_standard_baselines --limit 3
    python -m experiments.retriever.run_proposed_retriever --limit 3

Full 100-question runs:

    python -m experiments.retriever.run_standard_baselines
    python -m experiments.retriever.run_proposed_retriever

The standard runner evaluates Dense, BM25, Hybrid RRF, and Hybrid + Rerank.
The proposed runner evaluates Full Provision and Full Structured Retriever.
New runs create latest_* files, while reported_* files contain the submitted
results and must not be overwritten.

7. RUNNING THE END-TO-END EXPERIMENTS
-------------------------------------

Three-case smoke tests:

    python -m experiments.end_to_end.run_standard_baselines --limit 3
    python -m experiments.end_to_end.run_proposed_system --system gemini --limit 3

Full 300-question runs:

    python -m experiments.end_to_end.run_standard_baselines
    python -m experiments.end_to_end.run_proposed_system --system gemini

Each end-to-end runner performs inference, LLM judging, temporal evaluation,
per-case scoring, and summary aggregation. Rerun outputs are written under
experiments/end_to_end/result/rerun. The reported_* files contain the results
used in the thesis.

The optional Qwen reproduction requires a running Qwen 3.5 27B Ollama service
on an external GPU platform. Expose that service through ngrok and configure
QWEN_OLLAMA_URL in .env. Configure QWEN_OLLAMA_TOKEN only if the proxy requires
Bearer authentication, then run:

    python -m experiments.end_to_end.run_proposed_system --system qwen

8. IMPORTANT NOTES
------------------

- __init__.py files are required Python package markers and should be committed.
- Cache directories, virtual environments, build outputs, temporary rerun
  results, snapshots, archives, and .env are excluded through .gitignore.
- The source repository is substantially smaller than 2 GB, so link.txt for a
  separate source_code.zip is not required.
- The Qdrant snapshot is distributed separately through its Google Drive link
  because it is a generated database artifact.
