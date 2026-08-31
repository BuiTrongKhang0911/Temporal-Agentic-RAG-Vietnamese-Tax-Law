# Temporal-Aware Agentic RAG for Vietnamese Tax-Law Question Answering

This repository contains a Vietnamese tax-law question-answering system and
its reproducibility package. The system combines a version-aware legal corpus,
deterministic temporal filtering, a Structured Retriever, five Gemini agents,
fact-level evidence verification, and grounded answer generation.

## Main Components

- `source_code/`: production FastAPI backend, browser frontend, Agent 1-5
  workflow, Structured Retriever, and deterministic calculation tools.
- `corpus_construction/`: Step 1-3 corpus pipeline, 47 source documents,
  document artifacts, corpus manifest, and Qdrant snapshot link.
- `experiments/`: Retriever and end-to-end test sets, self-contained runners,
  frozen experiment runtime, and the reported results.

## Technologies

Python 3.12, FastAPI, Gemini, Qdrant, BAAI/bge-m3, BM25 with Vietnamese word
segmentation, and BAAI/bge-reranker-base.

See `readme.txt` for environment configuration, corpus setup, execution, and
experiment reproduction instructions.
