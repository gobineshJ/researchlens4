"""
Central settings for the whole backend. Nothing here should require
touching pipeline code to tune — crawl limits, chunk size, model
names, all live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_FILTER_MODEL = "openai/gpt-oss-20b"       # fast, cheap — used for many small relevance checks
GROQ_GENERATION_MODEL = "openai/gpt-oss-120b"  # higher quality — used once per report
# Note: llama-3.1-8b-instant and llama-3.3-70b-versatile (the original choices)
# were decommissioned by Groq on 2026-08-16. These are Groq's recommended
# replacements as of this update -- check console.groq.com/docs/deprecations
# if you hit a 404 again in future, since Groq's free-tier lineup does change.

# --- Crawl / discovery ---
CRAWL_MAX_DEPTH = 2
CRAWL_MAX_PAGES = 15
CRAWL_SAME_DOMAIN_ONLY = True
REQUEST_TIMEOUT = 8
USER_AGENT = "ResearchLens-Bot/0.1 (training project; respects robots.txt)"

# --- Chunking ---
CHUNK_SIZE_TOKENS = 650
CHUNK_OVERLAP_RATIO = 0.15

# --- Retrieval ---
TOP_K_CHUNKS = 5              # used when there's exactly one source document
TOP_K_CHUNKS_PER_SOURCE = 5   # used per document when there are multiple --
                              # guarantees every uploaded file/page gets
                              # representation instead of one global top-k
                              # that can silently drop an entire document

# --- ChromaDB ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = "researchlens"

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Output ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./generated_reports")
