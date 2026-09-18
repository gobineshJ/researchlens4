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
GROQ_FILTER_MODEL = "llama-3.1-8b-instant"       # fast, cheap — used for many small relevance checks
GROQ_GENERATION_MODEL = "llama-3.3-70b-versatile"  # higher quality — used once per report

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
TOP_K_CHUNKS = 5

# --- ChromaDB ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = "researchlens"

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Output ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./generated_reports")
