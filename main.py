import os
import json
import re
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# ----------------------
# Logging
# ----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("course_ai")

# ----------------------
# Config
# ----------------------
DATA_PATH = "courses.txt"
EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMB_CACHE_DIR = "./emb_cache"
FAISS_INDEX_PATH = Path(EMB_CACHE_DIR) / "courses.faiss"
EMB_ARRAY_PATH = Path(EMB_CACHE_DIR) / "course_embeddings.npy"
METADATA_PATH = Path(EMB_CACHE_DIR) / "metadata.json"
TOP_K = 5

Path(EMB_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# ----------------------
# Pydantic models
# ----------------------
class CourseHit(BaseModel):
    title: str
    short_description: Optional[str]
    duration_minutes: Optional[int]
    language: Optional[str]
    instructors: List[str]
    what_you_will_learn: List[str]
    score: float

class ChatResponse(BaseModel):
    results: List[CourseHit]
    filters_used: Dict[str, Any]

class QueryRequest(BaseModel):
    question: str


# ----------------------
# Helpers
# ----------------------
def load_course_json(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Elasticsearch response → extract _source from each hit
    if isinstance(data, dict) and "hits" in data and "hits" in data["hits"]:
        hits = data["hits"]["hits"]
        return [h["_source"] for h in hits if "_source" in h]

    raise ValueError("Unexpected file format: expected Elasticsearch response with hits.hits")



def build_text_for_embedding(item: Dict[str, Any]) -> str:
    title = item.get("title", "")
    desc = item.get("headline") or item.get("description") or ""
    desc = re.sub(r"<[^>]+>", " ", desc)
    what = item.get("what_you_will_learn") or []
    if isinstance(what, list):
        what_text = " ".join([str(x) for x in what])
    else:
        what_text = str(what)
    return f"{title}\n\n{desc}\n\nWhat you'll learn: {what_text}"


# ----------------------
# Filter extraction
# ----------------------
DURATION_REGEX = re.compile(r"(\d+)\s*(hours|hour|hrs|hr|minutes|minute|min)?", re.IGNORECASE)
DIFFICULTY_KEYWORDS = {
    "Beginner": ["beginner", "principiante", "مبتدئ", "débutant", "初級"],
    "Intermediate": ["intermediate", "intermedio", "متوسط", "intermédiaire", "中級"],
    "Advanced": ["advanced", "avanzado", "متقدم", "avancé", "上級"],
    "All": ["all","All Levels","all levels", "all levels of difficulty", "all levels of experience", "all levels of expertise"],
}

def extract_filters(question: str) -> Dict[str, Any]:
    q = question.lower()
    filters: Dict[str, Any] = {}

    # duration
    m = DURATION_REGEX.search(q)
    if m:
        num = int(m.group(1))
        unit = m.group(2) or "hours"
        if "min" in unit.lower():
            minutes = num
        else:
            minutes = num * 60
        filters["max_duration_minutes"] = minutes

    # difficulty
    for diff, kws in DIFFICULTY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            filters["difficulty_level"] = diff
            break

    # instructor
    instr_match = re.search(r"\bby\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", question)
    if instr_match:
        filters["instructor"] = instr_match.group(1)

    # language
    if "spanish" in q or "español" in q:
        filters["language"] = "es"
    elif "english" in q:
        filters["language"] = "en"
    elif "arabic" in q or "عربي" in q:
        filters["language"] = "ar"

    logger.debug("Extracted filters: %s", filters)
    return filters


# ----------------------
# Retriever
# ----------------------
class Retriever:
    def __init__(self):
        logger.info("Loading embedding model: %s", EMB_MODEL_NAME)
        self.model = SentenceTransformer(EMB_MODEL_NAME)
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None
        self._init_index()

    def _init_index(self):
        if FAISS_INDEX_PATH.exists() and EMB_ARRAY_PATH.exists() and METADATA_PATH.exists():
            logger.info("Loading cached embeddings and metadata from %s", EMB_CACHE_DIR)
            loaded = np.load(str(EMB_ARRAY_PATH))
            # Ensure embeddings array is 2D: (num_vectors, dim)
            if loaded.ndim == 1:
                if loaded.size == 0:
                    dim = self.model.get_sentence_embedding_dimension()
                    self.embeddings = np.empty((0, dim), dtype=np.float32)
                else:
                    self.embeddings = loaded.reshape(1, -1).astype(np.float32)
            elif loaded.ndim == 2:
                self.embeddings = loaded.astype(np.float32)
            else:
                raise ValueError("Cached embeddings have invalid shape")
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Cached embeddings shape: %s; metadata count: %d", self.embeddings.shape, len(self.metadata))
            d = self.embeddings.shape[1] if self.embeddings.shape[0] > 0 else self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(d)
            if self.embeddings.shape[0] > 0:
                logger.debug("Normalizing and adding %d cached vectors to FAISS (dim=%d)", self.embeddings.shape[0], d)
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings)
            logger.info("FAISS index initialized with %d vectors (dim=%d)", self.index.ntotal, d)
        else:
            self.build_index_from_data()

    def build_index_from_data(self):
        logger.info("Building index from data at %s", DATA_PATH)
        items = load_course_json(DATA_PATH)
        texts = []
        self.metadata = []
        for item in tqdm(items, desc="Preparing courses"):
            texts.append(build_text_for_embedding(item))
            self.metadata.append({
                "title": item.get("title"),
                "short_description": re.sub(r"<[^>]+>", " ", item.get("headline") or item.get("description") or ""),
                "duration_minutes": item.get("content_duration_minutes"),
                "language": item.get("language"),
                "instructors": item.get("instructors") or [],
                "what_you_will_learn": item.get("what_you_will_learn") or [],
            })
        logger.info("Prepared %d items; encoding embeddings...", len(texts))
        if len(texts) == 0:
            dim = self.model.get_sentence_embedding_dimension()
            embeddings = np.empty((0, dim), dtype=np.float32)
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if embeddings.shape[0] > 0:
                logger.debug("Normalizing embeddings with shape %s", embeddings.shape)
                faiss.normalize_L2(embeddings)
        self.embeddings = embeddings
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        np.save(str(EMB_ARRAY_PATH), embeddings)
        d = embeddings.shape[1] if embeddings.shape[0] > 0 else self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(d)
        if embeddings.shape[0] > 0:
            logger.info("Adding %d vectors to FAISS index (dim=%d)", embeddings.shape[0], d)
            self.index.add(embeddings)
        logger.info("Index built. FAISS ntotal=%d", self.index.ntotal)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

    def build_filter_mask(self, filters: Dict[str, Any]) -> Optional[np.ndarray]:
        if not filters:
            return None
        mask = np.ones(len(self.metadata), dtype=bool)
        for i, md in enumerate(self.metadata):
            # Difficulty: check if the difficulty keyword exists in short_description
            if "difficulty_level" in filters:
                diff = filters["difficulty_level"].lower()
                desc = (md.get("short_description") or "").lower()
                if diff != "all" and diff not in desc:
                    mask[i] = False

            # Instructor
            if "instructor" in filters:
                instrs = [s.lower() for s in md.get("instructors", [])]
                if not any(filters["instructor"].lower() in s for s in instrs):
                    mask[i] = False

            # Duration
            if "max_duration_minutes" in filters:
                dur = md.get("duration_minutes")
                if dur and int(dur) > filters["max_duration_minutes"]:
                    mask[i] = False

            # Language: match prefix of language code (e.g., 'ar' in 'ar_AR')
            if "language" in filters:
                lang_code = md.get("language")
                if lang_code is None or not lang_code.lower().startswith(filters["language"].lower()):
                    mask[i] = False

        logger.debug("Filter mask applied: %d/%d items pass", int(mask.sum()), len(mask))
        return mask


    def semantic_search(self, query: str, mask: Optional[np.ndarray], top_k: int):
        logger.info("Semantic search: '%s' (top_k=%d)", query, top_k)
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search skipped: FAISS index is empty")
            return []
        q_emb = self.model.encode(query, convert_to_numpy=True)
        faiss.normalize_L2(q_emb.reshape(1, -1))
        D, I = self.index.search(q_emb.reshape(1, -1), min(100, self.index.ntotal))
        results = []
        for score, idx in zip(D[0], I[0]):
            if mask is not None and not mask[idx]:
                continue
            md = self.metadata[idx]
            results.append({"score": float(score), "metadata": md})
            if len(results) >= top_k:
                break
        logger.info("Search produced %d results (requested top_k=%d)", len(results), top_k)
        return results


# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="Course AI Search Chatbot")

retriever = Retriever()

@app.post("/chat", response_model=ChatResponse)
async def chat(req: QueryRequest):
    logger.info("/chat called with question length=%d", len(req.question or ""))
    filters = extract_filters(req.question)
    mask = retriever.build_filter_mask(filters)
    if mask is not None and not mask.any():
        logger.info("No items match the filters: %s", filters)
        return ChatResponse(results=[], filters_used=filters)

    results = retriever.semantic_search(req.question, mask, TOP_K)
    hits: List[CourseHit] = []
    for r in results:
        md = r["metadata"]
        hits.append(CourseHit(
            title=md.get("title"),
            short_description=md.get("short_description"),
            duration_minutes=md.get("duration_minutes"),
            language=md.get("language"),
            instructors=md.get("instructors") or [],
            what_you_will_learn=md.get("what_you_will_learn") or [],
            score=r["score"]
        ))
    logger.info("/chat returning %d hits", len(hits))
    return ChatResponse(results=hits, filters_used=filters)


@app.get("/health")
async def health():
    size = retriever.index.ntotal if retriever.index else 0
    logger.debug("/health check: index_size=%d", size)
    return {"status": "ok", "index_size": size}