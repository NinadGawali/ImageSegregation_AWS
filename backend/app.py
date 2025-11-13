# backend/app.py
import os
import io
import time
import uuid
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torchvision import models, transforms
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from torchvision.models import resnet50, ResNet50_Weights

import boto3    
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Optional numeric / metrics libs (used if available)
try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.metrics import confusion_matrix, accuracy_score
except Exception:
    confusion_matrix = None
    accuracy_score = None

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-service")

# --------- CONFIG (set in .env) ----------
S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DB_PATH = os.getenv("DB_PATH", "images.db")
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "").strip("/")
# Comma separated allowed origins, e.g. http://localhost:8501,http://localhost:8000
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
PRESIGN_SAMPLE_LIMIT = int(os.getenv("PRESIGN_SAMPLE_LIMIT", "20"))
# ----------------------------------------

if not S3_BUCKET:
    raise RuntimeError("Please set S3_BUCKET in environment or .env")

# ---------- AWS S3 ----------
_s3_client = None
def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client

def save_bytes_to_s3(bucket: str, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    client = get_s3_client()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    except ClientError as e:
        logger.exception("S3 upload failed")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")
    return f"s3://{bucket}/{key}"

def list_objects_with_prefix(bucket: str, prefix: str, max_items: int = 1000) -> List[dict]:
    client = get_s3_client()
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                objects.append({
                    "Key": obj["Key"],
                    "LastModified": obj["LastModified"].isoformat() if hasattr(obj["LastModified"], "isoformat") else str(obj["LastModified"]),
                    "Size": obj["Size"]
                })
                if len(objects) >= max_items:
                    return objects
    except ClientError as e:
        logger.exception("S3 list failed")
        raise HTTPException(status_code=500, detail=f"S3 list failed: {e}")
    return objects

# ---------- SQLite ----------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bucket_uri TEXT NOT NULL,
        s3_key TEXT NOT NULL UNIQUE,
        folder TEXT NOT NULL,
        label TEXT NOT NULL,
        probability REAL NOT NULL,
        original_filename TEXT,
        timestamp TEXT NOT NULL
        -- Optionally add true_label and predicted_label columns if you maintain ground truth:
        -- , true_label TEXT, predicted_label TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_folder ON images(folder)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_label ON images(label)")
    conn.commit()
    conn.close()

def insert_image_record(bucket_uri: str, s3_key: str, folder: str, label: str,
                        probability: float, original_filename: Optional[str]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    try:
        cur.execute("""
            INSERT INTO images (bucket_uri, s3_key, folder, label, probability, original_filename, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (bucket_uri, s3_key, folder, label, probability, original_filename, ts))
        inserted_id = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        cur.execute("SELECT id FROM images WHERE s3_key = ?", (s3_key,))
        r = cur.fetchone()
        inserted_id = r[0] if r else None
    finally:
        conn.close()
    return inserted_id

def get_metadata_by_s3_key(s3_key: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM images WHERE s3_key = ?", (s3_key,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def search_metadata_by_folder_or_label(folder: Optional[str] = None, label: Optional[str] = None, limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if folder and label:
        cur.execute("SELECT * FROM images WHERE folder = ? AND label = ? ORDER BY timestamp DESC LIMIT ?", (folder, label, limit))
    elif folder:
        cur.execute("SELECT * FROM images WHERE folder = ? ORDER BY timestamp DESC LIMIT ?", (folder, limit))
    elif label:
        cur.execute("SELECT * FROM images WHERE label = ? ORDER BY timestamp DESC LIMIT ?", (label, limit))
    else:
        cur.execute("SELECT * FROM images ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

init_db(DB_PATH)

# ---------- Model (ResNet50) ----------

# ==========================================================
# Setup
# ==========================================================
logger.info("Loading fine-tuned ResNet50 model...")

# Path to your fine-tuned model
model_path = r"C:\Users\Akshan\Desktop\CC final CP\ImageSegregation_AWS\models\resnet50_pizza_steak_sushi.pth"

# Define same transform used during training
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Initialize ResNet50 for 3 classes
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# Load your fine-tuned weights
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Define your class names
categories = ["pizza", "steak", "sushi"]

# Prediction helper function
def predict_label_from_pil_sync(pil_image):
    input_tensor = data_transform(pil_image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=1)
        idx = int(top_idx[0][0].item())
        prob = float(top_prob[0][0].item())
    label = categories[idx]
    return label, prob

# Folder mapping (optional)
def map_label_to_folder(label: str) -> str:
    return label.lower()

# Thread executor for async inference
EXECUTOR = ThreadPoolExecutor(max_workers=1)


# logger.info("Loading ResNet50 model (may download weights on first run)...")
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model.eval()
# preprocess = weights.transforms()
# categories = weights.meta.get("categories", None)

# def predict_label_from_pil_sync(pil_image):
#     input_tensor = preprocess(pil_image).unsqueeze(0)
#     with torch.inference_mode():
#         logits = model(input_tensor)
#         probs = torch.nn.functional.softmax(logits, dim=-1)
#         top_prob, top_idx = torch.topk(probs, k=1)
#         idx = int(top_idx[0][0].item())
#         prob = float(top_prob[0][0].item())
#     label = categories[idx] if categories else f"class_{idx}"
#     return label, prob

# def map_label_to_folder(label: str) -> str:
#     l = label.lower()
#     if "cat" in l:
#         return "cats"
#     if "dog" in l:
#         return "dogs"
#     safe = l.replace(" ", "_").replace(",", "").replace("/", "_")
#     return safe[:40]

# EXECUTOR = ThreadPoolExecutor(max_workers=1)

# ---------- FastAPI app ----------
app = FastAPI(title="Image Classifier (S3 + SQLite)")

origins = [o.strip() for o in ALLOW_ORIGINS.split(",")] if ALLOW_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "msg": "Image classifier service (S3 + SQLite)"}

@app.post("/classify")
async def classify(image: UploadFile = File(...), save_to_s3: bool = Query(True)):
    contents = await image.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    loop = asyncio.get_running_loop()
    label, prob = await loop.run_in_executor(EXECUTOR, predict_label_from_pil_sync, pil)

    folder = map_label_to_folder(label)
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    prefix = f"{S3_BASE_PREFIX}/" if S3_BASE_PREFIX else ""
    s3_key = f"{prefix}{folder}/{filename}"

    result = {"label": label, "probability": prob, "folder": folder, "filename": filename}

    if save_to_s3:
        bucket_uri = save_bytes_to_s3(S3_BUCKET, s3_key, contents, content_type=image.content_type or "image/jpeg")
        result["bucket_uri"] = bucket_uri
        result["s3_key"] = s3_key
        rec_id = insert_image_record(bucket_uri=bucket_uri, s3_key=s3_key, folder=folder,
                                     label=label, probability=prob, original_filename=image.filename)
        result["db_id"] = rec_id
    else:
        result["saved"] = False

    return JSONResponse(result)

@app.get("/search")
def search(folder: Optional[str] = Query(None), label: Optional[str] = Query(None), max_items: int = Query(100, ge=1, le=1000)):
    results = []

    if folder:
        prefix = f"{S3_BASE_PREFIX}/" if S3_BASE_PREFIX else ""
        prefix = f"{prefix}{folder}/"
        s3_objs = list_objects_with_prefix(S3_BUCKET, prefix, max_items)
        for o in s3_objs[:max_items]:
            key = o["Key"]
            meta = get_metadata_by_s3_key(key)
            results.append({
                "s3_key": key,
                "last_modified": o.get("LastModified"),
                "size": o.get("Size"),
                "metadata": meta
            })
        return {"source": "s3", "bucket": S3_BUCKET, "prefix": prefix, "count": len(results), "items": results}

    rows = search_metadata_by_folder_or_label(folder=folder, label=label, limit=max_items)
    return {"source": "sqlite", "count": len(rows), "items": rows}

@app.get("/signed-url")
def signed_url(s3_key: str, expires_in: int = 3600):
    client = get_s3_client()
    try:
        url = client.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": s3_key}, ExpiresIn=expires_in)
    except ClientError as e:
        logger.exception("Failed to generate presigned URL")
        raise HTTPException(status_code=500, detail=str(e))
    return {"url": url, "s3_key": s3_key}

# ----------------- New: /stats endpoint and helpers -----------------

def safe_sqlite_connect(path: str = DB_PATH) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise FileNotFoundError(f"SQLite DB not found at: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def aggregate_counts_from_sqlite(path: str = DB_PATH) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    try:
        conn = safe_sqlite_connect(path)
    except FileNotFoundError:
        return counts
    cur = conn.cursor()
    # check columns
    cur.execute("PRAGMA table_info(images)")
    cols = [r["name"] for r in cur.fetchall()]
    # prefer 'label' column
    if "label" in cols:
        try:
            cur.execute("SELECT label, COUNT(*) as cnt FROM images GROUP BY label")
            for r in cur.fetchall():
                lbl = r["label"] or "UNKNOWN"
                counts[str(lbl)] = int(r["cnt"])
        except Exception as e:
            logger.exception("Failed to aggregate counts from sqlite by label")
    else:
        # try parsing metadata column if present
        if "metadata" in cols:
            try:
                cur.execute("SELECT metadata FROM images")
                for r in cur.fetchall():
                    try:
                        md = json.loads(r["metadata"])
                        lbl = md.get("label")
                        if lbl:
                            counts[lbl] = counts.get(lbl, 0) + 1
                    except Exception:
                        continue
            except Exception:
                pass
    conn.close()
    return counts

def list_s3_objects_and_aggregate_by_folder(bucket: str = S3_BUCKET, prefix: str = S3_BASE_PREFIX, max_items: int = 10000) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    # reuse list_objects_with_prefix but aggregate top-level folder
    pref = f"{prefix}/" if prefix else ""
    try:
        objs = list_objects_with_prefix(bucket, pref, max_items)
    except HTTPException as e:
        raise
    for o in objs:
        key = o["Key"]
        parts = key.split("/", 1)
        if len(parts) > 1 and parts[0]:
            lbl = parts[0]
        else:
            lbl = "ROOT"
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts

def compute_confusion_and_accuracy_from_sqlite(path: str = DB_PATH) -> Tuple[Optional[List[List[int]]], Optional[List[str]], Optional[float]]:
    """
    Looks for 'true_label' and 'predicted_label' (or 'label') columns in images table.
    Returns confusion matrix (2D list), labels list, and accuracy float.
    """
    try:
        conn = safe_sqlite_connect(path)
    except FileNotFoundError:
        return None, None, None

    cur = conn.cursor()
    cur.execute("PRAGMA table_info(images)")
    cols = [r["name"] for r in cur.fetchall()]
    # check columns
    if "true_label" not in cols:
        conn.close()
        return None, None, None
    pred_col = "predicted_label" if "predicted_label" in cols else ("label" if "label" in cols else None)
    if not pred_col:
        conn.close()
        return None, None, None

    cur.execute(f"SELECT true_label as t, {pred_col} as p FROM images WHERE true_label IS NOT NULL AND {pred_col} IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    true = [r["t"] for r in rows]
    pred = [r["p"] for r in rows]
    if not true or not pred or len(true) != len(pred):
        return None, None, None

    labels = sorted(list(set(true) | set(pred)))
    if confusion_matrix and accuracy_score and np is not None:
        try:
            cm = confusion_matrix(true, pred, labels=labels)
            acc = float(accuracy_score(true, pred))
            return cm.tolist(), labels, acc
        except Exception:
            pass

    # pure-python fallback
    idx = {l: i for i, l in enumerate(labels)}
    cm = [[0 for _ in labels] for __ in labels]
    correct = 0
    for t, p in zip(true, pred):
        i = idx[t]
        j = idx[p]
        cm[i][j] += 1
        if t == p:
            correct += 1
    acc = correct / len(true) if len(true) > 0 else None
    return cm, labels, acc

def sample_presign_timings(bucket: str = S3_BUCKET, prefix: str = S3_BASE_PREFIX, sample_limit: int = PRESIGN_SAMPLE_LIMIT) -> Dict[str, Any]:
    client = get_s3_client()
    timings: List[Tuple[str, Optional[float]]] = []
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=(f"{prefix}/" if prefix else ""), MaxKeys=sample_limit)
        items = resp.get("Contents", [])[:sample_limit]
        for obj in items:
            key = obj["Key"]
            t0 = time.time()
            try:
                client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=60)
                t1 = time.time()
                timings.append((key, (t1 - t0) * 1000.0))
            except Exception:
                timings.append((key, None))
    except ClientError as e:
        logger.exception("Failed to list S3 for presign sampling")
        return {"sampled": 0, "avg_ms": None, "median_ms": None, "min_ms": None, "max_ms": None, "samples": []}

    numeric = [t for (_, t) in timings if isinstance(t, (int, float))]
    summary: Dict[str, Any] = {}
    if numeric:
        summary["sampled"] = len(timings)
        summary["avg_ms"] = sum(numeric) / len(numeric)
        summary["median_ms"] = sorted(numeric)[len(numeric) // 2]
        summary["min_ms"] = min(numeric)
        summary["max_ms"] = max(numeric)
    else:
        summary["sampled"] = len(timings)
        summary["avg_ms"] = summary["median_ms"] = summary["min_ms"] = summary["max_ms"] = None
    summary["samples"] = [{"key": k, "ms": t} for (k, t) in timings]
    return summary

@app.get("/stats")
def get_stats(use_source: Optional[str] = Query("both"), s3_prefix: Optional[str] = Query(None)):
    """
    Returns aggregated statistics for the dataset.
    Query params:
      - use_source: 's3', 'sqlite', or 'both' (default 'both'). Prefers sqlite when available.
      - s3_prefix: optional S3 prefix to limit listing (overrides S3_BASE_PREFIX)
    Response JSON contains:
      {
        "source": "sqlite"|"s3"|"none",
        "per_tag_counts": {...},
        "per_tag_percent": {...},
        "total_items": int,
        "confusion_matrix": [[...]] or null,
        "labels": [...],
        "accuracy": float or null,
        "presign_timings_summary": {...},
        "errors": [...]
      }
    """
    errors: List[str] = []
    per_tag_counts: Dict[str, int] = {}
    per_tag_percent: Dict[str, float] = {}
    total_items = 0
    source_used = "none"

    src_pref = (use_source or "both").lower()
    s3_pref = s3_prefix if s3_prefix is not None else S3_BASE_PREFIX

    # Try sqlite first if requested
    if src_pref in ("sqlite", "both"):
        try:
            sqlite_counts = aggregate_counts_from_sqlite(DB_PATH)
            if sqlite_counts:
                per_tag_counts = sqlite_counts
                source_used = "sqlite"
        except Exception as e:
            logger.exception("sqlite aggregation failed")
            errors.append(f"sqlite_error: {str(e)}")

    # If no sqlite results or s3 explicitly requested, try S3
    if (not per_tag_counts and src_pref in ("s3", "both")) or src_pref == "s3":
        try:
            s3_counts = list_s3_objects_and_aggregate_by_folder(S3_BUCKET, s3_pref, max_items=10000)
            if s3_counts:
                per_tag_counts = s3_counts
                source_used = "s3"
        except Exception as e:
            logger.exception("s3 aggregation failed")
            errors.append(f"s3_error: {str(e)}")

    total_items = sum(per_tag_counts.values()) if per_tag_counts else 0
    if total_items > 0:
        for k, v in per_tag_counts.items():
            per_tag_percent[k] = (v / total_items) * 100.0

    # confusion matrix & accuracy (from sqlite if possible)
    cm, labels, acc = None, None, None
    try:
        cm, labels, acc = compute_confusion_and_accuracy_from_sqlite(DB_PATH)
    except Exception as e:
        logger.exception("failed computing confusion/accuracy")
        errors.append(f"confusion_error: {str(e)}")

    # presign timings sample
    presign_summary = None
    try:
        presign_summary = sample_presign_timings(S3_BUCKET, prefix=s3_pref, sample_limit=PRESIGN_SAMPLE_LIMIT)
    except Exception as e:
        logger.exception("failed sampling presign timings")
        errors.append(f"presign_error: {str(e)}")

    resp = {
        "source": source_used,
        "per_tag_counts": per_tag_counts,
        "per_tag_percent": per_tag_percent,
        "total_items": total_items,
        "confusion_matrix": cm,
        "labels": labels,
        "accuracy": acc,
        "presign_timings_summary": presign_summary,
        "errors": errors or None,
    }
    return JSONResponse(resp)

# ---------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
