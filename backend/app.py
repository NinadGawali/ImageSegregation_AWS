# backend/app.py
import os
import io
import time
import uuid
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

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
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            objects.append({
                "Key": obj["Key"],
                "LastModified": obj["LastModified"].isoformat() if hasattr(obj["LastModified"], "isoformat") else str(obj["LastModified"]),
                "Size": obj["Size"]
            })
            if len(objects) >= max_items:
                return objects
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
logger.info("Loading ResNet50 model (may download weights on first run)...")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()
categories = weights.meta.get("categories", None)

def predict_label_from_pil_sync(pil_image):
    input_tensor = preprocess(pil_image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=1)
        idx = int(top_idx[0][0].item())
        prob = float(top_prob[0][0].item())
    label = categories[idx] if categories else f"class_{idx}"
    return label, prob

def map_label_to_folder(label: str) -> str:
    l = label.lower()
    if "cat" in l:
        return "cats"
    if "dog" in l:
        return "dogs"
    safe = l.replace(" ", "_").replace(",", "").replace("/", "_")
    return safe[:40]

EXECUTOR = ThreadPoolExecutor(max_workers=1)

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
