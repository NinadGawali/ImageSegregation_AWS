# app_s3_sqlite.py
import os
import io
import time
import uuid
import sqlite3
from datetime import datetime
from typing import List, Optional
# from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights

import boto3
from botocore.exceptions import ClientError

from dotenv import load_dotenv
load_dotenv()


# ---------------------
# Config (env vars)
# ---------------------
S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DB_PATH = os.getenv("DB_PATH", "images.db")
# Optional: if you want to restrict to a base prefix in bucket:
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "")  # e.g., "project_images/"

if not S3_BUCKET:
    raise RuntimeError("Please set environment variable S3_BUCKET to your target bucket name.")

# ---------------------
# Initialize AWS S3 client (uses standard AWS creds)
# ---------------------
_s3_client = None
def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client

def save_bytes_to_s3(bucket: str, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    """
    Upload bytes to S3 and return the s3://bucket/key URI.
    Key should include any folder prefix, e.g., "cats/123.jpg" or "myprefix/cats/123.jpg".
    """
    client = get_s3_client()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    except ClientError as e:
        # bubble up a clear HTTPException to the API
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")
    return f"s3://{bucket}/{key}"

def list_objects_with_prefix(bucket: str, prefix: str, max_items: int = 1000) -> List[dict]:
    """
    List S3 objects under the given prefix. Returns list of dicts with 'Key' and 'LastModified', 'Size'.
    """
    client = get_s3_client()
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(**kwargs):
        contents = page.get("Contents", [])
        for obj in contents:
            objects.append({"Key": obj["Key"], "LastModified": obj["LastModified"].isoformat(), "Size": obj["Size"]})
            if len(objects) >= max_items:
                return objects
    return objects

# ---------------------
# SQLite helpers
# ---------------------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bucket_uri TEXT NOT NULL,
        s3_key TEXT NOT NULL,
        folder TEXT NOT NULL,
        label TEXT NOT NULL,
        probability REAL NOT NULL,
        original_filename TEXT,
        timestamp TEXT NOT NULL
    )
    """)
    # index for faster lookups by folder/label
    cur.execute("CREATE INDEX IF NOT EXISTS idx_folder ON images(folder)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_label ON images(label)")
    conn.commit()
    conn.close()

def insert_image_record(bucket_uri: str, s3_key: str, folder: str, label: str,
                        probability: float, original_filename: Optional[str]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO images (bucket_uri, s3_key, folder, label, probability, original_filename, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (bucket_uri, s3_key, folder, label, probability, original_filename, ts))
    inserted_id = cur.lastrowid
    conn.commit()
    conn.close()
    return inserted_id

def get_metadata_by_s3_key(s3_key: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM images WHERE s3_key = ?", (s3_key,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def search_metadata_by_folder_or_label(folder: Optional[str] = None, label: Optional[str] = None, limit: int = 100):
    """
    Simple search over the SQLite metadata. Returns matching rows.
    """
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

# Initialize DB on import
init_db(DB_PATH)

# ---------------------
# Model (ResNet50) - same as before
# ---------------------
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()
categories = weights.meta.get("categories", None)

def predict_label_from_pil(pil_image):
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

# ---------------------
# FastAPI app & endpoints
# ---------------------
app = FastAPI(title="Image Classifier (S3 + SQLite)")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get("/")
def root():
    return {"status": "ok", "msg": "Image classifier service with S3 upload and SQLite metadata"}

@app.post("/classify")
async def classify(image: UploadFile = File(...), save_to_s3: bool = Query(True)):
    """
    Accepts an image file, classifies it, saves to S3 (if save_to_s3=True), and records metadata in SQLite.
    Returns: label, probability, bucket_uri, s3_key, record_id
    """
    contents = await image.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    label, prob = predict_label_from_pil(pil)
    folder = map_label_to_folder(label)
    # unique filename
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    # prefix with base prefix if provided
    prefix = f"{S3_BASE_PREFIX.rstrip('/')}/" if S3_BASE_PREFIX else ""
    s3_key = f"{prefix}{folder}/{filename}"

    result = {"label": label, "probability": prob, "folder": folder, "filename": filename}

    if save_to_s3:
        # upload bytes to s3
        bucket_uri = save_bytes_to_s3(S3_BUCKET, s3_key, contents, content_type=image.content_type or "image/jpeg")
        result["bucket_uri"] = bucket_uri
        result["s3_key"] = s3_key
        # persist metadata to SQLite
        rec_id = insert_image_record(bucket_uri=bucket_uri, s3_key=s3_key, folder=folder,
                                     label=label, probability=prob, original_filename=image.filename)
        result["db_id"] = rec_id
    else:
        result["saved"] = False

    return JSONResponse(result)

@app.get("/search")
def search(folder: Optional[str] = Query(None), label: Optional[str] = Query(None), max_items: int = Query(100)):
    """
    Search images: listing is done by reading S3 objects under the folder prefix if folder is provided.
    We also include available metadata from SQLite when it exists.
    Priority logic:
      - If folder provided: list S3 objects under prefix (S3_BASE_PREFIX + folder/) and return their Keys + metadata.
      - Else if label provided: query SQLite for rows with that label.
      - Else: return recent rows from SQLite (limits apply).
    """
    results = []

    if folder:
        # list objects from the bucket for this folder
        prefix = f"{S3_BASE_PREFIX.rstrip('/')}/" if S3_BASE_PREFIX else ""
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

    # otherwise fallback to sqlite metadata queries
    rows = search_metadata_by_folder_or_label(folder=folder, label=label, limit=max_items)
    return {"source": "sqlite", "count": len(rows), "items": rows}

# Convenience endpoint to fetch a DB row by id
@app.get("/metadata/{record_id}")
def get_metadata(record_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM images WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Record not found")
    return dict(row)

# optional: generate presigned GET URL for a given s3_key so frontend can fetch directly
@app.get("/signed-url")
def signed_url(s3_key: str, expires_in: int = 3600):
    client = get_s3_client()
    try:
        url = client.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": s3_key}, ExpiresIn=expires_in)
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"url": url, "s3_key": s3_key}

# ---------------------
# Run
# ---------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
