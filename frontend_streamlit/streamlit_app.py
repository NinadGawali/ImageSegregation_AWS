# frontend_streamlit/streamlit_app.py
import streamlit as st
import requests
import time
from io import BytesIO
from collections import Counter
import math
import json
import os

# plotting libs (matplotlib required for visuals)
try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
except Exception:
    plt = None

# sklearn optional for clearer metrics (not required)
try:
    from sklearn.metrics import confusion_matrix, accuracy_score
    SKLEARN_INSTALLED = True
except Exception:
    confusion_matrix = None
    accuracy_score = None
    SKLEARN_INSTALLED = False

st.set_page_config(page_title="Image Classifier", layout="centered")

API_BASE = os.getenv("API_BASE", "http://localhost:8080")  # Backend FastAPI URL

st.title("Image Segregator & Search AWS S3")

nav = st.sidebar.radio("Pages", ["Upload & Classify", "Search", "Statistics & Metrics"])

# ------------------- Upload & Classify (unchanged) -------------------
if nav == "Upload & Classify":
    st.header("Upload & Classify")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    save_to_s3 = st.checkbox("Save to S3 (recommended)", value=True)
    if uploaded:
        st.image(uploaded, caption="Selected image", use_column_width=True)
    if st.button("Classify & Upload"):
        if not uploaded:
            st.warning("Please upload an image first.")
        else:
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
            params = {"save_to_s3": str(save_to_s3).lower()}
            with st.spinner("Classifying..."):
                try:
                    resp = requests.post(f"{API_BASE}/classify", files=files, params=params, timeout=120)
                    data = resp.json()
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()
            if not resp.ok:
                st.error(data.get("detail") or "Server error")
            else:
                st.success(f"Label: {data['label']} ({data['probability']*100:.2f}%)")
                st.write("Folder:", data.get("folder"))
                if data.get("s3_key"):
                    # get signed url
                    try:
                        r2 = requests.get(f"{API_BASE}/signed-url", params={"s3_key": data["s3_key"]}, timeout=30)
                        signed = r2.json()
                        if r2.ok:
                            st.image(signed["url"], caption="Saved image")
                        else:
                            st.write("Saved to S3:", data.get("bucket_uri"))
                    except Exception:
                        st.write("Saved to S3:", data.get("bucket_uri"))
                st.json(data)

# ------------------- Search (unchanged) -------------------
elif nav == "Search":
    st.header("Search Images")
    q = st.text_input("Folder name (e.g., cats, sports_car)")
    max_items = st.slider("Max items", min_value=10, max_value=200, value=60)
    if st.button("Search"):
        if not q:
            st.warning("Enter folder name to search (e.g., cats).")
        else:
            with st.spinner("Searching..."):
                try:
                    resp = requests.get(f"{API_BASE}/search", params={"folder": q, "max_items": max_items}, timeout=60)
                    j = resp.json()
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.stop()
            if not resp.ok:
                st.error(j.get("detail") or "Search error")
            else:
                st.write("Search source:", j.get("source"))
                if j.get("prefix_tried"):
                    st.write("Prefix tried:", j.get("prefix_tried"))
                items = j.get("items", [])
                if not items:
                    st.info("No results")
                else:
                    # display images in rows of 3
                    ncols = 3
                    cols = st.columns(ncols)
                    for idx, it in enumerate(items):
                        col = cols[idx % ncols]
                        s3_key = it.get("s3_key") or it.get("key") or it.get("object_key")
                        # try extracting metadata
                        metadata = {}
                        if isinstance(it.get("metadata"), dict):
                            metadata = it.get("metadata")
                        else:
                            # attempt parse if metadata is JSON-text
                            md_raw = it.get("metadata")
                            if md_raw:
                                try:
                                    metadata = json.loads(md_raw)
                                except Exception:
                                    metadata = {}
                        signed_url = None
                        if s3_key:
                            try:
                                r2 = requests.get(f"{API_BASE}/signed-url", params={"s3_key": s3_key}, timeout=30)
                                if r2.ok:
                                    signed_url = r2.json().get("url")
                            except Exception:
                                signed_url = None

                        with col:
                            if signed_url:
                                st.image(signed_url, use_column_width=True)
                            else:
                                st.write(metadata.get("label") if isinstance(metadata, dict) else it.get("label") or it.get("original_filename") or "Preview not available")
                            meta = metadata or {}
                            label_text = meta.get("label") if isinstance(meta, dict) else it.get("label")
                            st.caption(label_text or s3_key or "")

# ------------------- Statistics & Metrics (NEW: visualizations) -------------------
elif nav == "Statistics & Metrics":
    st.header("Statistics & Metrics")

    st.markdown("This page fetches aggregated statistics from the backend `/stats` endpoint and renders interactive visualizations.")

    with st.spinner("Fetching stats from backend..."):
        try:
            r = requests.get(f"{API_BASE}/stats", timeout=30)
            backend_stats = r.json() if r.ok else None
        except Exception as e:
            backend_stats = None
            st.error(f"Failed to fetch /stats: {e}")
            st.stop()

    if not backend_stats:
        st.info("No statistics available from backend. Make sure /stats is implemented and reachable.")
    else:
        # --- Summary cards ---
        col1, col2, col3 = st.columns(3)
        source = backend_stats.get("source", "unknown")
        total_items = backend_stats.get("total_items", 0)
        presign = backend_stats.get("presign_timings_summary") or {}
        sampled = presign.get("sampled", 0)

        col1.metric("Data Source", source)
        col2.metric("Total Items", total_items)
        col3.metric("Presign Sampled", sampled)

        st.markdown("---")

        # --- Per-tag counts bar chart + table ---
        per_tag_counts = backend_stats.get("per_tag_counts") or {}
        per_tag_percent = backend_stats.get("per_tag_percent") or {}

        st.subheader("Photos per Tag")
        if per_tag_counts:
            # create a DataFrame if pandas available
            if pd is not None:
                df = pd.DataFrame([
                    {"Tag": k, "Count": int(v), "Percent": float(per_tag_percent.get(k, 0.0))}
                    for k, v in per_tag_counts.items()
                ])
                df = df.sort_values("Count", ascending=False).reset_index(drop=True)
                st.table(df.head(20))
                # bar chart using matplotlib
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(8, max(3, len(df)*0.25)))
                    tags = df["Tag"].astype(str).values
                    counts = df["Count"].astype(int).values
                    y_pos = np.arange(len(tags))
                    ax.barh(y_pos, counts)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(tags)
                    ax.invert_yaxis()
                    ax.set_xlabel("Count")
                    ax.set_title("Images per Tag")
                    st.pyplot(fig)
                else:
                    st.write("matplotlib not available to plot bar chart.")
            else:
                # pandas not available: simple text table + ascii bar
                for k, v in sorted(per_tag_counts.items(), key=lambda x: -x[1])[:20]:
                    st.write(f"- {k}: {v} ({per_tag_percent.get(k, 0.0):.2f}%)")
        else:
            st.info("No per-tag counts returned by backend.")

        st.markdown("---")

        # --- Presign timings visualization ---
        st.subheader("Presign Generation Timings (server-side sample)")
        presign_samples = presign.get("samples", [])
        if presign_samples:
            # extract numeric times
            numeric_times = [s.get("ms") for s in presign_samples if isinstance(s.get("ms"), (int, float))]
            if numeric_times:
                avg = sum(numeric_times) / len(numeric_times)
                med = sorted(numeric_times)[len(numeric_times)//2]
                mn = min(numeric_times)
                mx = max(numeric_times)
                st.write(f"Sampled: {len(presign_samples)} (numeric: {len(numeric_times)}) — Avg: {avg:.1f} ms, Median: {med:.1f} ms, Min: {mn:.1f} ms, Max: {mx:.1f} ms")

                if plt is not None and np is not None:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(sorted(numeric_times), marker="o", linestyle="-")
                    ax.set_xlabel("Sample index (sorted)")
                    ax.set_ylabel("Presign generation time (ms)")
                    ax.set_title("Presign timings (server-side sample)")
                    st.pyplot(fig)
                else:
                    st.write("matplotlib/numpy required to plot timings.")
            else:
                st.info("Presign samples returned but no numeric timings available.")
            # show table of top slowest samples
            slowest = sorted(presign_samples, key=lambda s: (s.get("ms") is None, -(s.get("ms") or 0)))[:10]
            st.write("Top slowest samples (key, ms):")
            for s in slowest:
                st.write(f"- {s.get('key')}: {s.get('ms')}")
        else:
            st.info("No presign timing samples returned by backend.")

        st.markdown("---")

        # --- Confusion matrix & accuracy ---
        st.subheader("Confusion Matrix & Accuracy")
        cm = backend_stats.get("confusion_matrix")
        labels = backend_stats.get("labels") or []
        accuracy = backend_stats.get("accuracy")

        if accuracy is not None:
            st.write(f"Accuracy: {accuracy*100:.2f}%")
        else:
            st.write("Accuracy: N/A (backend did not provide)")

        if cm and labels:
            # convert to numpy array for plotting if available
            try:
                cm_arr = np.array(cm) if np is not None else cm
            except Exception:
                cm_arr = cm

            if plt is not None and np is not None:
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm_arr, interpolation="nearest")
                ax.set_title("Confusion Matrix (rows=true, cols=pred)")
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels)
                # annotate cells
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        val = cm_arr[i, j]
                        ax.text(j, i, int(val), ha="center", va="center")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
            else:
                # fallback: render as table
                if pd is not None:
                    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
                    st.table(df_cm)
                else:
                    st.write("Confusion matrix available but plotting libs not installed. Raw matrix:")
                    st.write(cm)
        else:
            st.info("No confusion matrix data available. To compute confusion matrix, backend must populate 'confusion_matrix' and 'labels' in /stats response, or include true_label/predicted_label in DB.")

        st.markdown("---")

        # --- Raw JSON (collapsible) ---
        with st.expander("Raw /stats JSON (for debugging)"):
            st.json(backend_stats)
