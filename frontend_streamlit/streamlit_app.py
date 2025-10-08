# frontend_streamlit/streamlit_app.py
import streamlit as st
import requests
from io import BytesIO

st.set_page_config(page_title="Image Classifier", layout="centered")

API_BASE = "http://localhost:8080"  # Backend FastAPI URL
  # or set via Streamlit secrets

st.title("Image Segregator & Search AWS S3")

nav = st.sidebar.radio("Pages", ["Upload & Classify", "Search"])

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
                    except Exception as e:
                        st.write("Saved to S3:", data.get("bucket_uri"))
                st.json(data)

# replace the Search page block in streamlit_app.py with:

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
                        s3_key = it.get("s3_key") or it.get("s3_key")  # handle both S3 list items and sqlite rows
                        # for sqlite rows, s3_key may be in 's3_key' or for fallback it is the row dict itself
                        # try extracting s3_key:
                        if not s3_key and it.get("s3_key") is None and it.get("bucket_uri"):
                            # this is a sqlite row
                            s3_key = it.get("s3_key")
                        # signed url
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
                                # if no s3_key/signed url, try to show original_filename as text
                                st.write(it.get("metadata", {}).get("label") if isinstance(it.get("metadata"), dict) else it.get("label") or it.get("original_filename") or "Preview not available")
                            # show caption info defensively
                            meta = it.get("metadata") or {}
                            label_text = meta.get("label") if isinstance(meta, dict) else it.get("label")
                            st.caption(label_text or s3_key or "")
