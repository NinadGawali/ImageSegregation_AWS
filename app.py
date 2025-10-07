# streamlit_app.py
import streamlit as st
from PIL import Image
import io
import os
import time
import uuid

import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# -------------------------
# Config
# -------------------------
IMAGE_DIR = "images"  # base folder to save labeled images
os.makedirs(IMAGE_DIR, exist_ok=True)

# -------------------------
# Model utilities
# -------------------------
@st.cache_resource
def load_model_and_transforms():
    """
    Load a pretrained ResNet50 and its recommended preprocessing transforms.
    Uses torchvision's Weights API which includes category names.
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", None)
    return model, preprocess, categories

model, preprocess, categories = load_model_and_transforms()

def predict_label(pil_image):
    """
    Returns (raw_label_string, score_float).
    We'll use the top-1 ImageNet label.
    """
    # preprocess -> batch
    input_tensor = preprocess(pil_image).unsqueeze(0)  # 1xCxHxW
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = torch.topk(probs, k=1)
        idx = int(top_idx[0][0].item())
        prob = float(top_prob[0][0].item())
    label = categories[idx] if categories else f"class_{idx}"
    return label, prob

def map_label_to_folder(label: str) -> str:
    """
    Map ImageNet label to a simpler folder name.
    - If label contains 'cat' -> 'cats'
    - If label contains 'dog' -> 'dogs'
    - else use a normalized short label (lower, underscores)
    """
    l = label.lower()
    if "cat" in l:
        return "cats"
    if "dog" in l:
        return "dogs"
    # take first word and sanitize
    safe = l.replace(" ", "_").replace(",", "").replace("/", "_")
    # limit length
    safe = safe[:30]
    return safe

def save_uploaded_image(file_bytes: bytes, folder: str, original_filename: str = None):
    """
    Save bytes into images/<folder>/ with a unique name.
    Returns saved filepath.
    """
    folder_path = os.path.join(IMAGE_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)
    # create unique name preserving extension if possible
    ext = ".jpg"
    if original_filename and "." in original_filename:
        ext = os.path.splitext(original_filename)[1] or ext
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(folder_path, filename)
    with open(filepath, "wb") as f:
        f.write(file_bytes)
    return filepath

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Image Labeler (Step A)", layout="wide")

st.title("Image Labeler — Step A (upload → classify → save by folder)")

page = st.sidebar.radio("Select page", ["Upload", "Browse labels"])

if page == "Upload":
    st.header("Upload an image")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded is not None:
        # Read as PIL
        file_bytes = uploaded.read()
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        st.image(pil, caption="Uploaded image", use_column_width=True)

        # Button to run classification / save
        if st.button("Classify & Save"):
            with st.spinner("Running model..."):
                label, score = predict_label(pil)
                folder = map_label_to_folder(label)
                saved_path = save_uploaded_image(file_bytes, folder, uploaded.name)
            st.success(f"Predicted: **{label}** ({score*100:.1f}%) — saved to folder **{folder}**")
            st.write(f"Saved file: `{saved_path}`")
            # show small gallery of that folder
            st.subheader(f"Recent images in `{folder}`")
            col1, col2, col3 = st.columns(3)
            files = sorted(os.listdir(os.path.join(IMAGE_DIR, folder)), reverse=True)[:9]
            for i, fname in enumerate(files):
                fpath = os.path.join(IMAGE_DIR, folder, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    col = (col1, col2, col3)[i % 3]
                    col.image(img, caption=fname, use_column_width=True)
                except Exception as e:
                    st.write(f"Could not load {fpath}: {e}")

elif page == "Browse labels":
    st.header("Browse saved images by label / folder")
    # list folders in IMAGE_DIR
    folders = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]
    folders = sorted(folders)
    if not folders:
        st.info("No labeled images yet. Upload one on the 'Upload' page.")
    else:
        chosen = st.selectbox("Choose label/folder", folders)
        if chosen:
            folder_path = os.path.join(IMAGE_DIR, chosen)
            files = sorted(os.listdir(folder_path), reverse=True)
            st.write(f"Found {len(files)} images in `{chosen}`")
            if files:
                # show grid of thumbnails (3 columns)
                cols = st.columns(3)
                for i, fname in enumerate(files):
                    fpath = os.path.join(folder_path, fname)
                    try:
                        img = Image.open(fpath).convert("RGB")
                        with cols[i % 3]:
                            st.image(img, caption=fname, use_column_width=True)
                    except Exception as e:
                        st.write(f"Error loading {fname}: {e}")

# Footer / tips
st.markdown("---")
st.markdown(
    """
**Notes / tips**
- This uses a pretrained ResNet50 (ImageNet) shipped by torchvision. It gets the top-1 ImageNet label, then maps to a folder.
- Mapping is simple: labels containing "cat" -> `cats`, "dog" -> `dogs`, otherwise a sanitized label is used as folder name.
- For production you can plug in a multi-label classifier or a model trained on your own classes.
"""
)
