# 🧠 Intelligent Image Classification & Segregation System

This project is an **AI-powered image classification and segregation system** built with **FastAPI**, **PyTorch**, and **AWS S3** integration.  

It allows users to:
- Upload images
- Classify them using a deep learning model
- Store results in **SQLite** and **Amazon S3**

The system is designed to demonstrate how machine learning, backend APIs, and cloud storage can integrate seamlessly for real-world applications.

## 🚀 Project Overview

This project provides a **RESTful API** that performs the following functions:

- Accepts image uploads via API or web interface.  
- Preprocesses and classifies images using a **ResNet-50** model from PyTorch.  
- Uploads images to an **AWS S3 bucket** for persistent storage.  
- Saves classification results and metadata (image name, label, confidence, etc.) in **SQLite**.  
- Exposes endpoints to fetch upload history and classification results.

An optional **Streamlit frontend** is included for easy interaction and visualization.

## 🧩 Model Details

- **Model:** ResNet-50 (Pre-trained on ImageNet)
- **Framework:** PyTorch  
- **Backend:** FastAPI (for inference and data flow)  
- **Device Support:** CUDA (GPU) if available, otherwise CPU  
- **Purpose:** Multi-class image classification  
- **Output:** Predicted label and confidence score

## ☁️ AWS S3 Integration

AWS S3 is used for secure and scalable image storage.

Each uploaded image:
1. Receives a unique **UUID-based filename**.
2. Gets uploaded to the configured **S3 bucket**.
3. Has its S3 URL stored in the database alongside classification data.

**Example S3 Path:**
```bash
s3://your-bucket-name/uploads/{uuid_filename}.jpg
```


**Required Environment Variables:**

| Variable | Description |
|-----------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `AWS_REGION` | AWS region (e.g., ap-south-1) |
| `S3_BUCKET_NAME` | Target S3 bucket name |

## 🧰 Installation Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/ImageSegregation_AWS.git
cd ImageSegregation_AWS
```
### 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a **.env** file in your project root:

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your_s3_bucket_name
```

## 🧪 Running the Application

### 🧱 Start the FastAPI Backend
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Once the server starts, open:
👉 http://127.0.0.1:8000/docs
to test all endpoints using the built-in Swagger UI.

### 💻 (Optional) Run the Streamlit Front-End
```bash
cd frontend
streamlit run app.py
```
Then visit 
👉 http://localhost:8501 
to upload, classify and search images via a simple UI.

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend** | FastAPI |
| **Model** | PyTorch (ResNet-50) |
| **Storage** | AWS S3 |
| **Database** | SQLite |
| **Frontend** | Streamlit |
| **Language** | Python 3.9+ |


## 🎥 Demo Video

📺 Watch the full demonstration here:  


https://github.com/user-attachments/assets/d06eb607-2bb0-4b02-9861-c7d3ba1ff7c3


## 🧑‍💻 Author

**Ninad Gawali**  
🔗 [GitHub Profile](https://github.com/NinadGawali)



