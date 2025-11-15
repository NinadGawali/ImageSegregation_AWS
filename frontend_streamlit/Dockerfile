# base image
FROM python:3.11-slim

# set working dir inside container
WORKDIR /app

# copy python deps file
COPY requirements.txt /app/requirements.txt

# install system deps needed by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# install python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app code
COPY . /app

# Environment for Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_PORT=8501
# Optional: set default API_BASE; override at deploy time with env var
ENV API_BASE="http://3.236.147.176:8080"

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"]
