FROM python:3.11-slim

WORKDIR /app

# System deps for pdfplumber PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY src/ ./src/
COPY pages/ ./pages/
COPY app.py main.py ./

# Pre-built ChromaDB - baked into the image so no ingestion step is needed
# at runtime. The DB is read-only; rebuild the image to update embeddings.
COPY chroma_db/ ./chroma_db/

# Cloud Run injects PORT=8080 by default; Streamlit must listen on it.
ENV CHROMA_DB_PATH=/app/chroma_db
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8080

CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0"]
