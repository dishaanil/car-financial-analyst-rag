# Car Financial RAG

A Retrieval-Augmented Generation (RAG) chatbot for automotive sector annual reports (BMW, Ford, Tesla).

## Architecture

![Architecture Diagram](architecture.png)

## Project Structure

After cloning, add a `Data/` folder with your PDF annual reports before running anything. The folder name must contain the company name and year so the app can tag each document correctly.

```
car-financial-rag/
├── Data/                        ← create this manually (not in repo)
│   ├── BMW_2021.pdf
│   ├── BMW_2022.pdf
│   ├── BMW_2023.pdf
│   ├── Ford_2021.pdf
│   ├── Ford_2022.pdf
│   ├── Ford_2023.pdf
│   ├── Tesla_2022.pdf
│   └── Tesla_2023.pdf
├── pages/
│   ├── home.py
│   └── chat.py
├── src/
│   ├── document_processor.py
│   ├── rag_chain.py
│   └── vector_store.py
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

> The `chroma_db/` folder is generated automatically when you run ingestion for the first time. You do not need to create it manually.

## Local Setup

```bash
cd car-financial-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create .env
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Ingest PDFs into ChromaDB (run once)
python main.py --ingest

# Run the app
.venv/bin/streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Docker Setup

```bash
# Set your key in the environment
export OPENAI_API_KEY=sk-...

# Build and run
docker compose up --build
```

The app will be available at `http://localhost:8501`.
