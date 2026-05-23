#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Build and deploy Car Financial RAG to GCP Cloud Run
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated  (gcloud auth login)
#   2. Docker installed and running locally
#   3. Fill in PROJECT_ID below
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="aecom-financial-rag"        # ← CHANGE THIS
REGION="us-central1"
SERVICE_NAME="car-financial-rag"
REPO_NAME="car-rag"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

echo "▶ Project : $PROJECT_ID"
echo "▶ Region  : $REGION"
echo "▶ Image   : $IMAGE"
echo ""

# ── Step 1: Enable required APIs ─────────────────────────────────────────────
echo "── Step 1/5: Enabling GCP APIs ──────────────────────────────────────────"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT_ID"

# ── Step 2: Create Artifact Registry repository (idempotent) ─────────────────
echo "── Step 2/5: Creating Artifact Registry repository ──────────────────────"
gcloud artifacts repositories create "$REPO_NAME" \
  --repository-format docker \
  --location "$REGION" \
  --project "$PROJECT_ID" 2>/dev/null \
  && echo "Repository created." \
  || echo "Repository already exists — continuing."

# Configure Docker to push to Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── Step 3: Store API keys in Secret Manager (run once) ──────────────────────
echo "── Step 3/5: Storing secrets in Secret Manager ──────────────────────────"
echo "   (Skipping if secrets already exist — update manually if needed)"

store_secret() {
  local name=$1
  local prompt=$2
  if gcloud secrets describe "$name" --project "$PROJECT_ID" &>/dev/null; then
    echo "   Secret '$name' already exists — skipping."
  else
    echo -n "   Enter value for $prompt: "
    read -rs value
    echo ""
    printf '%s' "$value" | gcloud secrets create "$name" \
      --data-file=- \
      --project "$PROJECT_ID"
    echo "   Secret '$name' created."
  fi
}

store_secret "OPENAI_API_KEY"    "OpenAI API key"
store_secret "LANGCHAIN_API_KEY" "LangChain/LangSmith API key"

# ── Step 4: Build image with Cloud Build ─────────────────────────────────────
echo "── Step 4/5: Building and pushing Docker image ──────────────────────────"
gcloud builds submit \
  --tag "$IMAGE" \
  --project "$PROJECT_ID" \
  --timeout 1200

# ── Step 5: Deploy to Cloud Run ───────────────────────────────────────────────
echo "── Step 5/5: Deploying to Cloud Run ────────────────────────────────────"
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --port 8080 \
  --set-secrets "OPENAI_API_KEY=OPENAI_API_KEY:latest,LANGCHAIN_API_KEY=LANGCHAIN_API_KEY:latest" \
  --set-env-vars "LANGCHAIN_TRACING_V2=true,LANGCHAIN_PROJECT=car-financial-rag,OPENAI_MODEL=gpt-4o-mini" \
  --project "$PROJECT_ID"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "✅ Deployment complete!"
URL=$(gcloud run services describe "$SERVICE_NAME" \
  --platform managed --region "$REGION" --project "$PROJECT_ID" \
  --format "value(status.url)")
echo "🌐 App URL: $URL"
