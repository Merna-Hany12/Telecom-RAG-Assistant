from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from rag_core import RagCore      
import requests
from fastapi.middleware.cors import CORSMiddleware


N8N_WEBHOOK_URL = "https://mernahany.app.n8n.cloud/webhook/ticket"

app = FastAPI(
    title="NileTel Arabic AI Assistant",
    description="RAG-based telecom support assistant with ticket automation",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG system once (very important for performance)
rag = RagCore()


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

# What the user sends
class QueryRequest(BaseModel):
    query: str


# What the API returns
class QueryResponse(BaseModel):
    answer: str
    needs_action: str
    sources: list
    displayed_source: str


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {"message": "NileTel AI Assistant API is running successfully!"}


# ============================================================
# MAIN ENDPOINT (/ask)
# ============================================================

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    This endpoint:
    1. Receives user query
    2. Runs RAG pipeline
    3. Checks if action is needed
    4. If YES → calls n8n webhook
    5. Returns response to UI (Streamlit)
    """

    print(f"\n[API] Received new query: {request.query}")

    # --------------------------------------------------------
    # 1. RUN RAG PIPELINE
    # --------------------------------------------------------
    response = rag.run_rag_pipeline(request.query)

    print(f"[API] Response ready | Needs Action: {response['needs_action']}")

    # --------------------------------------------------------
    # 2. IF ACTION NEEDED → CALL n8n
    # --------------------------------------------------------
    if response["needs_action"] == "YES":
        print("[API] Action detected → Triggering n8n workflow...")

        try:
            # Send POST request to n8n webhook
            res = requests.post(
                N8N_WEBHOOK_URL,   #  destination (n8n)
                json={             #  data sent to n8n
                    "query": request.query,
                    "answer": response["answer"],
                    "sources": response["sources"]
                },
                timeout=5  # prevent hanging if n8n is slow
            )

            print(f"[API] n8n status: {res.status_code}")
            print(f"[API] n8n response: {res.text}")

        except Exception as e:
            print(f"[API] n8n error: {str(e)}")

    else:
        print("[API] No action needed")

    # --------------------------------------------------------
    # 3. RETURN RESPONSE TO STREAMLIT
    # --------------------------------------------------------
    return response