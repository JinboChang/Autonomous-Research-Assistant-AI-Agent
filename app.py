"""
FastAPI wrapper for the Autonomous Research Assistant.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from dotenv import load_dotenv

# Load environment variables from .env if present.
load_dotenv()

from autonomous_research_assistant.main import run as run_agent
try:
    from lc_agent import run_langchain
except Exception:
    run_langchain = None


# Load environment variables from .env if present.
load_dotenv()

app = FastAPI(title="Autonomous Research Assistant")

# Allow all origins for dev simplicity; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


@app.post("/run")
def run(query: Query) -> dict:
    """
    Execute the agent on the provided question and return the report.
    """
    # Prefer LangChain wrapper if available; fallback to classic.
    if run_langchain:
        report = run_langchain(query.question)
    else:
        report = run_agent(query.question)
    return {"report": report, "format": "md"}


@app.get("/healthz")
def healthz() -> dict:
    """Basic health check."""
    return {"status": "ok"}


# Serve the frontend (web directory) for convenience.
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("web/index.html")
