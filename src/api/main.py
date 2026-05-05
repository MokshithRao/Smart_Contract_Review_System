from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

load_dotenv()

from src.api.ui.fastapi_ui import get_root_ui_response
from src.extraction.Text_extractor import extract_text
from src.main_pipeline import build_rag_review, classify_contract
from src.risk_filter import filter_important_clauses

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root_ui():
    return get_root_ui_response()


async def save_upload_to_temp(file: UploadFile) -> str:
    filename = file.filename or "uploaded_contract"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only PDF, DOCX and TXT are accepted.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file is too large. Maximum allowed size is 20 MB.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(contents)
        return temp_file.name


@app.post("/analyze")
async def analyze_contract(
    file: UploadFile | None = File(None), text: str = Form("")
) -> Dict[str, Any]:
    temp_path = ""
    try:
        if text and text.strip():
            text_to_analyze = text.strip()
        elif file is not None:
            temp_path = await save_upload_to_temp(file)
            extension = os.path.splitext(temp_path)[1].lower()

            if extension == ".txt":
                with open(temp_path, "r", encoding="utf-8", errors="replace") as text_file:
                    text_to_analyze = text_file.read()
            else:
                text_to_analyze = extract_text(temp_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Please provide either a contract file or text to analyze.",
            )

        if not text_to_analyze or not text_to_analyze.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the uploaded document or entered text.",
            )

        all_clauses = classify_contract(text_to_analyze)
        clauses = filter_important_clauses(all_clauses)[:20]

        rag_review = []
        rag_status = "skipped"
        if os.getenv("HF_API_TOKEN"):
            try:
                rag_review = build_rag_review(text_to_analyze)
                rag_status = "completed"
            except Exception as error:
                logger.warning("RAG review unavailable for this request: %s", error)
                rag_status = "failed"
        else:
            logger.info("Skipping RAG review - HF_API_TOKEN not set")
            rag_status = "missing_api_token"

        return {
            "clauses": clauses,
            "all_clauses": all_clauses,
            "rag_review": rag_review,
            "rag_status": rag_status,
        }

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Contract analysis failed.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while analyzing contract.",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Unable to remove temporary file: %s", temp_path)
