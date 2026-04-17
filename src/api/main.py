from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.extraction.Text_extractor import extract_text
from src.main_pipeline import process_contract

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root_ui() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Contract Intelligence</title>
      <style>
        :root {
          color-scheme: normal;
          font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f6f8fb;
          color: #111827;
        }
        * {
          box-sizing: border-box;
        }
        body {
          margin: 0;
          min-height: 100vh;
          display: flex;
          justify-content: center;
          padding: 24px;
        }
        .page {
          width: 100%;
          max-width: 1080px;
        }
        .hero {
          display: grid;
          gap: 24px;
          padding: 32px 32px 24px;
          background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
          border-radius: 28px;
          color: #ffffff;
          margin-bottom: 28px;
        }
        .hero h1 {
          margin: 0;
          font-size: clamp(2.5rem, 3vw, 4rem);
          line-height: 1.05;
        }
        .hero p {
          margin: 0;
          max-width: 720px;
          font-size: 1.05rem;
          opacity: .9;
        }
        .card {
          background: #ffffff;
          border-radius: 24px;
          box-shadow: 0 20px 80px rgba(15, 23, 42, 0.08);
          padding: 28px;
          margin-bottom: 24px;
        }
        .upload-section {
          display: grid;
          gap: 18px;
        }
        .upload-label {
          font-size: .95rem;
          font-weight: 600;
          color: #374151;
        }
        .upload-input {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }
        input[type='file'] {
          width: 100%;
          padding: 14px 16px;
          border-radius: 16px;
          border: 1px solid #d1d5db;
          background: #f9fafb;
          cursor: pointer;
        }
        button {
          border: none;
          border-radius: 14px;
          padding: 14px 22px;
          background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
          color: #ffffff;
          font-weight: 700;
          cursor: pointer;
          transition: transform .2s ease, box-shadow .2s ease, opacity .2s ease;
        }
        button:hover {
          transform: translateY(-1px);
          box-shadow: 0 18px 36px rgba(37, 99, 235, 0.2);
        }
        button:disabled {
          opacity: 0.65;
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
        }
        .status {
          min-height: 22px;
          font-size: .98rem;
          color: #2563eb;
          font-weight: 600;
        }
        .result-card {
          display: grid;
          gap: 18px;
        }
        .result-card h2 {
          margin: 0;
          font-size: 1.2rem;
          color: #111827;
        }
        .result-panel {
          background: #111827;
          color: #f9fafb;
          border-radius: 20px;
          padding: 22px;
          overflow-x: auto;
          font-family: 'Source Code Pro', monospace;
          font-size: .95rem;
          line-height: 1.6;
          min-height: 220px;
          white-space: pre-wrap;
        }
        .footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          font-size: .92rem;
          color: #6b7280;
          padding-top: 8px;
        }
        @media (max-width: 720px) {
          body { padding: 16px; }
          .hero { padding: 24px; }
          .card { padding: 20px; }
        }
      </style>
    </head>
    <body>
      <main class="page">
        <section class="hero">
          <div>
            <p style="margin: 0; font-size: .95rem; letter-spacing: .08em; text-transform: uppercase; color: #93c5fd;">Contract Intelligence</p>
            <h1>Professional contract review in one upload.</h1>
            <p>Analyze PDF, DOCX, or TXT contracts and surface important or risky clauses instantly. This tool connects your contract text to the backend pipeline for fast clause extraction and risk summarization.</p>
          </div>
        </section>

        <section class="card upload-section">
          <div>
            <p class="upload-label">Upload contract file</p>
            <form id="upload-form">
              <div class="upload-input">
                <input type="file" id="file-input" name="file" accept=".pdf,.docx,.txt" required />
                <button type="button" id="submit-button">Analyze Document</button>
              </div>
            </form>
          </div>
          <div class="status" id="status">Select a contract file and click Analyze.</div>
        </section>

        <section class="card result-card">
          <h2>Analysis output</h2>
          <div id="result" class="result-panel">No analysis yet. Upload a document to begin.</div>
        </section>

        <footer class="footer">
          <div>Supports PDF, DOCX, and TXT uploads</div>
          <div>Backend API: <code>/analyze</code></div>
        </footer>
      </main>

      <script>
        const fileInput = document.getElementById('file-input');
        const submitButton = document.getElementById('submit-button');
        const status = document.getElementById('status');
        const result = document.getElementById('result');

        submitButton.addEventListener('click', async () => {
          if (!fileInput.files.length) {
            status.textContent = 'Select a contract file before analyzing.';
            return;
          }

          setLoading(true, 'Uploading and analyzing contract. This may take a moment...');
          result.textContent = '';

          const formData = new FormData();
          formData.append('file', fileInput.files[0]);

          try {
            const response = await fetch('/analyze', {
              method: 'POST',
              body: formData,
            });

            if (!response.ok) {
              const errorBody = await response.json().catch(() => ({}));
              throw new Error(errorBody.detail || 'Analysis failed.');
            }

            const payload = await response.json();
            status.textContent = 'Analysis complete.';
            renderResult(payload);
          } catch (error) {
            status.textContent = 'Analysis failed.';
            result.textContent = 'Error: ' + error.message;
          } finally {
            setLoading(false);
          }
        });

        function setLoading(isLoading, message) {
          submitButton.disabled = isLoading;
          fileInput.disabled = isLoading;
          status.textContent = message || (isLoading ? 'Working...' : 'Ready to analyze your contract.');
        }

        function renderResult(payload) {
          if (!payload || !payload.clauses) {
            result.textContent = 'No clauses returned from analysis.';
            return;
          }

          if (!payload.clauses.length) {
            result.textContent = 'The contract was processed successfully, but no important clauses were identified.';
            return;
          }

          const lines = payload.clauses.map((clause, index) => {
            const label = clause.label || 'uncertain';
            const confidence = clause.confidence?.toFixed(2) ?? '0.00';
            const text = clause.text?.trim() ?? '';
            const topK = Array.isArray(clause.top_k)
              ? clause.top_k.map(item => `    • ${item.label}: ${item.score?.toFixed(2) ?? '0.00'}`).join('\n')
              : '';

            return [`Clause ${index + 1}:`, `Label: ${label} · Confidence: ${confidence}`, `Text: ${text}`, topK ? 'Top predictions:\n' + topK : ''].filter(Boolean).join('\n');
          });

          result.textContent = lines.join('\n\n---\n\n');
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


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
async def analyze_contract(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_path = ""
    try:
        temp_path = await save_upload_to_temp(file)
        extension = os.path.splitext(temp_path)[1].lower()

        if extension == ".txt":
            with open(temp_path, "r", encoding="utf-8", errors="replace") as text_file:
                text = text_file.read()
        else:
            text = extract_text(temp_path)

        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the uploaded document.",
            )

        results = process_contract(text)
        return {"clauses": results}

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
