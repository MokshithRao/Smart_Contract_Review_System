from fastapi.responses import HTMLResponse


def get_root_ui_response() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Smart Contract Review</title>
      <style>
        body { margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7fb; color: #111827; }
        .page { display: flex; justify-content: center; padding: 2rem; }
        .card { width: min(960px, 100%); background: white; border-radius: 24px; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08); padding: 2rem; }
        .hero { display: grid; gap: 1rem; }
        .hero h1 { margin: 0; font-size: clamp(2rem, 3vw, 3rem); }
        .hero p { margin: 0; color: #475569; line-height: 1.7; }
        .upload-area { margin-top: 1.5rem; display: grid; gap: 1rem; }
        textarea { width: 100%; min-height: 160px; padding: 1rem; border: 1px solid #cbd5e1; border-radius: 16px; resize: vertical; background: #f8fafc; color: #0f172a; }
        .control-row { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }
        input[type=file] { flex: 1 1 0; min-width: 0; padding: 1rem 1rem; border: 1px solid #cbd5e1; border-radius: 12px; background: #f8fafc; cursor: pointer; }
        .control-row button { white-space: nowrap; }
        button { padding: 0.95rem 1.5rem; border: none; border-radius: 999px; background: #2563eb; color: white; font-weight: 600; cursor: pointer; transition: transform 0.2s ease, background 0.2s ease; }
        button:hover { transform: translateY(-1px); background: #1d4ed8; }
        button.clear-button { background: #dc2626; }
        button.clear-button:hover { background: #b91c1c; }
        button.secondary { background: #64748b; }
        button.secondary:hover { background: #475569; }
        .status { color: #334155; font-weight: 600; }
        .result { margin-top: 1.5rem; display: grid; gap: 1rem; }
        .card-result { padding: 1.25rem; background: #f8fafc; border-radius: 16px; border: 1px solid #e2e8f0; }
        .card-result h2 { margin: 0 0 0.5rem; font-size: 1.05rem; color: #0f172a; }
        .card-result pre { margin: 0; white-space: pre-wrap; word-break: break-word; color: #1e293b; }
        .secondary { color: #64748b; }
        .meta { display: grid; gap: 0.5rem; margin-top: 0.75rem; }
      </style>
    </head>
    <body>
      <div class="page">
        <div class="card">
          <div class="hero">
            <div>
              <h1>Smart Contract Review System</h1>
              <p>Upload a PDF, DOCX, or TXT contract to extract key clauses and risk signals from your agreement.</p>
            </div>
            <div class="upload-area">
              <form id="upload-form">
                <label for="text-input">Or paste contract text here:</label>
                <textarea id="text-input" name="text" rows="6" placeholder="Paste your contract text or key clauses here..."></textarea>
                <div class="control-row">
                  <input type="file" id="file-input" name="file" accept=".pdf,.docx,.txt" />
                  <button type="button" id="clear-button" class="clear-button">Clear file / text</button>
                </div>
                <button type="submit">Analyze Contract</button>
              </form>
              <div id="status" class="status"></div>
            </div>
          </div>
          <div id="result" class="result"></div>
        </div>
      </div>
      <script>
        const form = document.getElementById('upload-form');
        const status = document.getElementById('status');
        const result = document.getElementById('result');

        const clearButton = document.getElementById('clear-button');
        const textInput = document.getElementById('text-input');

        clearButton.addEventListener('click', () => {
          document.getElementById('file-input').value = '';
          textInput.value = '';
          status.textContent = 'File and text cleared.';
          result.innerHTML = '';
        });

        form.addEventListener('submit', async event => {
          event.preventDefault();
          status.textContent = 'Analyzing input...';
          result.innerHTML = '';

          const input = document.getElementById('file-input');
          const textValue = textInput.value.trim();
          const formData = new FormData();

          if (textValue) {
            formData.append('text', textValue);
          }

          if (input.files.length) {
            formData.append('file', input.files[0]);
          }

          if (!textValue && !input.files.length) {
            status.textContent = 'Please select a file or enter text first.';
            return;
          }

          try {
            const response = await fetch('/analyze', {
              method: 'POST',
              body: formData,
            });

            if (!response.ok) {
              const body = await response.json();
              throw new Error(body.detail || 'Analysis failed.');
            }

            const data = await response.json();
            status.textContent = 'Analysis complete.';
            const clauses = data.clauses || [];
            if (!clauses.length) {
              result.innerHTML = '<div class="card-result"><h2>No clauses were identified.</h2><p class="secondary">Try another contract or verify the uploaded file type.</p></div>';
              return;
            }

            clauses.forEach((clause, index) => {
              const section = document.createElement('div');
              section.className = 'card-result';
              section.innerHTML = `
                <h2>${index + 1}. ${clause.label || 'Unknown clause'}</h2>
                <div class="secondary meta">
                  <span><strong>Confidence:</strong> ${clause.confidence?.toFixed(2) ?? 'N/A'}</span>
                  ${clause.top_k ? `<span><strong>Top predictions:</strong> ${clause.top_k.map(pred => `${pred.label} (${pred.score.toFixed(2)})`).join(', ')}</span>` : ''}
                </div>
                <pre>${clause.text || ''}</pre>
              `;
              result.appendChild(section);
            });
          } catch (error) {
            status.textContent = 'Error: ' + error.message;
            result.innerHTML = '';
          }
        });
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
