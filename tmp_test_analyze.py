from fastapi.testclient import TestClient
from src.api.main import app
import tempfile
import os

client = TestClient(app)
with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
    tmp.write('This agreement shall be governed by the laws of California. Party may terminate for convenience.')
    tmp_name = tmp.name

with open(tmp_name, 'rb') as f:
    response = client.post('/analyze', files={'file': ('contract.txt', f, 'text/plain')})
    print('status', response.status_code)
    print(response.text)

os.unlink(tmp_name)
