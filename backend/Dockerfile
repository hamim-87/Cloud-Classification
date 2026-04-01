FROM python:3.12-slim

WORKDIR /app

# Install dependencies (torch CPU-only to keep image small)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/main.py .
COPY backend/inference.py .
COPY backend/rag/ ./rag/

# Copy model weights
COPY backend/model/ ./model/

RUN mkdir -p /app/data /app/papers /app/faiss_index

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
