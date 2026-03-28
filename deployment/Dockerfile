FROM python:3.11-slim

# Install poppler for pdf2image (vision rendering)
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port Gradio runs on
EXPOSE 7860

# Run ingestion first if chroma_db doesn't exist, then launch app
CMD ["sh", "-c", "[ -d chroma_db ] || python ingest.py && python app.py"]
