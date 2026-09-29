FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects PORT env variable
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
