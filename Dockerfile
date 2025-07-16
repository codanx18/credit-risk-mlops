FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /app/src
COPY ./models /app/models
COPY ./data/processed /app/data/processed
COPY ./data/raw /app/data/raw

EXPOSE 8000     

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]