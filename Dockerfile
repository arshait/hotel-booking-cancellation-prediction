FROM python:3.11.15-slim-bookworm

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY hotel_ml.py predict.py ./
COPY models/model.joblib models/model.joblib

EXPOSE 9696

CMD ["python", "-m", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
