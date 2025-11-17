FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir flask xgboost pandas

COPY model.xgb .
COPY predict.py .

EXPOSE 8000

CMD ["python", "predict.py"]
