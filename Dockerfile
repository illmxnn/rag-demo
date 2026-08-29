FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY evaluation ./evaluation
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[documents]"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
