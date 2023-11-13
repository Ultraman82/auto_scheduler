FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

ARG srcDir=src
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./app ./app

# COPY $srcDir/run.py .

EXPOSE 8481
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8481"]
