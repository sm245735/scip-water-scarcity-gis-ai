FROM python:3.11.13-slim-bookworm
RUN pip install --no-cache-dir psycopg[binary]==3.2.10
WORKDIR /app
