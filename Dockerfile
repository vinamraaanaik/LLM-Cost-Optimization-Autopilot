FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Run unprivileged. /data holds the SQLite database and live routing rules (a mounted volume).
RUN useradd --create-home --uid 1000 autopilot \
    && mkdir /data \
    && chown autopilot:autopilot /data

WORKDIR /service
COPY pyproject.toml README.md ./
COPY app ./app
COPY dashboard ./dashboard
COPY configs ./configs
COPY data ./data
RUN pip install -e . && chown -R autopilot:autopilot /service
USER autopilot

EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
