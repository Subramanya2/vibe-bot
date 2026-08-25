FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY static ./static
COPY run_daily.py check_provider.py diagnose_groq.py ./

# Sample data ships empty; generate it at build so the image runs standalone.
RUN python data/generate.py --employees 500 --days 120

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
  "import httpx;httpx.get('http://localhost:8000/health',timeout=2).raise_for_status()"
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
