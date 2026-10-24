# Python app image
FROM python:3.12-slim

# Don't write .pyc files; print logs immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

# Change this to however you normally start your project,
# e.g. ["python", "run.py"] or ["python", "-m", "mypackage"]
CMD ["python3", "main.py"]
