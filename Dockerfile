FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers stdout - cleaner container logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first so Docker can cache this layer -
# dependencies only get reinstalled when requirements.txt actually changes,
# not on every code edit. This makes rebuilds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

EXPOSE 5000

# gunicorn is the production WSGI server - Flask's built-in dev server
# is single-threaded and not meant for real traffic
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
