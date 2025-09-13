# Placeholder; full Dockerfile will be finalized in step 7.

FROM python:3.10-slim AS base

# Set working directory inside container
WORKDIR /app

# Recommended Python env flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies first for better layer caching
# - Copies only requirements.txt, installs, then copies the rest of the project
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files into the image
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
