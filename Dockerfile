# Dockerfile must start with FROM instruction
# WORKDIR sets working directory for any RUN,CMD,ENTRYPOINT,COPY,ADD instructions that follow it in the Dockerfile
# If code is in same directory as Dockerfile, copy all app files from server into the container using COPY
# Install all app's python dependencies in requirements.txt in the container using RUN pip install -r requirements.txt
# EXPOSE instruction informs Docker that container listens on specified network ports at runtime. Streamlit default port is 8501
# HEALTHCHECK tells Docker how to test a container to check if it still working
# ENTRYPOINT allows you to configure a container that will run as an executable
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]