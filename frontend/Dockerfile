FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

#CMD ["streamlit", "run", "app.py", "--server.port=8991", "--server.address=0.0.0.0"]