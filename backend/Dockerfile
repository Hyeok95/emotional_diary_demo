FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

WORKDIR /app

# 파이썬 설치
RUN apt-get update && apt-get install -y python3 python3-pip

# 필수 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8999"]