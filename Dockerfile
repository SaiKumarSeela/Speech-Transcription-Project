FROM python:3.10.15-slim-bullseye

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y git ffmpeg && apt-get clean

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
