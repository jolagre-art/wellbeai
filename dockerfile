FROM python:3.11-slim

WORKDIR /app

COPY app/ /app
COPY requirements.txt /app/requirements.txt
COPY models/ /app/models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

RUN ls -R /app