FROM python:3.12

WORKDIR /frontend

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

CMD ["streamlit", "run", "--server.address", "0.0.0.0", "app.py"]
