FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Reentrena el modelo dentro de la imagen para que use las mismas
# versiones de librerías que la app (requiere data/datos.csv en el repo).
RUN python train.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
