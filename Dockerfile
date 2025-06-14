# Dockerfile (exemple, assurez-vous que les chemins sont corrects)
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le modèle et les données depuis le dépôt GitHub
COPY P8_Cityscapes_leftImg8bit_trainvaltest/ P8_Cityscapes_leftImg8bit_trainvaltest/
COPY P8_Cityscapes_gtFine_trainvaltest/ P8_Cityscapes_gtFine_trainvaltest/

COPY main.py .

ENV PORT 8080
EXPOSE $PORT

# Utilise un shell pour s'assurer que la variable $PORT est correctement substituée
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
