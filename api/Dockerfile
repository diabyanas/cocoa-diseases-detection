# image Python
FROM python:3.11.5

# Copie des fichiers de l'application dans le conteneur
COPY . /app

# Le répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances de l'application
RUN pip install -r requirements.txt

# Exposition de port sur lequel l'application FastAPI écoute
EXPOSE $PORT

# Commande pour exécuter l'application FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", $PORT]
