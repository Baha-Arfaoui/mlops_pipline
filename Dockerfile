# Utiliser une image Python comme base
FROM python:3.12-slim

# Install Git
RUN apt-get update && apt-get install -y git
# Install system dependencies (if needed, e.g., for psutil or elasticsearch)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement le fichier requirements.txt
COPY requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier les autres fichiers du projet dans le conteneur
COPY . /app

# Exposer le port sur lequel Flask va tourner
EXPOSE 8000

# Commande pour lancer l’application Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000", "--debug"]

