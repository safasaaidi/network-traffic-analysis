# Image Python officielle
FROM python:3.11-slim

# Dossier de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les librairies
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Backend matplotlib sans interface graphique
ENV MPLBACKEND=Agg

# Commande à exécuter
CMD ["python", "scripts/network_analysis.py"]
