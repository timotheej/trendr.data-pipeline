# Dockerfile pour Trendr Data Pipeline - Version Production
FROM python:3.11-slim

# Métadonnées de l'image
LABEL maintainer="trendr-team"
LABEL version="1.0"
LABEL description="Trendr Data Pipeline - Automatic POI Monitoring System"

# Variables d'environnement pour Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV TZ=Europe/Paris

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    cron \
    tzdata \
    gosu \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Création d'un utilisateur non-root pour la sécurité
RUN groupadd -r trendr && useradd -r -g trendr trendr

# Copie des dépendances Python d'abord (optimisation du cache Docker)
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie de tous les fichiers du projet
COPY . .

# Création des répertoires avec bonnes permissions
RUN mkdir -p logs photos cache \
    && chmod +x *.py *.sh \
    && chown -R trendr:trendr /app

# Script de health check
RUN echo '#!/usr/bin/env python3\n\
import sys\n\
import os\n\
sys.path.append("/app")\n\
try:\n\
    from monitoring_system import TrendrMonitoringSystem\n\
    monitor = TrendrMonitoringSystem()\n\
    report = monitor.generate_health_report()\n\
    if report["system_health"] in ["healthy", "warning"]:\n\
        print("✅ Service healthy")\n\
        sys.exit(0)\n\
    else:\n\
        print(f"❌ Service unhealthy: {report[\"system_health\"]}")\n\
        sys.exit(1)\n\
except Exception as e:\n\
    print(f"❌ Health check failed: {e}")\n\
    sys.exit(1)\n' > /app/healthcheck.py \
    && chmod +x /app/healthcheck.py

# Script d'entrée personnalisé
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Démarrage de Trendr Data Pipeline..."\n\
echo "📅 Date: $(date)"\n\
echo "🌍 Timezone: $(cat /etc/timezone)"\n\
\n\
# Vérification des variables d'\''environnement\n\
if [ -z "$SUPABASE_URL" ]; then\n\
    echo "❌ ERREUR: SUPABASE_URL manquant"\n\
    exit 1\n\
fi\n\
\n\
if [ -z "$SUPABASE_KEY" ]; then\n\
    echo "❌ ERREUR: SUPABASE_KEY manquant"\n\
    exit 1\n\
fi\n\
\n\
echo "✅ Variables d'\''environnement configurées"\n\
echo "🎯 Lancement du service: $1"\n\
\n\
# Basculer vers l'\''utilisateur trendr\n\
exec gosu trendr "$@"\n' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Pas de port exposé - système de monitoring interne

# Health check pour Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /app/healthcheck.py

# Point d'entrée et commande par défaut
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "monitoring_system.py", "--daemon"]
