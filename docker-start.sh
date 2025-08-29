#!/bin/bash

# 🐳 Script de démarrage Docker pour Trendr
# Script super simple pour les débutants en Docker

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 TRENDR DOCKER SETUP${NC}"
echo "=================================="

# 1. Vérification de Docker
echo -e "${YELLOW}📋 Vérification de Docker...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker n'est pas installé !${NC}"
    echo "Installer Docker Desktop depuis: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Vérifier Docker Compose (nouvelle version ou ancienne)
DOCKER_COMPOSE_CMD="docker compose"
if ! docker compose version &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    else
        echo -e "${RED}❌ Docker Compose n'est pas disponible !${NC}"
        echo "Docker Compose est généralement inclus avec Docker Desktop"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Docker est installé${NC}"
docker --version
$DOCKER_COMPOSE_CMD version

# 2. Vérification du fichier .env
echo -e "${YELLOW}📋 Vérification de la configuration...${NC}"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Fichier .env manquant, création à partir du template...${NC}"
    
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}📝 Fichier .env créé. VOUS DEVEZ LE CONFIGURER MAINTENANT !${NC}"
        echo ""
        echo -e "${RED}🛑 ÉTAPES OBLIGATOIRES :${NC}"
        echo "1. Ouvrir le fichier .env dans votre éditeur"
        echo "2. Remplacer TOUTES les valeurs 'your-xxx-key' par vos vraies clés API"
        echo "3. Sauvegarder le fichier"
        echo "4. Relancer ce script"
        echo ""
        echo -e "${BLUE}Fichier à éditer : $(pwd)/.env${NC}"
        exit 1
    else
        echo -e "${RED}❌ Fichier .env.example manquant !${NC}"
        exit 1
    fi
fi

# 3. Vérification des clés API dans .env
echo -e "${YELLOW}🔑 Vérification des clés API...${NC}"

# Source le fichier .env
set -a
source .env
set +a

# Vérification des variables critiques
missing_vars=()

if [[ -z "$SUPABASE_URL" || "$SUPABASE_URL" == "your-"* ]]; then
    missing_vars+=("SUPABASE_URL")
fi

if [[ -z "$SUPABASE_KEY" || "$SUPABASE_KEY" == "your-"* ]]; then
    missing_vars+=("SUPABASE_KEY")
fi

if [[ -z "$GOOGLE_PLACES_API_KEY" || "$GOOGLE_PLACES_API_KEY" == "your-"* ]]; then
    missing_vars+=("GOOGLE_PLACES_API_KEY")
fi

if [[ -z "$GOOGLE_CUSTOM_SEARCH_API_KEY" || "$GOOGLE_CUSTOM_SEARCH_API_KEY" == "your-"* ]]; then
    missing_vars+=("GOOGLE_CUSTOM_SEARCH_API_KEY")
fi

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${RED}❌ Variables manquantes ou non configurées :${NC}"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo -e "${YELLOW}📝 Éditez le fichier .env et configurez ces variables${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Configuration OK${NC}"

# 4. Création des répertoires
echo -e "${YELLOW}📁 Création des répertoires...${NC}"
mkdir -p logs data/cache data/reports data/backups
echo -e "${GREEN}✅ Répertoires créés${NC}"

# 5. Fonction d'aide
show_help() {
    echo ""
    echo -e "${BLUE}🚀 COMMANDES DISPONIBLES :${NC}"
    echo ""
    echo -e "${GREEN}# Démarrer le système complet${NC}"
    echo "  $DOCKER_COMPOSE_CMD up -d"
    echo ""
    echo -e "${GREEN}# Voir les logs en temps réel${NC}"
    echo "  $DOCKER_COMPOSE_CMD logs -f"
    echo ""
    echo -e "${GREEN}# Voir le statut des services${NC}"
    echo "  $DOCKER_COMPOSE_CMD ps"
    echo ""
    echo -e "${GREEN}# Arrêter le système${NC}"
    echo "  $DOCKER_COMPOSE_CMD down"
    echo ""
    echo -e "${GREEN}# Lancer le pipeline manuellement (Paris)${NC}"
    echo "  $DOCKER_COMPOSE_CMD run --rm trendr-pipeline python3 run_pipeline.py --city Paris --mode full"
    echo ""
    echo -e "${GREEN}# Accéder à l'API de monitoring${NC}"
    echo "  curl http://localhost:8080/health"
    echo "  curl http://localhost:8080/trending/Paris"
    echo ""
}

# 6. Menu interactif
echo ""
echo -e "${BLUE}🎯 QUE VOULEZ-VOUS FAIRE ?${NC}"
echo "1) Démarrer le système complet"
echo "2) Voir l'aide avec toutes les commandes"
echo "3) Construire les images Docker (première fois)"
echo "4) Arrêter le système"
echo "5) Voir les logs"
echo "6) Lancer le pipeline manuellement"
echo "7) Quitter"

read -p "Votre choix (1-7): " choice

case $choice in
    1)
        echo -e "${YELLOW}🚀 Démarrage du système...${NC}"
        $DOCKER_COMPOSE_CMD up -d
        echo ""
        echo -e "${GREEN}✅ Système démarré !${NC}"
        echo -e "${BLUE}🌐 API disponible sur: http://localhost:8080${NC}"
        echo -e "${BLUE}📋 Status: $DOCKER_COMPOSE_CMD ps${NC}"
        echo -e "${BLUE}📊 Logs: $DOCKER_COMPOSE_CMD logs -f${NC}"
        ;;
    2)
        show_help
        ;;
    3)
        echo -e "${YELLOW}🔨 Construction des images Docker...${NC}"
        $DOCKER_COMPOSE_CMD build --no-cache
        echo -e "${GREEN}✅ Images construites !${NC}"
        ;;
    4)
        echo -e "${YELLOW}🛑 Arrêt du système...${NC}"
        $DOCKER_COMPOSE_CMD down
        echo -e "${GREEN}✅ Système arrêté !${NC}"
        ;;
    5)
        echo -e "${YELLOW}📊 Affichage des logs...${NC}"
        $DOCKER_COMPOSE_CMD logs -f
        ;;
    6)
        echo -e "${YELLOW}▶️  Lancement du pipeline Paris...${NC}"
        $DOCKER_COMPOSE_CMD run --rm trendr-pipeline python3 run_pipeline.py --city Paris --mode full
        ;;
    7)
        echo -e "${GREEN}👋 Au revoir !${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Choix invalide${NC}"
        show_help
        ;;
esac

echo ""
echo -e "${BLUE}💡 TIP: Relancez ce script à tout moment avec: ./docker-start.sh${NC}"