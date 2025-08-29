#!/bin/bash

# Script de diagnostic Docker pour Mac

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 DIAGNOSTIC DOCKER${NC}"
echo "==================="

# 1. Vérifier si Docker est installé
echo -e "${YELLOW}📋 Vérification de l'installation Docker...${NC}"

if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker CLI installé${NC}"
    docker --version
else
    echo -e "${RED}❌ Docker CLI non trouvé${NC}"
    echo "Installer Docker Desktop depuis: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# 2. Vérifier si Docker Desktop est lancé
echo -e "${YELLOW}📋 Vérification du daemon Docker...${NC}"

if docker info &> /dev/null; then
    echo -e "${GREEN}✅ Docker daemon en cours d'exécution${NC}"
    echo -e "   Conteneurs: $(docker ps --format 'table {{.Names}}' | wc -l | xargs)"
    echo -e "   Images: $(docker images -q | wc -l | xargs)"
else
    echo -e "${RED}❌ Docker daemon non démarré${NC}"
    echo ""
    echo -e "${YELLOW}🚀 COMMENT DÉMARRER DOCKER :${NC}"
    echo ""
    echo "Sur Mac:"
    echo "1. 🔍 Chercher 'Docker' dans Spotlight (Cmd+Space)"
    echo "2. 🖱️  Cliquer sur 'Docker Desktop'"
    echo "3. ⏳ Attendre que l'icône Docker apparaisse dans la barre de menu"
    echo "4. ✅ L'icône doit être blanche (pas grise)"
    echo ""
    echo "Ou bien:"
    echo "• Ouvrir le Finder"
    echo "• Aller dans Applications"
    echo "• Double-cliquer sur Docker"
    echo ""
    echo -e "${BLUE}💡 Une fois Docker Desktop lancé, relancer ce script !${NC}"
    exit 1
fi

# 3. Vérifier l'espace disque
echo -e "${YELLOW}📋 Vérification de l'espace disque...${NC}"
docker_space=$(docker system df --format 'table {{.Size}}' 2>/dev/null | tail -n +2 | head -1 || echo "0B")
echo -e "${GREEN}✅ Espace Docker utilisé: ${docker_space}${NC}"

# 4. Test de fonctionnement
echo -e "${YELLOW}📋 Test de fonctionnement...${NC}"

if docker run --rm hello-world &> /dev/null; then
    echo -e "${GREEN}✅ Docker fonctionne correctement !${NC}"
else
    echo -e "${RED}❌ Problème avec Docker${NC}"
    echo "Essayer de redémarrer Docker Desktop"
    exit 1
fi

# 5. Vérification spécifique Trendr
echo -e "${YELLOW}📋 Préparation pour Trendr...${NC}"

# Vérifier les répertoires
if [ -d "logs" ] && [ -d "data" ]; then
    echo -e "${GREEN}✅ Répertoires Trendr présents${NC}"
else
    echo -e "${YELLOW}⚠️  Création des répertoires Trendr...${NC}"
    mkdir -p logs data/cache data/reports data/backups
    echo -e "${GREEN}✅ Répertoires créés${NC}"
fi

# Vérifier le fichier .env
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ Fichier .env présent${NC}"
    
    # Vérifier les clés
    if grep -q "your-" .env; then
        echo -e "${YELLOW}⚠️  Certaines clés API ne sont pas configurées${NC}"
    else
        echo -e "${GREEN}✅ Clés API configurées${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Fichier .env manquant${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}📝 Fichier .env créé depuis le template${NC}"
        echo -e "${RED}🛑 VOUS DEVEZ CONFIGURER VOS CLÉS API !${NC}"
    fi
fi

echo ""
echo -e "${GREEN}🎉 DOCKER EST PRÊT POUR TRENDR !${NC}"
echo ""
echo -e "${BLUE}🚀 PROCHAINES ÉTAPES :${NC}"
echo "1. Configurer le fichier .env avec vos clés API"
echo "2. Lancer: ./docker-start.sh"
echo "3. Choisir option 3 pour construire les images"
echo "4. Choisir option 1 pour démarrer le système"