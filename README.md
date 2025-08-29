# 🚀 Trendr Data Pipeline - Production Ready

## Overview

Système autonome d'ingestion et classification POIs optimisé pour **coût minimal** et **données fraîches**.

### 🎯 Fonctionnalités Clés

- **Rotation par arrondissements** : Détection nouveaux POIs (1.89€/mois pour Paris)
- **Classification sociale intelligente** : Tags basés sur social proofs + Michelin, médias
- **Collections dynamiques** : 7 types ("Romantic Date Spots", "Instagram-Worthy", etc.)
- **Système autonome** : 24/7 sans intervention
- **Optimisé API coûts** : 92% réduction vs système précédent

## 🏃‍♂️ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Créer `.env` avec :
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GOOGLE_PLACES_API_KEY=your_places_key
GOOGLE_CUSTOM_SEARCH_API_KEY=your_search_key
GOOGLE_CUSTOM_SEARCH_ENGINE_ID=your_engine_id
```

### 3. Lancement Autonome

#### Système Automatisé (Recommandé)
```bash
# Démarrer le système de monitoring 24/7
./start-monitoring.sh daemon

# Arrêter le système
./stop-monitoring.sh

# Voir la documentation complète
cat AUTOMATION.md
```

#### Exécution Manuelle
```bash
# Pipeline complet 
python run_pipeline.py --city Paris --mode full

# Classification seulement
python run_pipeline.py --city Paris --mode classification

# Collections seulement  
python run_pipeline.py --city Paris --mode collections
```

## 📊 Architecture Optimisée

### Rotation Intelligente
- **Jour 1** : 1er arrondissement → **Jour 20** : 20ème arrondissement
- **Détection nouveaux POIs** via comparaison place_ids
- **Coût quotidien** : ~0.06€ (4 catégories × 3 POIs × $0.017)

### Classification Social Proof
```python
# Tags détectés automatiquement
trendy_tags = ["michelin_mentioned", "media_featured", "photo-worthy"]
hidden_gem_tags = ["local-favorite", "authentic", "wine_specialist"] 
```

### Collections Basées Tags
- **Romantic Date Spots** : `date-spot + intimate`
- **Instagram-Worthy** : `photo-worthy + trendy`  
- **Locals Only** : `local-favorite - tourist-friendly`

## 🛠️ Scripts Utiles

### Ingestion Ciblée
```bash
# Test rotation avec quartier spécifique
python scripts/google_places_ingester.py --test

# Quartier spécifique
python scripts/google_places_ingester.py --neighborhood "Marais"
```

### Classification
```bash
# Classification test (3 POIs)
python scripts/intelligent_classifier.py --test

# POI spécifique
python scripts/intelligent_classifier.py --poi-name "Breizh Café"
```

### Collections
```bash
# Générer collections pour Paris
python ai/collection_generator.py --city Paris
```

### Maintenance
```bash
# Nettoyage base de données
python scripts/cleanup_database.py --analyze-only
python scripts/cleanup_database.py --execute
```

## 📈 Performance Production

### Coûts Optimisés
- **Ancien système** : ~540€/mois pour 216 POIs
- **Nouveau système** : ~45€/mois pour même coverage
- **Réduction** : 92%

### Scalabilité
- **Paris** : 1.89€/mois (20 arrondissements)
- **+ Lyon** : +1.89€/mois (9 arrondissements) 
- **+ Marseille** : +1.89€/mois (16 arrondissements)

### Résultats Typiques (par jour)
- **11 nouveaux POIs** ingérés
- **Photos automatiques** pour chaque POI
- **Classification** avec confidence 70%+
- **Collections** mises à jour automatiquement

## 🚀 Déploiement

### Docker
```bash
./docker-start.sh
```

### Cron Job (Recommandé)
```bash
# Crontab : tous les jours à 2h du matin
0 2 * * * cd /path/to/trendr && python run_pipeline.py --city Paris --mode full >> daily_pipeline.log 2>&1
```

## 🔧 Configuration Avancée

### `config.json` Production
```json
{
  "pipeline_config": {
    "daily_api_limit": 950,
    "batch_size": 25,
    "max_pois_per_category": 150
  },
  "api_cost_optimization": {
    "optimized_fields": "place_id,name,formatted_address,geometry,types,rating,user_ratings_total,website,price_level",
    "max_api_updates_per_day": 20
  }
}
```

## 📁 Structure Projet

```
trendr-data-pipeline/
├── 🚀 run_pipeline.py          # Point d'entrée principal
├── 📊 config.json              # Configuration optimisée
├── scripts/
│   ├── google_places_ingester.py    # Ingestion rotation
│   ├── intelligent_classifier.py    # Classification sociale
│   └── cleanup_database.py          # Maintenance
├── utils/
│   ├── database.py              # Manager Supabase optimisé
│   └── photo_manager.py         # Gestion photos automatique
└── ai/
    └── collection_generator.py  # Collections dynamiques
```

## ✅ Monitoring

### Logs Importants
```bash
tail -f daily_pipeline.log  # Pipeline principal
tail -f data_pipeline.log   # Détails techniques
```

### Métriques Supabase
- **POIs table** : 239+ entrées
- **Collections table** : 7+ collections actives
- **Proof_sources table** : 758+ social proofs

## 🎯 Prêt pour Production

Le système est **autonome**, **cost-efficient**, et **scalable**. 

**Lancement production** :
```bash
python run_pipeline.py --city Paris --mode full
```

---

*Développé pour Trendr - Fresh data, social proof driven* 🚀