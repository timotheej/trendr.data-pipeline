# 🤖 Trendr Automation System

## Vue d'ensemble

Le système Trendr est entièrement autonome avec monitoring 24/7, gestion des quotas API, et exécution programmée des tâches.

## 🚀 Démarrage du Système Automatisé

### 1. Démarrage Daemon (Recommandé pour Production)

```bash
# Démarrer le système de monitoring en arrière-plan
./start-monitoring.sh daemon

# Vérifier que le monitoring tourne
tail -f logs/monitoring.log
```

### 2. Démarrage Foreground (Développement/Debug)

```bash
# Démarrer en mode visible pour debug
./start-monitoring.sh foreground
```

### 3. Arrêt du Système

```bash
# Arrêter proprement le monitoring
./stop-monitoring.sh
```

## 📅 Planification Automatique

Le système exécute automatiquement :

### Tâches Quotidiennes

- **02:00** - Pipeline complet (`full`)
  - Ingestion de nouveaux POIs (rotation par arrondissement)
  - Classification et social proofs
  - Mise à jour des collections
  - Traitement des photos

- **14:00** - Classification seulement (`classification`)
  - Mise à jour des social proofs
  - Reclassification des POIs récents

- **18:00** - Collections seulement (`collections`)
  - Régénération des collections
  - Optimisation SEO

### Monitoring Continu

- **Toutes les 5 minutes** - Health Check
  - Vérification quotas API
  - État base de données
  - Alertes système

## 📊 Gestion des Quotas API

### Limites Quotidiennes
- **Google Search API** : 95 requêtes/jour (5 de marge)
- **Google Places API** : Inclus dans le système de rotation
- **Seuil d'alerte** : 80% du quota utilisé

### Optimisations Automatiques
- Rotation par arrondissement (1 par jour)
- Social proof avant API calls coûteux
- Cache intelligent des résultats
- Prioritisation des POIs tendance

## 🎯 Tables de Monitoring

### `api_usage`
Tracking quotas en temps réel :
```sql
SELECT date, api_type, queries_count 
FROM api_usage 
ORDER BY date DESC;
```

### `monitoring_reports`
Rapports santé système :
```sql
SELECT monitoring_date, summary->'system_health' as health
FROM monitoring_reports 
ORDER BY monitoring_date DESC;
```

## 🔧 Configuration Avancée

### Modifier les Horaires

Éditer `monitoring_system.py` :
```python
'pipeline_schedule': {
    'full_run': '02:00',      # Pipeline complet
    'classification_only': '14:00',  # Classification
    'collections_only': '18:00'     # Collections
}
```

### Ajuster les Seuils

```python
'daily_api_limit': 95,        # Limite quotidienne
'alert_threshold': 80,        # Seuil d'alerte (%)
'max_errors_per_day': 10,     # Erreurs max
'health_check_interval': 300, # Interval santé (sec)
```

## 🛠️ Commandes Utiles

### Health Check Manuel

```bash
# Vérifier l'état du système
python3 monitoring_system.py --health-check
```

### Exécution Pipeline Manuelle

```bash
# Pipeline complet
python3 monitoring_system.py --run-pipeline full

# Classification seulement  
python3 monitoring_system.py --run-pipeline classification

# Collections seulement
python3 monitoring_system.py --run-pipeline collections
```

### Tests Système

```bash
# Tester le monitoring
python3 test-monitoring.py

# Test pipeline direct
python3 run_pipeline.py --city Paris --mode full
```

## 📋 Crontab Alternative

Si vous préférez cron à la place du daemon :

```bash
# Ajouter à crontab -e
0 2 * * * cd /path/to/trendr && python3 run_pipeline.py --city Paris --mode full >> daily_pipeline.log 2>&1
0 14 * * * cd /path/to/trendr && python3 run_pipeline.py --city Paris --mode classification >> classification.log 2>&1
0 18 * * * cd /path/to/trendr && python3 run_pipeline.py --city Paris --mode collections >> collections.log 2>&1
*/5 * * * * cd /path/to/trendr && python3 monitoring_system.py --health-check >> health.log 2>&1
```

## 🚨 Alertes et Erreurs

### Alertes Automatiques
- Quota API > 80%
- Erreurs > 10/jour
- Base de données inaccessible
- Pipeline en échec

### Logs à Surveiller
- `logs/monitoring.log` - Logs système
- `data_pipeline.log` - Logs pipeline
- Supabase monitoring_reports - Historique

## 🎯 États du Système

### system_health
- `healthy` - Tout fonctionne
- `warning` - Seuils atteints
- `error` - Problèmes critiques

### pipeline_status  
- `idle` - En attente
- `running` - En cours
- `completed` - Terminé avec succès
- `failed` - Échec
- `error` - Erreur critique

## 🔍 Troubleshooting

### Le monitoring ne démarre pas
```bash
# Vérifier les dépendances
pip3 install -r requirements.txt

# Vérifier la configuration
python3 -c "import config; print('Config OK')"
```

### Quota API dépassé
```bash
# Vérifier l'usage
python3 -c "
from monitoring_system import TrendrMonitoringSystem
m = TrendrMonitoringSystem()
print(m.check_api_quota())
"
```

### Base de données inaccessible
```bash
# Test connexion
python3 -c "
from utils.database import SupabaseManager
db = SupabaseManager()
print(db.client.table('poi').select('id').limit(1).execute())
"
```

---

## 🚀 Résumé : Système 100% Autonome

Une fois démarré avec `./start-monitoring.sh daemon`, le système :

✅ **Fonctionne 24/7** sans intervention  
✅ **Gère les quotas API** automatiquement  
✅ **Exécute les tâches** selon planning  
✅ **Surveille la santé** du système  
✅ **Enregistre tout** en base de données  
✅ **Optimise les coûts** intelligemment  

Le système Trendr est **production-ready** ! 🎉