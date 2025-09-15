# Google Places Ingester

Un outil pour ingérer des POIs (Points of Interest) depuis l'API Google Places vers la base de données Supabase du projet Gatto.

## 🎯 Fonctionnalités

- **Ingestion par nom** : Recherche et ingestion d'un POI par son nom
- **Ingestion par Place ID** : Ingestion directe avec un Google Place ID
- **Association spatiale automatique** : Attribution automatique du quartier et arrondissement parisien
- **Gestion des doublons** : Détection et mise à jour des POIs existants
- **Limitation de débit** : Respect des quotas API Google Places
- **Mode dry-run** : Test sans modification de la base

## 📋 Usage

### Commande de base
```bash
python scripts/google_places_ingester.py --poi-name "Nom du Restaurant" --city Paris --commit
```

### Options principales
```bash
# Ingestion par nom (recommandé)
python scripts/google_places_ingester.py --poi-name "Breizh Café" --city Paris --commit

# Ingestion par Google Place ID
python scripts/google_places_ingester.py --place-id "ChIJ..." --city Paris --commit

# Mode dry-run (test sans commit)
python scripts/google_places_ingester.py --poi-name "Le Mary Celeste" --city Paris --dry-run

# Output JSON pour debug
python scripts/google_places_ingester.py --poi-name "Pierre Hermé" --city Paris --json-output
```

### Paramètres

| Paramètre | Description | Exemple |
|-----------|-------------|---------|
| `--poi-name` | Nom du POI à rechercher | `"L'As du Fallafel"` |
| `--place-id` | Google Place ID direct | `"ChIJ..."` |
| `--city` | Ville de recherche | `"Paris"` |
| `--commit` | Sauvegarder en base (obligatoire) | - |
| `--dry-run` | Mode test sans sauvegarde | - |
| `--json-output` | Sortie au format JSON | - |

## 🏗️ Architecture

### Pipeline d'ingestion
1. **Recherche** : Text Search API → candidats potentiels
2. **Validation** : Sélection du meilleur candidat (score, distance)
3. **Détails** : Place Details API → informations complètes
4. **Normalisation** : Nettoyage et formatage des données
5. **Sauvegarde** : Upsert en base avec association spatiale

### Données extraites

#### Informations de base
- **Identifiants** : `google_place_id`, `name`
- **Localisation** : `lat`, `lng`, `address_street`, `city`, `country`
- **Catégorisation** : `category`, `subcategories`
- **Contact** : `phone`, `website`

#### Données enrichies
- **Horaires** : `opening_hours` (format `{"periods": [...]}`)
- **Prix** : `price_level` (`free`, `inexpensive`, `moderate`, `expensive`, `very_expensive`)
- **Photos** : `primary_photo_ref` (référence Google)
- **Géographie** : `district_name`, `neighbourhood_name` (arrondissement + quartier)

#### Métadonnées
- **Statut** : `eligibility_status` (toujours `"hold"` en V1)
- **Badges** : `badges`, `buzz_score`, `gatto_score`
- **Tracking** : `created_at`, `updated_at`, `last_ingested_from_google_at`

## 🗺️ Association spatiale

L'ingester associe automatiquement chaque POI à :
- **Arrondissement** (`district_name`) : Basé sur `admin_level=9`
- **Quartier** (`neighbourhood_name`) : Basé sur `admin_level≥10` ou `place`

Utilise la fonction SQL `update_all_paris_pois()` avec l'opérateur spatial `ST_Covers`.

## 🔧 Configuration

### Variables d'environnement (.env)
```bash
GOOGLE_PLACES_API_KEY=your_google_places_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

### Quotas et limites
- **Budget quotidien** : $1.50 (configuré dans `config.json`)
- **Requêtes max/jour** : 950
- **Limitation débit** : Token bucket (5000 tokens)
- **Champs API** : Optimisés pour réduire les coûts

## 🚀 Intégration pipeline

### Via run_pipeline.py (recommandé)
```bash
# Ingestion + scan mentions automatique
python run_pipeline.py --mode ingest --seed-poi-name "Septime" --seed-city Paris
```

### Gestion des erreurs
- **POI non trouvé** : Message d'information, pas d'erreur
- **API quota dépassé** : Arrêt automatique avec message
- **Doublon détecté** : Mise à jour avec `updated_at`
- **Coordonnées invalides** : Validation et rejet

## 📊 Format des données

### Opening Hours
```json
{
  "periods": [
    {
      "open": {"day": 1, "time": "0900"},
      "close": {"day": 1, "time": "2300"}
    }
  ]
}
```
- **Jour** : 0=Dimanche, 1=Lundi, ..., 6=Samedi
- **Heure** : Format HHMM (24h)
- **Note** : `open_now` supprimé (temporaire)

### Catégories
Mapping des types Google vers catégories Gatto :
- `restaurant`, `bar`, `cafe`, `shop`, `service`, `attraction`, `hotel`

### Photos
- **Stockage** : Références Google (`primary_photo_ref`)
- **Affichage** : Via Google Places Photo API
- **Pas de stockage local** (architecture KISS V1)

## 🔍 Debug et monitoring

### Logs détaillés
```bash
# Mode verbose avec tous les détails
python scripts/google_places_ingester.py --poi-name "Restaurant" --city Paris --json-output --dry-run
```

### Vérification ingestion
```bash
# Check si POI existe déjà
python -c "
from supabase import create_client
import config
client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
result = client.table('poi').select('name,district_name,neighbourhood_name').eq('name', 'Nom POI').execute()
print(result.data)
"
```

## ⚠️ Limitations actuelles

1. **Géographie** : Optimisé pour Paris uniquement
2. **Photos** : Une seule photo de référence (`primary_photo_ref`)
3. **Prix** : `price_level` souvent NULL (données Google limitées)
4. **Statut** : Tous les POIs en `"hold"` par défaut
5. **Horaires** : Pas de calcul `open_now` dynamique

## 🛠️ Développement

### Tests
```bash
# Test complet avec POI existant
python scripts/google_places_ingester.py --poi-name "Breizh Café" --city Paris --dry-run

# Test avec nouveau POI
python scripts/google_places_ingester.py --poi-name "Restaurant Test" --city Paris --dry-run
```

### Extensions futures
- Support multi-villes
- Gestion photos multiples
- Calcul scores qualité
- Validation coordonnées avancée
- Mode batch pour ingestion massive