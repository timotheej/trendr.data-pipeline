# GATTO Mention Scanner - KISS Edition

KISS (Keep It Simple, Stupid) mention scanner with 3 simplified modes and fixed scoring weights.

## Removed Code
- Complex weighted systems, legacy matcher_scores, experimental algorithms
- Strategy-based scanning, complex query builders, legacy match pipelines  
- Multi-source resolvers, threshold resolution chains, experimental config loaders

## 🆕 Récentes Améliorations

**4 Patches KISS** (additifs seulement, pas de refonte) :

1. **CSE Limite** : Défaut 30 résultats (max 50), CLI `--cse-num` avec clamp automatique
2. **Catégorie Auto** : Résolution dynamique depuis base POI, fallback `--category`, fini le hardcode "restaurant"  
3. **SERP-only Clarifié** : Documentation précise "site: uniquement, pas de requêtes ouvertes"
4. **Time Decay** : Flag `--time-decay` optionnel, décroissance exp. sur `published_at` (désactivé par défaut)

## 📋 Prérequis

Depuis la racine du repository :

```bash
export PYTHONPATH="$(pwd)"
export GOOGLE_CUSTOM_SEARCH_API_KEY="votre_clé_api"
export GOOGLE_CUSTOM_SEARCH_ENGINE_ID="votre_cx_id"
```

## 3 Modes (Simplified)

### balanced
Catalogue actif + CSE ouverte
- `collect_from_catalog_active_sources()` + `collect_from_cse()`

```bash
python -m scripts.mention_scanner \
  --mode balanced \
  --poi-names "Septime,Le Chateaubriand" \
  --city-slug paris
```

### serp-only
Uniquement sites/domaines listés (via site:)
- `collect_from_catalog_filtered(sources=...)` - pas de requêtes ouvertes

```bash
python -m scripts.mention_scanner \
  --mode serp-only \
  --poi-name "Le Rigmarole" \
  --sources "lefooding.com,timeout.fr"
```

### open
CSE ouverte sans sources
- `collect_from_cse()` seulement

```bash
python -m scripts.mention_scanner \
  --mode open \
  --poi-name "Septime" \
  --city-slug paris \
  --debug
```

## Golden Rule: CSE Templates
**Toujours inclure** `{poi_name}`, `{city_name}`, `{category}` dans tous les gabarits CSE.

Pays/langue: appliquer `gl`/`hl`/`cr`/`lr` selon le pays du POI.
Pas de filtre `site:.tld` par défaut.

## KISS Scoring
```
base_score = 0.60*name + 0.25*geo + 0.15*authority
final_score = (base_score - penalties) * time_decay_multiplier (clamp 0–1)
```

### 3 Composantes
- **name_score**: 2 signaux distincts (fuzzy + trigram) avec normalisation stopwords simples
- **geo_score**: distance + bonus arrondissement/CP (une seule source de vérité geo)  
- **authority**: confirmed > catalog > discovered (poids simple)

### 2 Pénalités
- **country mismatch** → reject immédiat
- **city mismatch** → -0.15 (clamp max_penalty)

### Time Decay (Optionnel)
- **Désactivé par défaut** - utilisez `--time-decay` pour activer
- **Décroissance exponentielle**: `exp(-age_days / tau_days)` avec tau=90j
- **Limite d'âge**: Articles > 365j → score = 0

## 🔧 Options CLI Détaillées

### Options Principales

```bash
--mode {balanced,open,serp-only}        # Mode de scan (défaut: balanced)
--poi-name "POI_NAME"                   # POI à scanner
--poi-names "POI1,POI2,POI3"           # Plusieurs POIs (CSV)
--sources "domain1.com,domain2.com"    # Domaines cibles (serp-only uniquement)
--city-slug {paris,lyon,marseille}     # Ville (défaut: paris)
```

### Paramètres CSE

```bash
--cse-num 30                           # Nombre de résultats CSE (défaut 30, max 50)
--no-cache                             # Désactiver le cache CSE
--allow-no-cse                         # Permettre exécution sans CSE
```

### Catégorie et Scoring

```bash
--category restaurant                  # Catégorie POI fallback (défaut: restaurant)
--time-decay                           # Activer décroissance temporelle sur published_at
--no-time-decay                        # Désactiver décroissance temporelle (défaut)
```

### Seuils de Scoring

```bash
--threshold-high 0.35                  # Seuil score élevé (surcharge config)
--threshold-mid 0.20                   # Seuil score moyen (surcharge config)
```

### Sortie & Debug

```bash
--jsonl-out path/to/output.jsonl       # Fichier JSONL de sortie
--log-drop-reasons                     # Logger les raisons de rejet
--dump-serp path/to/dump/dir           # Dumper les réponses CSE brutes
--debug                                # Logging debug détaillé
```

## 📊 Pipeline de Traitement

### 1. **Résolution Dynamique des Villes**

- **City Profile** : Chargement automatique des profils de ville
- **Locale Parameters** : Configuration CSE par pays (`gl=fr&cr=countryFR`)
- **Templates** : Remplacement de `{city_name}` par la ville réelle

### 2. **Génération de Requêtes**

- **Templates configurables** depuis `config.json`
- **Variables disponibles** : `{poi_name}`, `{poi_name_normalized}`, `{city_name}`, `{category_synonym}`
- **Catégorie auto** : Résolution depuis base POI, fallback CLI `--category` (défaut: restaurant)
- **Catégorie obligatoire** : Toutes les requêtes incluent le type (restaurant, bar, etc.)

### 3. **Recherche CSE Géolocalisée**

- **API Google CSE** avec paramètres géographiques
- **Cache persistant** avec TTL configurable
- **Rate limiting** et retry automatique
- **Filtrage pays** : Résultats locaux prioritaires

### 4. **Matching POI ↔ Article**

- **Normalisation** : Accents, ponctuation, casse
- **Score trigram** : Similarité textuelle
- **Token matching** : Discrimination par mots-clés
- **Seuils configurables** depuis config

### 5. **Scoring Multi-Composants**

#### Scoring Géographique (Configurable)

```json
"geo_scoring": {
  "city_name_score": 0.4,        // Paris détecté dans title/snippet
  "postal_code_score": 0.3,      // 75001-75020 détecté
  "admin_region_score": 0.2,     // Île-de-France détecté
  "country_score": 0.1,          // France détecté
  "url_city_segment_score": 0.3, // /paris/ dans l'URL
  "distance_full_score": 0.3,    // <3km du centroid ville
  "distance_half_score": 0.15    // 3-15km du centroid
}
```

#### Scoring Final

- **Name matching** : Correspondance nom POI (poids: 50%)
- **Geo signals** : Indicateurs géographiques (poids: 20%)
- **Category signals** : Type d'établissement (poids: 15%)
- **Authority** : Poids du domaine source (poids: 15%)

### 6. **Règles d'Acceptation Intelligentes**

```
Auto-accept SI :
  name_score >= 0.5 ET (geo_score >= 0.4 OU authority >= 0.5)

Fallback SI :
  name_score >= 0.5 ET authority >= 0.6 (sites officiels)

Reject sinon
```

### 7. **Déduplication Multi-Langue**

- **Fenêtre temporelle** configurée
- **Détection doublons** français/anglais
- **Filtrage domaines exclus** (réseaux sociaux, etc.)

## 📁 Architecture Modulaire

```
scripts/mention_scanner/
├── scanner.py              # Orchestrateur principal + CLI
├── city_profiles.py        # Profils villes + géolocalisation
├── cse_client.py          # Client Google CSE + cache + géoloc
├── config_resolver.py      # Résolution config centralisée
├── matching.py            # Matching POI-article + normalisation
├── scoring.py             # Scoring multi-composants configurable
├── dedup.py              # Déduplication intelligente
├── domains.py            # Résolution domaines → authority
└── logging_ext.py        # JSONL writer + summaries
```

## ⚙️ Configuration Centralisée

Tous les paramètres sont configurables via `config.json` (aucun hardcodé) :

### Configuration Mention Scanner

```json
{
  "mention_scanner": {
    "mode": "balanced",
    "match_score": {
      "high": 0.35,
      "mid": 0.2
    },
    "scoring": {
      "name_weight": 0.5,
      "geo_weight": 0.2,
      "cat_weight": 0.15,
      "authority_weight": 0.15,
      "max_penalty": 0.4
    },
    "geo_scoring": {
      "city_name_score": 0.4,
      "postal_code_score": 0.3,
      "admin_region_score": 0.2,
      "country_score": 0.1,
      "url_city_segment_score": 0.3,
      "distance_full_score": 0.3,
      "distance_half_score": 0.15,
      "distance_full_threshold_km": 3,
      "distance_half_threshold_km": 15
    },
    "limits": {
      "cse_num": 30,
      "poi_limit": 10,
      "max_candidates_per_poi": 100
    },
    "time_decay": {
      "enabled": false,
      "tau_days": 90,
      "max_age_days": 365
    },
    "debug_mode": {
      "threshold_high": 0.3,
      "threshold_mid": 0.15,
      "token_required_for_mid": false
    }
  }
}
```

### Configuration Templates

```json
{
  "query_strategy": {
    "templates": [
      "{multi_site_query} \"{poi_name} {category_synonym} {city_name}\""
    ],
    "global_templates": [
      "\"{poi_name}\" {category_synonym} {city_name}"
    ],
    "geo_hints": ["Paris", "750", "1er", "2e", ...],
    "category_synonyms": {
      "restaurant": ["restaurant", "bistrot", "brasserie"],
      "bar": ["bar à cocktails", "cocktail bar", "bar"],
      "cafe": ["café", "coffee shop", "coffee"]
    }
  }
}
```

## 🚀 Exemples d'Usage

### Scan Balanced Multi-Ville

```bash
# Paris - Recherche équilibrée
python -m scripts.mention_scanner \
  --mode balanced \
  --poi-name "L'Ambroisie" \
  --city-slug paris \
  --cse-num 15

# Lyon - Support natif
python -m scripts.mention_scanner \
  --mode balanced \
  --poi-name "Paul Bocuse" \
  --city-slug lyon \
  --cse-num 15
```

### Scan Multi-POI avec nouvelles fonctionnalités

```bash
# Avec catégorie personnalisée et time decay
python -m scripts.mention_scanner \
  --mode balanced \
  --poi-names "Septime,Le Chateaubriand,L'Astrance" \
  --city-slug paris \
  --category restaurant \
  --cse-num 45 \
  --time-decay \
  --jsonl-out out/multi_poi.jsonl

# Bar à vin avec limite CSE élevée  
python -m scripts.mention_scanner \
  --mode open \
  --poi-name "Le Mary Celeste" \
  --city-slug paris \
  --category "bar à vin" \
  --cse-num 50
```

### Debug Mode avec Seuils Abaissés

```bash
# Active le debug mode depuis la config
export SCAN_DEBUG=1

python -m scripts.mention_scanner \
  --mode balanced \
  --poi-name "Restaurant Test" \
  --city-slug paris \
  --debug \
  --log-drop-reasons
```

### Scan Ciblé avec Sources Spécifiques

```bash
python -m scripts.mention_scanner \
  --mode serp-only \
  --poi-name "Pierre Hermé" \
  --sources "lefooding.com,gaultmillau.com,guide.michelin.com" \
  --city-slug paris \
  --no-cache \
  --dump-serp out/debug
```

## 📊 Exemple de Summary

```
🎯 Balanced Scan Results:
  • Total mentions: 23
  • Accepted: 18
  • Rejected: 5
  • CSE calls: 8
  • Cache hits: 2
  • Top domains: {'fr.gaultmillau.com': 5, 'guide.michelin.com': 4, 'lefooding.com': 3}
```

## 📄 Format JSONL

Chaque ligne contient une mention avec sa décision :

```json
{
  "poi_id": "3deddea0-c61d-46c4-9d3f-3f5c13945d5b",
  "poi_name": "L'Ambroisie",
  "query": "\"L'Ambroisie\" restaurant Paris",
  "domain": "guide.michelin.com",
  "url": "http://guide.michelin.com/fr/fr/ile-de-france/paris/restaurant/l-ambroisie",
  "score": 0.6628571428571429,
  "threshold_used": "high=0.35,mid=0.2",
  "decision": "accept",
  "drop_reasons": [],
  "strategy": "balanced",
  "ts": "2025-09-13T18:41:03.823880+00:00"
}
```

## 🔍 Debug et Monitoring

### Mode Debug Complet

```bash
export SCAN_DEBUG=1  # Active les seuils debug depuis config
--debug --log-drop-reasons --dump-serp out/debug
```

### Audit Détaillé

Le mode debug génère un audit JSONL complet dans `out/audit_*.jsonl` avec :

- Geo signals détectés
- Name matching breakdown
- Score components
- Sanity checks

### Vérification Géolocalisation

```bash
# Vérifier les paramètres CSE par ville
python -c "
from scripts.mention_scanner.city_profiles import city_manager
print('Paris:', city_manager.get_search_locale('paris'))
print('Lyon:', city_manager.get_search_locale('lyon'))
"
```

## 🔄 Intégration Pipeline

Le scanner s'intègre parfaitement avec `run_pipeline.py` :

```bash
# Scan mentions avec pipeline complet
python run_pipeline.py \
  --mode mentions \
  --seed-poi-name "L'Ambroisie" \
  --seed-city paris \
  --cse-num 15 \
  --debug
```

## 📈 Améliorations Récentes

### ✅ Architecture Nettoyée

- Suppression du code mort (`ContentFetcher`)
- Configuration 100% centralisée
- Aucun hardcodé restant

### ✅ Support Multi-Ville

- Templates dynamiques par ville
- Géolocalisation CSE automatique
- Profils de ville extensibles (Paris, Lyon, Marseille)

### ✅ Géolocalisation Intelligente

- Paramètres CSE par pays (`gl=fr&cr=countryFR`)
- Réduction du bruit multilingue
- Scoring géographique configurable

### ✅ Scoring Avancé

- Règles d'acceptation intelligentes
- Debug mode configurable
- Audit complet des décisions

---

**Note** : Ce scanner est entièrement configurable et n'utilise aucune valeur en dur. Tous les paramètres, seuils et scores proviennent de `config.json` avec fallbacks documentés. L'architecture modulaire permet une extension facile pour de nouvelles villes et fonctionnalités.
