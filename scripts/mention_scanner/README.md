# Gatto Mention Scanner - Documentation

Le Gatto Mention Scanner est un système modulaire de détection et scoring de mentions de POIs (Points d'Intérêt) dans les résultats de recherche.

## 📋 Prérequis

Depuis la racine du repository :
```bash
export PYTHONPATH="$(pwd)"
```

## 🎯 Modes de Scanning

### 1. Mode `open` - Recherche Globale
Effectue des recherches globales sans restriction de domaine.

```bash
python -m scripts.mention_scanner.scanner \
  --mode open \
  --poi-name "Le Rigmarole" \
  --cse-num 10 \
  --jsonl-out out/mentions.open.jsonl \
  --debug
```

**Caractéristiques :**
- **Requêtes générées** : `"POI_NAME" Paris`, `"poi_name_normalized" Paris`
- **Pas de prefix `site:`** : Recherche sur tout le web
- **Templates utilisés** : `global_templates` uniquement
- **Portée** : Large, tous domaines confondus

### 2. Mode `serp-only` - Recherche Ciblée
Effectue des recherches ciblées sur des domaines spécifiques.

```bash
python -m scripts.mention_scanner.scanner \
  --mode serp-only \
  --poi-name "Le Rigmarole" \
  --sources "lefooding.com,timeout.fr,sortiraparis.com" \
  --cse-num 10 \
  --jsonl-out out/mentions.serp.jsonl \
  --debug
```

**Caractéristiques :**
- **Requêtes générées** : `site:domain.com "POI_NAME"`, `site:domain.com "POI_NAME Paris"`, etc.
- **Prefix `site:`** : Recherche restreinte par domaine
- **Templates utilisés** : `templates` + `global_templates`
- **Portée** : Ciblée, domaines spécifiés

### 3. Mode `balanced` - Hybride (Par défaut)
Actuellement utilise l'implémentation `serp-only` comme fallback.

```bash
python -m scripts.mention_scanner.scanner \
  --mode balanced \
  --poi-name "Le Rigmarole" \
  --sources "lefooding.com,timeout.fr" \
  --debug
```

## 🔧 Options CLI Détaillées

### Options Principales
```bash
--mode {balanced,open,serp-only,strategy}   # Mode de scan (défaut: balanced)
--poi-name "POI_NAME"                       # POI à scanner
--poi-names "POI1,POI2,POI3"               # Plusieurs POIs (CSV)
--sources "domain1.com,domain2.com"        # Domaines cibles (serp-only)
--city-slug paris                          # Ville (défaut: paris)
```

### Paramètres CSE
```bash
--cse-num 10                               # Nombre de résultats CSE (1-10)
--no-cache                                 # Désactiver le cache CSE
--allow-no-cse                             # Permettre exécution sans CSE
```

### Seuils de Scoring
```bash
--threshold-high 0.92                      # Seuil score élevé (surcharge config)
--threshold-mid 0.85                       # Seuil score moyen (surcharge config)
```

### Sortie & Debug
```bash
--jsonl-out path/to/output.jsonl           # Fichier JSONL de sortie
--log-drop-reasons                         # Logger les raisons de rejet
--dump-serp path/to/dump/dir               # Dumper les réponses CSE brutes
--debug                                    # Logging debug détaillé
```

## 📊 Pipeline de Traitement

### 1. **Génération de Requêtes**
- **Templates depuis config** : `config.json` → `mention_scanner.query_strategy`
- **Variables disponibles** : `{poi_name}`, `{poi_name_normalized}`, `{domain}`, `{geo_hint}`, `{category_synonym}`
- **Déduplication** : Ordre conservé, limite `max_templates_per_poi`

### 2. **Recherche CSE**
- **API Google Custom Search** avec rate limiting
- **Cache persistant** avec TTL configurable
- **Retry automatique** sur erreurs 429/5xx

### 3. **Matching POI ↔ Article**
- **Normalisation** : Suppression accents, ponctuation
- **Score trigram** : Similarité textuelle
- **Score géographique** : Indices Paris/arrondissements
- **Seuils configurables** : high/mid depuis config

### 4. **Scoring Final**
- **Authority score** : Poids par domaine/groupe
- **Name match** : Correspondance nom POI
- **Geo hints** : Indicateurs géographiques
- **Time decay** : Décroissance temporelle
- **Formule** : Score final = Σ(composants × poids)

### 5. **Règles d'Acceptation**
```
Accepté SI : score >= threshold_high ET geo_score >= threshold_mid
```

### 6. **Déduplication**
- **Fenêtre temporelle** : `dedup.window_days` depuis config
- **Déduplication par URL** : Évite les doublons

## 📁 Architecture Modulaire

```
scripts/mention_scanner/
├── scanner.py              # Point d'entrée CLI + orchestrateur
├── config_resolver.py      # Résolution config + seuils
├── cse_client.py          # Client Google CSE + cache
├── matching.py            # Matching POI-article + normalisation
├── scoring.py             # Algorithmes de scoring + acceptation
├── dedup.py              # Déduplication temporelle
├── logging_ext.py        # JSONL writer + summary
└── domains.py            # Extraction domaines + utilitaires
```

## 📊 Exemple de Summary

```
==================================================
GATTO MENTION SCANNER - RUN SUMMARY
==================================================
POIs processed: 1
CSE queries: 6
SERP items returned: 39
Cache hits/misses: 4/2 (66.7% hit rate)
Candidates found: 0
Accepted: 0
Rejected: 0
Upserts: 0
Top domains (count):
  lefooding.com: 5
  timeout.fr: 3
==================================================
```

## 📄 Format JSONL

Chaque ligne du fichier JSONL contient :
```json
{
  "poi_id": "poi_Le_Rigmarole",
  "poi_name": "Le Rigmarole", 
  "query": "site:lefooding.com \"Le Rigmarole\"",
  "domain": "lefooding.com",
  "url": "https://lefooding.com/restaurant/le-rigmarole",
  "score": 0.89,
  "threshold_used": "high=0.92,mid=0.85",
  "decision": "reject",
  "drop_reasons": ["score_too_low: 0.89 < 0.92"],
  "strategy": "serp_only",
  "ts": "2025-09-10T18:30:45.123Z"
}
```

## ⚙️ Configuration

Tous les paramètres sont configurables via `config.json` :

```json
{
  "mention_scanner": {
    "match_score": {
      "high": 0.92,
      "mid": 0.85
    },
    "limits": {
      "cse_num": 10,
      "max_templates_per_poi": 6
    },
    "query_strategy": {
      "templates": [
        "site:{domain} \"{poi_name}\"",
        "site:{domain} \"{poi_name_normalized}\"", 
        "site:{domain} \"{poi_name} {geo_hint}\"",
        "site:{domain} \"{poi_name}\" {category_synonym}"
      ],
      "global_templates": [
        "\"{poi_name}\" Paris",
        "\"{poi_name_normalized}\" Paris"
      ],
      "geo_hints": ["Paris", "750", "1er", "2e", ...],
      "category_synonyms": {
        "restaurant": ["restaurant", "bistrot", "brasserie"]
      }
    },
    "dedup": {
      "window_days": 21
    }
  }
}
```

## 🚀 Exemples d'Usage

### Scan Ouvert Multi-POI
```bash
python -m scripts.mention_scanner.scanner \
  --mode open \
  --poi-names "Septime,Le Chateaubriand,L'Astrance" \
  --cse-num 5 \
  --jsonl-out out/multi_poi.jsonl
```

### Scan Ciblé avec Cache Désactivé
```bash
python -m scripts.mention_scanner.scanner \
  --mode serp-only \
  --poi-name "Pierre Hermé" \
  --sources "lefooding.com,sortiraparis.com" \
  --no-cache \
  --dump-serp out/debug \
  --debug
```

### Test avec Seuils Personnalisés
```bash
python -m scripts.mention_scanner.scanner \
  --mode serp-only \
  --poi-name "Du Pain et des Idées" \
  --sources "timeout.fr" \
  --threshold-high 0.8 \
  --threshold-mid 0.6 \
  --log-drop-reasons
```

## 🔍 Debug et Troubleshooting

### Activer le Debug Complet
```bash
--debug --log-drop-reasons --dump-serp out/debug
```

### Vérifier la Configuration CSE
```bash
echo $GOOGLE_CUSTOM_SEARCH_API_KEY
echo $GOOGLE_CUSTOM_SEARCH_ENGINE_ID
```

### Analyser les Rejets
Le flag `--log-drop-reasons` log les raisons précises de rejet dans le JSONL.

## 🔄 Migration depuis l'Ancien Scanner

- ❌ **`scripts/gatto_mention_scanner.py`** → supprimé
- ✅ **`python -m scripts.mention_scanner.scanner`** → nouvelle interface
- ✅ **Arguments CLI identiques** → compatibilité préservée
- ✅ **Logique métier inchangée** → mêmes décisions

---

**Note** : Ce scanner est entièrement configurable et n'utilise aucune valeur en dur. Tous les seuils, limites et paramètres proviennent de `config.json` avec des fallbacks documentés.