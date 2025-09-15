# Neighbourhoods / Urban Areas Management

Scripts pour l'extraction et l'ingestion des quartiers/zones urbaines dans Gatto.

## 📁 Structure

```
neighbourhoods/
├── README.md                           # Ce fichier
├── extract_neighbourhoods.py          # Extraction depuis OpenStreetMap
├── ingest_neighbourhoods.py          # Ingestion métadonnées (via Supabase REST API)
├── generate_geometry_sql.py           # Génération SQL pour géométries
├── ingest_geometries.py              # Ingestion directe PostgreSQL (non utilisé)
├── neighbourhoods_paris.geojson      # Données extraites pour Paris
├── neighbourhoods_marseille.geojson  # Données extraites pour Marseille  
├── pending_geometries_paris.jsonl    # Géométries en attente d'injection
└── pending_geometries_paris.sql      # Script SQL généré
```

## 🚀 Workflow Complet

### 1. Extraction depuis OpenStreetMap
```bash
python neighbourhoods/extract_neighbourhoods.py "Paris"
# → Génère neighbourhoods_paris.geojson
```

### 2. Ingestion des métadonnées
```bash
python neighbourhoods/ingest_neighbourhoods.py neighbourhoods/neighbourhoods_paris.geojson "Paris"
# → Ingère les métadonnées + génère pending_geometries_paris.jsonl
```

### 3. Injection des géométries
```bash
# Option A: Générer du SQL à exécuter manuellement
python neighbourhoods/generate_geometry_sql.py neighbourhoods/pending_geometries_paris.jsonl
# → Génère pending_geometries_paris.sql à exécuter dans Supabase

# Option B: Injection directe PostgreSQL (nécessite psycopg2)
python neighbourhoods/ingest_geometries.py neighbourhoods/pending_geometries_paris.jsonl
```

## 🗄️ Schema Base

```sql
-- Table urban_areas
CREATE TABLE urban_areas (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    city_name text NOT NULL,
    name text NOT NULL,
    type text NOT NULL CHECK (type IN ('admin', 'place')),
    admin_level text,
    place_type text,
    geometry geometry(GEOMETRY, 4326),
    created_at timestamp DEFAULT now(),
    UNIQUE(city_name, name, type)
);
```

## 📊 Types de Données

- **admin**: Quartiers administratifs (arrondissements, quartiers officiels)
- **place**: Lieux nommés (neighbourhoods, quarters, suburbs)

## 🔧 Configuration

Les scripts utilisent la configuration Gatto existante (`config.py` + `.env`) :
- `SUPABASE_URL`
- `SUPABASE_KEY`

## 💡 Notes Techniques

- **Limitation API REST**: Supabase REST API ne gère pas bien les géométries PostGIS complexes
- **Solution**: Ingestion métadonnées via REST + géométries via SQL direct
- **Format**: GeoJSON → PostGIS via `ST_GeomFromGeoJSON()`
- **Conflits**: Gestion via `ON CONFLICT (city_name, name, type)`

## 🧹 Nettoyage

Pour nettoyer les fichiers temporaires :
```bash
rm neighbourhoods/pending_geometries_*.jsonl
rm neighbourhoods/pending_geometries_*.sql
```