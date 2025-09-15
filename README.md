# Gatto Data Pipeline

POI → quartiers/arrondissements OSM mapping pipeline with robust spatial ingestion and association.

## Features

- **Robust Spatial Ingestion**: Normalizes geometries with proper SRID, validation, and MultiPolygon conversion
- **Reliable POI Association**: Associates POIs to urban areas using ST_Covers with smart tie-breaking
- **Quality Assurance**: Comprehensive spatial data validation and coordinate integrity checks
- **CLI Tools**: Easy-to-use command-line interface for all operations
- **Idempotent Operations**: All scripts can be safely re-run without data corruption

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your Supabase credentials
   ```

2. **Setup Database Schema**:
   ```bash
   make spatial-setup
   ```

3. **Extract and Ingest Neighborhoods**:
   ```bash
   make extract-neighborhoods CITY="Paris"
   make ingest-geometries FILE=neighbourhoods_paris.geojson
   ```

4. **Associate POIs to Urban Areas**:
   ```bash
   make associate-pois CITY="Paris"
   ```

5. **Run Quality Checks**:
   ```bash
   make qa-spatial CITY="Paris"
   ```

## Spatial Ingestion & Linking

### Geometry Normalization

All spatial data is automatically normalized using the robust chain:
```sql
ST_Multi(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)))
```

This ensures:
- ✅ Proper SRID 4326 (WGS84)
- ✅ Valid geometries (no self-intersections)  
- ✅ Consistent MultiPolygon type
- ✅ Indexed spatial operations

### POI Association Strategy

POIs are associated to urban areas using a robust spatial matching algorithm:

1. **Primary Operator**: `ST_Covers` (more reliable than `ST_Contains`)
2. **Tie-breaking**: Smallest area first for precision
3. **Administrative Priority**: Arrondissements → Quartiers → Places
4. **Fallback Methods**: `ST_Intersects` for edge cases

### Database Triggers

Automatic geometry conversion triggers protect data integrity:
- Converts `geometry_geojson` → normalized `geometry` on insert/update
- Validates geometry quality and logs warnings
- Handles conversion errors gracefully

## Architecture

```
data-pipeline/
├── sql/                          # Database schema & functions
│   ├── schema_urban_areas.sql    # Table structure with proper constraints
│   ├── triggers_geometry_normalization.sql  # Auto-conversion triggers
│   ├── functions_poi_zone_linking.sql       # Spatial association functions
│   └── qa_spatial_checks.sql     # Comprehensive QA queries
├── scripts/
│   ├── associate_pois.py         # CLI for POI association
│   └── spatial_qa.py            # Quality assurance tool
├── neighbourhoods/
│   ├── extract_neighbourhoods.py # OSM data extraction
│   ├── ingest_geometries.py      # Geometry ingestion (normalized)
│   └── generate_geometry_sql.py  # SQL generation (normalized)
└── Makefile                     # Convenient commands
```

## Commands

### Data Operations
```bash
# Extract neighborhoods from OSM
make extract-neighborhoods CITY="Paris"

# Ingest geometries with normalization
make ingest-geometries FILE=neighbourhoods_paris.geojson

# Associate POIs to urban areas
make associate-pois CITY="Paris"              # Using ST_Covers (recommended)
make associate-pois-intersects CITY="Paris"   # Using ST_Intersects (fallback)
make associate-pois-dry-run CITY="Paris"      # Simulate without changes
```

### Quality Assurance
```bash
# Comprehensive spatial QA
make qa-spatial CITY="Paris"

# Quick essential checks
make qa-quick CITY="Paris"

# Coordinate validation (detect lat/lng swaps)
make qa-coordinates CITY="Paris"

# Current association status
make db-status CITY="Paris"
```

### Complete Workflow
```bash
# Run entire spatial workflow for a city
make workflow-paris
```

## CLI Tools

### POI Association Tool
```bash
# Basic usage
python scripts/associate_pois.py --city "Paris" --operator covers

# Advanced options
python scripts/associate_pois.py \
    --city "Paris" \
    --operator covers \
    --dry-run \
    --verbose

# Validation and summary
python scripts/associate_pois.py --city "Paris" --validate-coords
python scripts/associate_pois.py --city "Paris" --summary-only
```

### Spatial QA Tool
```bash
# Full quality assessment
python scripts/spatial_qa.py --city "Paris" --full-report

# Quick checks only
python scripts/spatial_qa.py --city "Paris" --quick-check
```

## Quality Assurance

The spatial QA system provides comprehensive validation:

### Geometry Checks
- SRID validation (ensures 4326)
- Geometry type consistency (MultiPolygon)
- Validity verification (no self-intersections)
- Area size analysis (detect suspicious geometries)

### POI Association Checks  
- Coordinate range validation
- Lat/lng swap detection
- Coverage percentage analysis
- Unmatched POI identification

### Performance Metrics
- Spatial index verification
- Query execution statistics
- Association success rates

## Database Functions

### Core Functions
- `find_urban_areas_for_point(lat, lng, city)`: Find areas containing a point
- `link_pois_to_urban_areas(city)`: Associate all POIs using ST_Covers
- `link_pois_to_urban_areas_intersects(city)`: Fallback using ST_Intersects
- `validate_poi_coordinates(city)`: Detect coordinate issues

### Calling from Supabase Edge Functions
```javascript
const { data, error } = await supabase.rpc('link_pois_to_urban_areas', {
  p_city_name: 'Paris'
});
```

## Error Handling

### Common Issues & Solutions

**Invalid Geometries**:
```sql
-- Check invalid geometries
SELECT name, ST_IsValidReason(geometry) FROM urban_areas WHERE NOT ST_IsValid(geometry);

-- Fix with trigger re-processing
UPDATE urban_areas SET geometry_geojson = geometry_geojson WHERE id = 'problem-id';
```

**SRID Problems**:
```sql
-- Check SRID consistency
SELECT DISTINCT ST_SRID(geometry) FROM urban_areas;

-- All geometries should return 4326
```

**POI Association Issues**:
```bash
# Validate coordinates
make qa-coordinates CITY="Paris"

# Check for lat/lng swaps  
python scripts/associate_pois.py --city "Paris" --validate-coords

# Try alternative operator
make associate-pois-intersects CITY="Paris"
```

## Development

### Running Tests
```bash
make test-spatial
```

### Code Quality
```bash
make lint
make clean
```

### Monitoring
```bash
make logs
```

## Configuration

Set environment variables in `.env`:
```env
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
```

## Support

For issues with spatial operations:
1. Run `make qa-spatial` to identify problems
2. Check logs with `make logs` 
3. Validate configuration with CLI tools
4. Review geometry quality with SQL checks

## License

MIT License - See LICENSE file for details.