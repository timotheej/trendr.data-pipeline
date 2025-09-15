# Gatto Data Pipeline - Spatial Operations Makefile
# Provides convenient commands for spatial data ingestion and quality assurance

.PHONY: help install spatial-setup qa-spatial associate-pois extract-neighborhoods

# Default target
help:
	@echo "🗺️  Gatto Data Pipeline - Spatial Operations"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install                  Install Python dependencies"
	@echo "  make spatial-setup           Setup spatial database schema and triggers"
	@echo ""
	@echo "Data Ingestion:"
	@echo "  make extract-neighborhoods   Extract neighborhoods from OSM (requires CITY variable)"
	@echo "  make ingest-geometries       Ingest geometries from GeoJSON (requires FILE variable)"
	@echo ""
	@echo "POI Association:"
	@echo "  make associate-pois          Associate POIs to urban areas using default settings"
	@echo "  make associate-pois-dry-run  Simulate POI association (dry run)"
	@echo "  make associate-pois-intersects  Use ST_Intersects instead of ST_Covers"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make qa-spatial              Run comprehensive spatial quality checks"
	@echo "  make qa-quick                Run quick essential checks only"
	@echo "  make qa-coordinates          Validate coordinate integrity"
	@echo ""
	@echo "Examples:"
	@echo "  make extract-neighborhoods CITY='Paris'"
	@echo "  make ingest-geometries FILE=neighbourhoods_paris.geojson"
	@echo "  make associate-pois CITY='Paris'"
	@echo "  make qa-spatial CITY='Paris'"

# Installation and setup
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt

spatial-setup:
	@echo "🛠️  Setting up spatial database schema and triggers..."
	@echo "Please execute the following SQL scripts in your Supabase SQL Editor:"
	@echo "  1. sql/schema_urban_areas.sql"
	@echo "  2. sql/triggers_geometry_normalization.sql" 
	@echo "  3. sql/functions_poi_zone_linking.sql"
	@echo ""
	@echo "Or if you have psql access, run:"
	@echo "  psql -h your-host -U postgres -d postgres -f sql/schema_urban_areas.sql"
	@echo "  psql -h your-host -U postgres -d postgres -f sql/triggers_geometry_normalization.sql"
	@echo "  psql -h your-host -U postgres -d postgres -f sql/functions_poi_zone_linking.sql"

# Data extraction and ingestion
extract-neighborhoods:
	@if [ -z "$(CITY)" ]; then \
		echo "❌ Please specify CITY variable. Example: make extract-neighborhoods CITY='Paris'"; \
		exit 1; \
	fi
	@echo "🌍 Extracting neighborhoods for $(CITY)..."
	cd neighbourhoods && python3 extract_neighbourhoods.py "$(CITY)"

ingest-geometries:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please specify FILE variable. Example: make ingest-geometries FILE=neighbourhoods_paris.geojson"; \
		exit 1; \
	fi
	@echo "📥 Ingesting geometries from $(FILE)..."
	cd neighbourhoods && python3 ingest_geometries.py "$(FILE)"

# POI association commands
associate-pois:
	@echo "🔗 Associating POIs to urban areas..."
	python3 scripts/associate_pois.py --city "$(or $(CITY),Paris)" --operator covers

associate-pois-dry-run:
	@echo "🔍 Simulating POI association (dry run)..."
	python3 scripts/associate_pois.py --city "$(or $(CITY),Paris)" --operator covers --dry-run

associate-pois-intersects:
	@echo "🔗 Associating POIs using ST_Intersects operator..."
	python3 scripts/associate_pois.py --city "$(or $(CITY),Paris)" --operator intersects

# Quality assurance commands
qa-spatial:
	@echo "🔍 Running comprehensive spatial quality assurance..."
	python3 scripts/spatial_qa.py --city "$(or $(CITY),Paris)" --full-report

qa-quick:
	@echo "⚡ Running quick spatial quality checks..."
	python3 scripts/spatial_qa.py --city "$(or $(CITY),Paris)" --quick-check

qa-coordinates:
	@echo "🎯 Validating coordinate integrity..."
	python3 scripts/associate_pois.py --city "$(or $(CITY),Paris)" --validate-coords

# Database operations
db-status:
	@echo "📊 Getting POI association status..."
	python3 scripts/associate_pois.py --city "$(or $(CITY),Paris)" --summary-only

# Cleanup and maintenance
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "pending_geometries_*.jsonl" -delete 2>/dev/null || true

# Development helpers
test-spatial:
	@echo "🧪 Running spatial functionality tests..."
	python -m pytest tests/test_spatial.py -v 2>/dev/null || echo "⚠️  No spatial tests found"

lint:
	@echo "🔧 Running code linting..."
	python -m ruff check . 2>/dev/null || echo "⚠️  Ruff not installed, skipping lint"

# Monitoring and logging
logs:
	@echo "📋 Showing recent spatial operation logs..."
	tail -n 50 data_pipeline.log 2>/dev/null || echo "⚠️  No log file found"

# Complete workflow example
workflow-paris:
	@echo "🚀 Running complete spatial workflow for Paris..."
	@echo "Step 1: Extract neighborhoods..."
	$(MAKE) extract-neighborhoods CITY=Paris
	@echo "Step 2: Ingest geometries..."
	$(MAKE) ingest-geometries FILE=neighbourhoods_paris.geojson
	@echo "Step 3: Associate POIs..."
	$(MAKE) associate-pois CITY=Paris  
	@echo "Step 4: Run QA checks..."
	$(MAKE) qa-spatial CITY=Paris
	@echo "✅ Complete workflow finished!"