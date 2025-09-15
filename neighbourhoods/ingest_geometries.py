#!/usr/bin/env python3
"""
Script pour injecter les géométries des quartiers depuis un fichier JSONL
Utilise une connexion PostgreSQL directe pour contourner les limitations de l'API REST Supabase
Usage: python ingest_geometries.py pending_geometries_paris.jsonl
"""

import json
import sys
import os
import logging
import psycopg2
from typing import Dict, Any
from urllib.parse import urlparse

# Ajouter le répertoire parent au path pour importer nos modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeometryIngester:
    """Ingestion des géométries via connexion PostgreSQL directe"""
    
    def __init__(self):
        """Initialise avec la connexion PostgreSQL directe"""
        logger.info("🔌 Connexion directe à PostgreSQL...")
        
        # Parser l'URL Supabase pour extraire les paramètres de connexion
        url = urlparse(config.SUPABASE_URL)
        
        # Construire la string de connexion PostgreSQL
        # Supabase utilise le port 5432 et require SSL
        self.conn_params = {
            'host': url.hostname,
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': config.SUPABASE_KEY.split('.')[-1] if config.SUPABASE_KEY else None,
            'sslmode': 'require'
        }
        
        # Note: Pour Supabase, le mot de passe est souvent le service_role_key
        # Si ça ne marche pas, il faut utiliser les vraies credentials PostgreSQL
        logger.warning("⚠️  Utilisation des credentials Supabase - ajustez si nécessaire")
        
        self.conn = None
        self.stats = {
            'processed': 0,
            'errors': 0,
            'skipped': 0
        }
    
    def connect(self):
        """Établit la connexion PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            logger.info("✅ Connexion PostgreSQL établie")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur connexion PostgreSQL: {e}")
            logger.error("💡 Vérifiez vos credentials PostgreSQL dans .env")
            logger.error("💡 Pour Supabase, utilisez les vraies credentials DB, pas l'API key")
            return False
    
    def ingest_geometry(self, record: Dict[str, Any]) -> bool:
        """Ingère une géométrie dans la table urban_areas"""
        try:
            cursor = self.conn.cursor()
            
            city_name = record['city_name']
            name = record['name']
            area_type = record['type']
            geometry = record['geometry']
            properties = record.get('properties', {})
            
            # Convertir la géométrie GeoJSON en format WKT pour PostGIS
            geometry_json = json.dumps(geometry)
            
            # Requête SQL pour insérer/mettre à jour avec géométrie
            sql = """
            INSERT INTO urban_areas (city_name, name, type, admin_level, place_type, geometry, created_at)
            VALUES (%s, %s, %s, %s, %s, ST_Multi(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))), NOW())
            ON CONFLICT (city_name, name, type)
            DO UPDATE SET
                admin_level = EXCLUDED.admin_level,
                place_type = EXCLUDED.place_type,
                geometry = EXCLUDED.geometry,
                created_at = NOW()
            """
            
            cursor.execute(sql, (
                city_name,
                name,
                area_type,
                properties.get('admin_level'),
                properties.get('place_type'),
                geometry_json
            ))
            
            self.conn.commit()
            cursor.close()
            
            logger.debug(f"✅ Géométrie insérée: {name}")
            self.stats['processed'] += 1
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Erreur ingestion '{name}': {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def ingest_jsonl_file(self, file_path: str):
        """Traite un fichier JSONL complet"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
            
            logger.info(f"📁 Lecture: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f if line.strip()]
            
            logger.info(f"🚀 Traitement: {len(records)} géométries")
            
            # Traitement record par record
            for i, record in enumerate(records, 1):
                try:
                    self.ingest_geometry(record)
                    
                    # Log de progression
                    if i % 10 == 0 or i == len(records):
                        logger.info(f"📊 Progression: {i}/{len(records)}")
                
                except Exception as e:
                    logger.error(f"❌ Erreur record {i}: {e}")
                    continue
            
            logger.info("✅ Traitement des géométries terminé")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement fichier: {e}")
            raise
    
    def print_summary(self, file_path: str):
        """Affiche le résumé du traitement"""
        processed = self.stats['processed']
        errors = self.stats['errors']
        skipped = self.stats['skipped']
        total = processed + errors + skipped
        
        print("\n" + "="*60)
        print(f"📊 RÉSUMÉ - INGESTION GÉOMÉTRIES")
        print("="*60)
        print(f"📁 Fichier traité:       {os.path.basename(file_path)}")
        print(f"🔢 Records total:        {total}")
        print(f"✅ Géométries insérées: {processed}")
        print(f"⏭️  Ignorées:             {skipped}")
        print(f"❌ Erreurs:              {errors}")
        print("="*60)
        
        if total > 0:
            success_rate = (processed / total) * 100
            print(f"📈 Taux de succès:       {success_rate:.1f}%")
        
        if processed > 0:
            print("🎉 Géométries injectées avec succès!")
            print("💡 Vous pouvez maintenant utiliser la table urban_areas")
    
    def close(self):
        """Ferme la connexion"""
        if self.conn:
            self.conn.close()
            logger.info("🔐 Connexion fermée")

def validate_config():
    """Valide la configuration du projet"""
    if not config.SUPABASE_URL:
        raise EnvironmentError(
            "❌ Configuration Supabase manquante\n"
            "Vérifiez votre variable SUPABASE_URL dans .env"
        )
    logger.info("✅ Configuration validée")

def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
        print("Usage: python ingest_geometries.py <fichier.jsonl>")
        print("Exemple: python ingest_geometries.py pending_geometries_paris.jsonl")
        print("\n🔧 Script PostgreSQL direct pour contourner les limites API REST")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    logger.info(f"🗺️  Gatto - Ingestion Géométries")
    logger.info(f"📁 Source: {file_path}")
    
    ingester = None
    
    try:
        # Validation
        validate_config()
        
        # Traitement
        ingester = GeometryIngester()
        if not ingester.connect():
            sys.exit(1)
        
        ingester.ingest_jsonl_file(file_path)
        ingester.print_summary(file_path)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Traitement interrompu")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)
    
    finally:
        if ingester:
            ingester.close()

if __name__ == "__main__":
    main()