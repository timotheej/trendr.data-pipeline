#!/usr/bin/env python3
"""
Script d'ingestion des quartiers dans la base Gatto
Utilise l'infrastructure existante (SupabaseManager + config)
Usage: python ingest_neighbourhoods.py neighbourhoods_paris.geojson "Paris"
"""

import json
import sys
import os
import logging
from typing import Dict, Any, List

# Ajouter le répertoire parent au path pour importer nos modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager
import config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UrbanAreasIngester:
    """Ingestion des zones urbaines dans la base Gatto"""
    
    def __init__(self):
        """Initialise avec notre infrastructure existante"""
        logger.info("🔌 Connexion à la base Gatto...")
        self.db = SupabaseManager()
        
        # Vérifier que la table urban_areas existe
        try:
            self.db.client.table('urban_areas').select('id').limit(1).execute()
            logger.info("✅ Table urban_areas accessible")
        except Exception as e:
            logger.error(f"❌ Erreur accès table urban_areas: {e}")
            logger.error("💡 Assurez-vous que la table urban_areas existe en base")
            raise
        
        # Statistiques
        self.stats = {
            'total_features': 0,
            'processed': 0,
            'errors': 0,
            'skipped': 0
        }
    
    def load_geojson(self, file_path: str) -> Dict[str, Any]:
        """Charge et valide le fichier GeoJSON"""
        try:
            logger.info(f"📁 Lecture: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('type') != 'FeatureCollection':
                raise ValueError("Fichier GeoJSON invalide (pas un FeatureCollection)")
            
            features = data.get('features', [])
            logger.info(f"✅ {len(features)} quartiers trouvés")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur lecture GeoJSON: {e}")
            raise
    
    def extract_metadata(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les métadonnées d'un quartier sans géométrie"""
        properties = feature.get('properties', {})
        
        return {
            'name': properties.get('name'),
            'type': properties.get('type', 'place'),
            'admin_level': properties.get('admin_level'),
            'place_type': properties.get('place_type')
        }
    
    def ingest_neighbourhood_metadata(self, city_name: str, feature: Dict[str, Any]) -> str:
        """Ingère les métadonnées d'un quartier (sans géométrie pour l'instant)"""
        try:
            metadata = self.extract_metadata(feature)
            name = metadata.get('name')
            neighbourhood_type = metadata.get('type', 'place')
            
            if not name:
                logger.warning(f"Quartier sans nom ignoré")
                self.stats['skipped'] += 1
                return 'skipped'
            
            # Valider le type
            if neighbourhood_type not in ['admin', 'place']:
                logger.warning(f"Type '{neighbourhood_type}' invalide pour {name}, forcé à 'place'")
                neighbourhood_type = 'place'
            
            # Vérifier si le quartier existe déjà
            existing = self.db.client.table('urban_areas')\
                .select('id')\
                .eq('city_name', city_name)\
                .eq('name', name)\
                .eq('type', neighbourhood_type)\
                .execute()
            
            if existing.data:
                # Quartier existe déjà - on pourrait mettre à jour les métadonnées
                logger.debug(f"🔄 Quartier existant: {name}")
                self.stats['processed'] += 1
                return 'exists'
            else:
                # Nouveau quartier - on ajoute les métadonnées seulement
                # La géométrie sera ajoutée plus tard via un script SQL séparé
                logger.info(f"📝 Nouveau quartier détecté: {name} ({neighbourhood_type})")
                logger.info(f"   ⚠️  Géométrie à ajouter manuellement via SQL")
                
                # Stocker dans un fichier de référence pour traitement ultérieur
                self._log_for_geometry_processing(city_name, name, neighbourhood_type, feature)
                
                self.stats['processed'] += 1
                return 'metadata_ready'
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Erreur traitement '{metadata.get('name', 'unknown')}': {e}")
            return 'error'
    
    def _log_for_geometry_processing(self, city_name: str, name: str, type_: str, feature: Dict[str, Any]):
        """Log les quartiers pour traitement géométrique ultérieur"""
        log_file = f"pending_geometries_{city_name.lower().replace(' ', '_')}.jsonl"
        
        log_entry = {
            'city_name': city_name,
            'name': name,
            'type': type_,
            'geometry': feature.get('geometry'),
            'properties': feature.get('properties')
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def ingest_geojson(self, file_path: str, city_name: str):
        """Traite un fichier GeoJSON complet"""
        try:
            # Chargement
            geojson_data = self.load_geojson(file_path)
            features = geojson_data['features']
            self.stats['total_features'] = len(features)
            
            logger.info(f"🚀 Traitement: {len(features)} quartiers pour {city_name}")
            logger.warning("📝 Version métadonnées seulement - géométries traitées séparément")
            
            # Traitement feature par feature
            for i, feature in enumerate(features, 1):
                try:
                    result = self.ingest_neighbourhood_metadata(city_name, feature)
                    
                    # Log de progression
                    if i % 10 == 0 or i == len(features):
                        logger.info(f"📊 Progression: {i}/{len(features)}")
                
                except Exception as e:
                    logger.error(f"❌ Erreur feature {i}: {e}")
                    continue
            
            logger.info("✅ Traitement métadonnées terminé")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement: {e}")
            raise
    
    def print_summary(self, city_name: str):
        """Affiche le résumé du traitement"""
        total = self.stats['total_features']
        processed = self.stats['processed']
        errors = self.stats['errors']
        skipped = self.stats['skipped']
        
        print("\n" + "="*60)
        print(f"📊 RÉSUMÉ - ZONES URBAINES {city_name.upper()}")
        print("="*60)
        print(f"📁 Features analysées:    {total}")
        print(f"✅ Traitées:             {processed}")
        print(f"⏭️  Ignorées:             {skipped}")
        print(f"❌ Erreurs:              {errors}")
        print("="*60)
        
        if processed > 0:
            log_file = f"pending_geometries_{city_name.lower().replace(' ', '_')}.jsonl"
            if os.path.exists(log_file):
                print(f"📄 Fichier créé: {log_file}")
                print("🔧 Prochaines étapes:")
                print("   1. Utiliser ce fichier pour créer un script SQL")
                print("   2. Insérer les géométries via PostGIS direct")
                print("   3. Ou utiliser un script PostgreSQL/psycopg2 dédié")
        
        if total > 0:
            print("\n🎉 Analyse terminée!")
            print("💡 Les métadonnées sont prêtes pour l'ingestion géométrique")

def validate_config():
    """Valide la configuration du projet"""
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        raise EnvironmentError(
            "❌ Configuration Supabase manquante\n"
            "Vérifiez vos variables SUPABASE_URL et SUPABASE_KEY dans .env"
        )
    logger.info("✅ Configuration Gatto validée")

def main():
    """Fonction principale"""
    if len(sys.argv) != 3:
        print("Usage: python ingest_neighbourhoods.py <fichier.geojson> <nom_ville>")
        print("Exemple: python ingest_neighbourhoods.py neighbourhoods_paris.geojson \"Paris\"")
        print("\n🏗️  Script intégré au projet Gatto")
        print("📝 Traite les métadonnées - géométries via script séparé")
        sys.exit(1)
    
    file_path = sys.argv[1]
    city_name = sys.argv[2]
    
    logger.info(f"🌍 Gatto - Zones Urbaines: {city_name}")
    logger.info(f"📁 Source: {file_path}")
    
    try:
        # Validation
        validate_config()
        
        # Traitement
        ingester = UrbanAreasIngester()
        ingester.ingest_geojson(file_path, city_name)
        ingester.print_summary(city_name)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Traitement interrompu")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()