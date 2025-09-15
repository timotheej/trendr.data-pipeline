#!/usr/bin/env python3
"""
Script pour générer du SQL d'insertion des géométries depuis un fichier JSONL
Crée un fichier SQL à exécuter directement dans Supabase SQL Editor
Usage: python generate_geometry_sql.py pending_geometries_paris.jsonl
"""

import json
import sys
import os
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sql_from_jsonl(input_file: str, output_file: str):
    """Génère un fichier SQL depuis un JSONL"""
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Fichier non trouvé: {input_file}")
        
        logger.info(f"📁 Lecture: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        
        logger.info(f"🚀 Génération SQL pour {len(records)} géométries")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("-- Script d'insertion des géométries des quartiers\n")
            f.write("-- Généré automatiquement depuis un fichier JSONL\n")
            f.write("-- À exécuter dans Supabase SQL Editor\n\n")
            
            f.write("BEGIN;\n\n")
            
            for i, record in enumerate(records, 1):
                city_name = record['city_name']
                name = record['name']
                area_type = record['type']
                geometry = record['geometry']
                properties = record.get('properties', {})
                
                # Échapper les guillemets dans les noms
                name_escaped = name.replace("'", "''")
                city_name_escaped = city_name.replace("'", "''")
                
                # Convertir la géométrie en JSON string pour PostGIS
                geometry_json = json.dumps(geometry).replace("'", "''")
                
                admin_level = properties.get('admin_level', 'NULL')
                place_type = properties.get('place_type', 'NULL')
                
                if admin_level != 'NULL':
                    admin_level = f"'{admin_level}'"
                if place_type != 'NULL':
                    place_type = f"'{place_type}'"
                
                f.write(f"-- {i}/{len(records)}: {name}\n")
                f.write(f"""INSERT INTO urban_areas (city_name, name, type, admin_level, place_type, geometry, created_at)
VALUES (
    '{city_name_escaped}',
    '{name_escaped}',
    '{area_type}',
    {admin_level},
    {place_type},
    ST_Multi(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON('{geometry_json}'), 4326))),
    NOW()
)
ON CONFLICT (city_name, name, type)
DO UPDATE SET
    admin_level = EXCLUDED.admin_level,
    place_type = EXCLUDED.place_type,
    geometry = EXCLUDED.geometry,
    created_at = NOW();

""")
            
            f.write("COMMIT;\n\n")
            f.write(f"-- Fin du script - {len(records)} géométries traitées\n")
        
        logger.info(f"✅ Fichier SQL généré: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Erreur génération SQL: {e}")
        raise

def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
        print("Usage: python generate_geometry_sql.py <fichier.jsonl>")
        print("Exemple: python generate_geometry_sql.py pending_geometries_paris.jsonl")
        print("\n🔧 Génère un fichier SQL à exécuter dans Supabase")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.jsonl', '.sql')
    
    logger.info(f"🗺️  Gatto - Génération SQL Géométries")
    logger.info(f"📁 Source: {input_file}")
    logger.info(f"📄 Destination: {output_file}")
    
    try:
        generate_sql_from_jsonl(input_file, output_file)
        
        print("\n" + "="*60)
        print("📊 GÉNÉRATION SQL TERMINÉE")
        print("="*60)
        print(f"📁 Fichier généré: {output_file}")
        print("🔧 Prochaines étapes:")
        print("   1. Ouvrir Supabase SQL Editor")
        print(f"   2. Copier/coller le contenu de {output_file}")
        print("   3. Exécuter le script")
        print("="*60)
        print("🎉 Les géométries seront injectées en base!")
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()