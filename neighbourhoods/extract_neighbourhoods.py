#!/usr/bin/env python3
"""
Script pour extraire les quartiers/arrondissements d'une ville via Overpass API
Utilise OpenStreetMap Overpass API pour récupérer les données géospatiales
Usage: python extract_neighbourhoods.py "Paris"
"""

import requests
import json
import sys
from typing import List, Dict, Any

def query_overpass(query: str) -> Dict[str, Any]:
    """Exécute une requête Overpass API"""
    url = "https://overpass-api.de/api/interpreter"
    
    try:
        print(f"🌐 Envoi requête Overpass...")
        response = requests.post(url, data={"data": query}, timeout=90)
        response.raise_for_status()
        result = response.json()
        print(f"✅ Réponse reçue: {len(result.get('elements', []))} éléments")
        return result
    except requests.RequestException as e:
        print(f"❌ Erreur API Overpass: {e}")
        return {"elements": []}

def get_city_area_id(city_name: str) -> str:
    """Trouve l'ID de zone de la ville pour les requêtes suivantes"""
    query = f"""
    [out:json][timeout:30];
    (
      relation["name"="{city_name}"]["boundary"="administrative"]["admin_level"~"^[2-8]$"];
    );
    out ids;
    """
    
    print(f"🔍 Recherche de l'ID de zone pour {city_name}...")
    result = query_overpass(query)
    
    if result.get("elements"):
        area_id = 3600000000 + result["elements"][0]["id"]  # Convention Overpass
        print(f"✅ ID de zone trouvé: {area_id}")
        return str(area_id)
    
    print(f"⚠️  Aucun ID de zone trouvé pour {city_name}")
    return None

def get_neighbourhoods_by_area(area_id: str) -> List[Dict[str, Any]]:
    """Récupère les quartiers dans une zone donnée"""
    
    # Étape 1: Quartiers administratifs
    query_admin = f"""
    [out:json][timeout:60];
    (
      relation["boundary"="administrative"]["admin_level"~"^(9|10)$"](area:{area_id});
    );
    out geom;
    """
    
    print(f"🏛️  Recherche des quartiers administratifs...")
    admin_result = query_overpass(query_admin)
    
    neighbourhoods = []
    
    # Traiter les résultats admin
    for element in admin_result.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("alt_name")
        if name:
            neighbourhoods.append({
                "name": name,
                "type": "admin",
                "admin_level": tags.get("admin_level"),
                "element": element
            })
    
    print(f"📍 {len(neighbourhoods)} quartiers administratifs trouvés")
    
    # Étape 2: Si peu de résultats, ajouter les places
    if len(neighbourhoods) < 5:
        print(f"🏘️  Ajout des places (neighbourhoods, quarters, suburbs)...")
        
        query_places = f"""
        [out:json][timeout:60];
        (
          node["place"~"^(neighbourhood|quarter|suburb)$"](area:{area_id});
          way["place"~"^(neighbourhood|quarter|suburb)$"](area:{area_id});
          relation["place"~"^(neighbourhood|quarter|suburb)$"](area:{area_id});
        );
        out geom;
        """
        
        places_result = query_overpass(query_places)
        
        for element in places_result.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name") or tags.get("alt_name")
            if name:
                neighbourhoods.append({
                    "name": name,
                    "type": "place",
                    "place_type": tags.get("place"),
                    "element": element
                })
        
        place_count = len(neighbourhoods) - len(admin_result.get("elements", []))
        print(f"📍 {place_count} places ajoutées")
    
    return neighbourhoods

def get_neighbourhoods_direct(city_name: str) -> List[Dict[str, Any]]:
    """Méthode directe si la recherche par area_id échoue"""
    
    query = f"""
    [out:json][timeout:90];
    (
      // Quartiers administratifs autour de la ville
      relation["boundary"="administrative"]["admin_level"~"^(9|10)$"](around:50000)[name~"{city_name}",i];
      
      // Places autour de la ville  
      node["place"~"^(neighbourhood|quarter|suburb)$"](around:30000)[name~"{city_name}",i];
      way["place"~"^(neighbourhood|quarter|suburb)$"](around:30000)[name~"{city_name}",i];
      relation["place"~"^(neighbourhood|quarter|suburb)$"](around:30000)[name~"{city_name}",i];
    );
    out geom;
    """
    
    print(f"🎯 Recherche directe autour de {city_name}...")
    result = query_overpass(query)
    
    neighbourhoods = []
    
    for element in result.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("alt_name")
        
        if not name:
            continue
        
        # Filtrer pour garder seulement ceux liés à la ville
        if city_name.lower() not in name.lower() and not any(
            city_name.lower() in str(v).lower() for v in tags.values()
        ):
            continue
        
        if tags.get("boundary") == "administrative":
            neighbourhood_type = "admin"
        elif "place" in tags:
            neighbourhood_type = "place"
        else:
            continue
        
        neighbourhoods.append({
            "name": name,
            "type": neighbourhood_type,
            "element": element
        })
    
    print(f"📍 {len(neighbourhoods)} quartiers trouvés (recherche directe)")
    return neighbourhoods

def are_points_equal(p1: List[float], p2: List[float], tolerance: float = 1e-6) -> bool:
    """Vérifie si deux points sont égaux avec une tolérance"""
    return (abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance)

def assemble_single_ring(segments: List[List[List[float]]]) -> List[List[float]]:
    """
    Assemble des segments en un seul ring fermé valide.
    Retourne None si impossible d'assembler.
    """
    if not segments:
        return None
    
    # Commencer avec le premier segment
    ring = segments[0].copy()
    used_segments = {0}
    
    # Essayer de connecter les autres segments de manière séquentielle
    while len(used_segments) < len(segments):
        found_connection = False
        current_end = ring[-1]
        
        for i, segment in enumerate(segments):
            if i in used_segments:
                continue
                
            segment_start = segment[0]
            segment_end = segment[-1]
            
            # Vérifier si on peut connecter ce segment
            if are_points_equal(current_end, segment_start):
                # Connexion directe: ajouter segment (sans dupliquer le point de connexion)
                ring.extend(segment[1:])
                used_segments.add(i)
                found_connection = True
                break
            elif are_points_equal(current_end, segment_end):
                # Connexion inversée: ajouter segment inversé
                reversed_segment = segment[:-1]  # Enlever le dernier point
                reversed_segment.reverse()
                ring.extend(reversed_segment)
                used_segments.add(i)
                found_connection = True
                break
        
        if not found_connection:
            # Impossible de connecter plus de segments
            break
    
    # Vérifier si on a utilisé tous les segments
    if len(used_segments) != len(segments):
        return None
    
    # Fermer le ring si nécessaire
    if len(ring) >= 3:
        if not are_points_equal(ring[0], ring[-1]):
            ring.append(ring[0])
        
        # Validation basique: au moins 4 points (triangle fermé)
        if len(ring) >= 4:
            return ring
    
    return None

def element_to_geojson_geometry(element: Dict[str, Any]) -> Dict[str, Any]:
    """Convertit un élément OSM en géométrie GeoJSON"""
    try:
        if element["type"] == "node":
            return {
                "type": "Point",
                "coordinates": [element["lon"], element["lat"]]
            }
        
        elif element["type"] == "way" and "geometry" in element:
            coords = [[node["lon"], node["lat"]] for node in element["geometry"]]
            if len(coords) >= 3:
                # Fermer le polygone
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                return {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
        
        elif element["type"] == "relation" and "members" in element:
            # Récupérer seulement les outer ways avec géométrie
            outer_members = [
                member for member in element["members"]
                if (member.get("type") == "way" and 
                    member.get("role") in ["outer", ""] and 
                    "geometry" in member and 
                    len(member["geometry"]) > 0)
            ]
            
            if not outer_members:
                return None
            
            print(f"🔗 Traitement de {len(outer_members)} segments pour {element.get('tags', {}).get('name', 'inconnu')}")
            
            # Cas simple: un seul way déjà fermé
            if len(outer_members) == 1:
                coords = [[node["lon"], node["lat"]] for node in outer_members[0]["geometry"]]
                if len(coords) >= 4 and are_points_equal(coords[0], coords[-1]):
                    print(f"  ✅ Way simple fermé avec {len(coords)} points")
                    return {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
            
            # Cas complexe: essayer d'assembler les segments
            segments = []
            for member in outer_members:
                coords = [[node["lon"], node["lat"]] for node in member["geometry"]]
                if len(coords) >= 2:
                    segments.append(coords)
            
            if not segments:
                return None
            
            # Essayer de créer un seul ring connecté
            assembled_ring = assemble_single_ring(segments)
            if assembled_ring:
                print(f"  ✅ Ring assemblé avec {len(assembled_ring)} points")
                return {
                    "type": "Polygon",
                    "coordinates": [assembled_ring]
                }
            else:
                print(f"  ❌ Impossible d'assembler un ring valide")
                return None
    
    except Exception as e:
        print(f"⚠️  Erreur géométrie: {e}")
    
    return None

def create_geojson(neighbourhoods: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Crée un GeoJSON des quartiers"""
    features = []
    
    for neighbourhood in neighbourhoods:
        geometry = element_to_geojson_geometry(neighbourhood["element"])
        
        if geometry:
            features.append({
                "type": "Feature",
                "properties": {
                    "name": neighbourhood["name"],
                    "type": neighbourhood["type"],
                    **{k: v for k, v in neighbourhood.items() if k not in ["name", "type", "element"]}
                },
                "geometry": geometry
            })
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
        print("Usage: python extract_neighbourhoods.py \"Nom de la ville\"")
        print("Exemple: python extract_neighbourhoods.py \"Paris\"")
        sys.exit(1)
    
    city_name = sys.argv[1]
    print(f"🌍 Extraction des quartiers de {city_name}")
    print("=" * 50)
    
    neighbourhoods = []
    
    # Méthode 1: Recherche par area_id
    area_id = get_city_area_id(city_name)
    if area_id:
        neighbourhoods = get_neighbourhoods_by_area(area_id)
    
    # Méthode 2: Recherche directe si échec
    if not neighbourhoods:
        print("🔄 Tentative de recherche directe...")
        neighbourhoods = get_neighbourhoods_direct(city_name)
    
    if not neighbourhoods:
        print(f"❌ Aucun quartier trouvé pour {city_name}")
        print("💡 Essayez un nom alternatif ou vérifiez l'orthographe")
        sys.exit(1)
    
    # Créer le GeoJSON
    print(f"\n🗺️  Création du GeoJSON...")
    geojson_data = create_geojson(neighbourhoods)
    
    # Sauvegarder
    filename = f"neighbourhoods_{city_name.lower().replace(' ', '_')}.geojson"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)
    
    # Résumé
    print(f"\n📊 Résultats:")
    print(f"   • Quartiers trouvés: {len(neighbourhoods)}")
    print(f"   • Avec géométrie: {len(geojson_data['features'])}")
    print(f"   • Fichier: {filename}")
    
    # Types
    admin_count = sum(1 for n in neighbourhoods if n["type"] == "admin")
    place_count = sum(1 for n in neighbourhoods if n["type"] == "place")
    print(f"   • Admin: {admin_count}, Places: {place_count}")
    
    # Aperçu
    if neighbourhoods:
        print(f"\n📝 Quartiers trouvés:")
        for i, n in enumerate(neighbourhoods[:15], 1):
            icon = "🏛️" if n["type"] == "admin" else "🏘️"
            print(f"   {i:2}. {icon} {n['name']}")
        
        if len(neighbourhoods) > 15:
            print(f"   ... et {len(neighbourhoods) - 15} autres")
    
    print("\n✅ Extraction terminée!")

if __name__ == "__main__":
    main()