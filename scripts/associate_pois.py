#!/usr/bin/env python3
"""
Minimal POI spatial association - KISS version
Only contains what's needed for run_pipeline.py
"""

import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def update_poi_association(poi_id: str) -> Dict[str, Any]:
    """Update spatial association for a single POI - KISS approach using existing function"""
    try:
        from supabase import create_client
        client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        
        # Get POI info before update
        poi_before = client.table('poi').select('name, district_name, neighbourhood_name').eq('id', poi_id).execute()
        if not poi_before.data:
            return {'success': False, 'error': 'POI not found'}
        
        # Use existing function (yes it updates all, but it's simple and works)
        result = client.rpc('update_all_paris_pois').execute()
        
        # Get POI info after update to see what changed
        poi_after = client.table('poi').select('name, district_name, neighbourhood_name').eq('id', poi_id).execute()
        
        if poi_after.data:
            poi_data = poi_after.data[0]
            return {
                'success': True,
                'district_name': poi_data.get('district_name'),
                'neighbourhood_name': poi_data.get('neighbourhood_name')
            }
        else:
            return {'success': False, 'error': 'POI not found after update'}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}