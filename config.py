"""
KISS Data Pipeline Configuration
Config-first approach with typed configuration objects
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class H3Config:
    """H3 cell configuration"""
    seed_res: int
    max_res: int
    scan_cap_per_cell: int
    subdivide_on_cap: bool

@dataclass
class PipelineConfig:
    """General pipeline configuration"""
    cities: List[str]
    poi_categories: List[str]
    daily_api_limit: int
    batch_size: int
    max_pois_per_category: int
    update_interval_days: int

@dataclass
class QualityConfig:
    """Quality gate configuration"""
    rating_min: float
    min_reviews: int
    enable_details_for_hold: bool

@dataclass
class RatingSnapshotConfig:
    """Rating snapshot configuration"""
    days_interval: int

@dataclass
class PhotoConfig:
    """Photo processing configuration"""
    max_width: int

@dataclass
class GoogleIngesterConfig:
    """Google ingester configuration"""
    quality: QualityConfig
    rating_snapshot: RatingSnapshotConfig
    photo: PhotoConfig
    category_map: Dict[str, str]
    subcategory_map: Dict[str, str]

@dataclass
class Config:
    """Main configuration object"""
    pipeline_config: PipelineConfig
    google_ingester: GoogleIngesterConfig
    h3: H3Config
    
    # Environment variables
    supabase_url: str = field(init=False)
    supabase_key: str = field(init=False)
    google_places_api_key: str = field(init=False)
    google_custom_search_api_key: str = field(init=False)
    google_custom_search_engine_id: str = field(init=False)
    openai_api_key: Optional[str] = field(init=False)
    anthropic_api_key: Optional[str] = field(init=False)
    photos_storage_path: str = field(init=False)

    def __post_init__(self):
        """Load environment variables after initialization"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.google_places_api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        self.google_custom_search_api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
        self.google_custom_search_engine_id = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.photos_storage_path = os.getenv('PHOTOS_STORAGE_PATH', os.path.expanduser('~/gatto_photos'))
        
        # Validate required environment variables
        required_env = ['SUPABASE_URL', 'SUPABASE_KEY', 'GOOGLE_PLACES_API_KEY']
        missing_env = [env for env in required_env if not getattr(self, env.lower())]
        if missing_env:
            raise ValueError(f"Missing required environment variables: {missing_env}")

# Global config instance
_config_instance: Optional[Config] = None

def load_config(config_path: str = "config.json") -> Config:
    """
    Load configuration from JSON file with typed objects
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Typed Config object
    """
    global _config_instance
    
    if _config_instance is not None:
        return _config_instance
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Extract sections from config
        pipeline_data = config_data['pipeline_config']
        google_data = config_data['google_ingester']
        h3_data = config_data.get('h3', {
            'seed_res': 9,
            'max_res': 11,
            'scan_cap_per_cell': 60,
            'subdivide_on_cap': True
        })
        
        # Build typed configuration objects
        pipeline_config = PipelineConfig(
            cities=pipeline_data['cities'],
            poi_categories=pipeline_data['poi_categories'],
            daily_api_limit=pipeline_data['daily_api_limit'],
            batch_size=pipeline_data['batch_size'],
            max_pois_per_category=pipeline_data['max_pois_per_category'],
            update_interval_days=pipeline_data['update_interval_days']
        )
        
        quality_config = QualityConfig(
            rating_min=google_data['quality']['rating_min'],
            min_reviews=google_data['quality']['min_reviews'],
            enable_details_for_hold=google_data['quality']['enable_details_for_hold']
        )
        
        rating_snapshot_config = RatingSnapshotConfig(
            days_interval=google_data['rating_snapshot']['days_interval']
        )
        
        photo_config = PhotoConfig(
            max_width=google_data['photo']['max_width']
        )
        
        google_ingester_config = GoogleIngesterConfig(
            quality=quality_config,
            rating_snapshot=rating_snapshot_config,
            photo=photo_config,
            category_map=google_data['category_map'],
            subcategory_map=google_data['subcategory_map']
        )
        
        h3_config = H3Config(
            seed_res=h3_data['seed_res'],
            max_res=h3_data['max_res'],
            scan_cap_per_cell=h3_data['scan_cap_per_cell'],
            subdivide_on_cap=h3_data['subdivide_on_cap']
        )
        
        _config_instance = Config(
            pipeline_config=pipeline_config,
            google_ingester=google_ingester_config,
            h3=h3_config
        )
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        return _config_instance
        
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")

def get_config() -> Config:
    """Get the global configuration instance"""
    if _config_instance is None:
        return load_config()
    return _config_instance

# Legacy compatibility - export environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
GOOGLE_PLACES_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY')
GOOGLE_CUSTOM_SEARCH_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
PHOTOS_STORAGE_PATH = os.getenv('PHOTOS_STORAGE_PATH', os.path.expanduser('~/gatto_photos'))