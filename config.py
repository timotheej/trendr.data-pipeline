"""
Trendr Data Pipeline Configuration
Environment variables and API key management
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Google APIs
GOOGLE_PLACES_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY')
GOOGLE_CUSTOM_SEARCH_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')

# AI APIs for Collection Generation (Optional)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Photo Storage Path (configurable)
PHOTOS_STORAGE_PATH = os.getenv('PHOTOS_STORAGE_PATH', os.path.expanduser('~/trendr_photos'))