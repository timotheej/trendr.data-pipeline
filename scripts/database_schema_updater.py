#!/usr/bin/env python3
"""
Database Schema Updater
Updates Supabase database schema to support photo management and latest features
"""
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import SupabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSchemaUpdater:
    """Updates database schema for new features"""
    
    def __init__(self):
        self.db = SupabaseManager()
    
    def check_missing_columns(self) -> dict:
        """Check which columns are missing from the database"""
        missing_columns = {}
        
        try:
            # Test POI table columns by trying to select them
            poi_columns_to_check = [
                'google_place_id',
                'primary_photo', 
                'primary_photo_quality',
                'all_photos',
                'photos_updated_at',
                'last_google_sync',
                'business_status',
                'opening_hours',
                'is_open_now',
                'photo_references',
                'google_types'
            ]
            
            missing_columns['poi'] = []
            
            for column in poi_columns_to_check:
                try:
                    # Try to select this column
                    result = self.db.client.table('poi')\
                        .select(column)\
                        .limit(1)\
                        .execute()
                    logger.info(f"✅ Column 'poi.{column}' exists")
                except Exception as e:
                    if "could not find" in str(e).lower() or "pgrst204" in str(e).lower():
                        missing_columns['poi'].append(column)
                        logger.warning(f"❌ Column 'poi.{column}' missing")
                    else:
                        logger.error(f"Error checking column poi.{column}: {e}")
            
            # Check collections table
            collections_columns_to_check = ['cover_photo']
            missing_columns['collections'] = []
            
            for column in collections_columns_to_check:
                try:
                    result = self.db.client.table('collections')\
                        .select(column)\
                        .limit(1)\
                        .execute()
                    logger.info(f"✅ Column 'collections.{column}' exists")
                except Exception as e:
                    if "could not find" in str(e).lower() or "pgrst204" in str(e).lower():
                        missing_columns['collections'].append(column)
                        logger.warning(f"❌ Column 'collections.{column}' missing")
                    else:
                        logger.error(f"Error checking column collections.{column}: {e}")
            
            return missing_columns
            
        except Exception as e:
            logger.error(f"Error checking database schema: {e}")
            return {"error": str(e)}
    
    def get_schema_update_sql(self) -> str:
        """Get the SQL commands to update the schema"""
        schema_file = Path(__file__).parent.parent / "database" / "schema_update.sql"
        
        if not schema_file.exists():
            logger.error(f"Schema update file not found: {schema_file}")
            return ""
        
        try:
            with open(schema_file, 'r') as f:
                sql_content = f.read()
            return sql_content
        except Exception as e:
            logger.error(f"Error reading schema file: {e}")
            return ""
    
    def apply_schema_update_manually(self) -> bool:
        """Show instructions for manual schema update"""
        sql_content = self.get_schema_update_sql()
        
        if not sql_content:
            return False
        
        print("\n" + "="*60)
        print("🔧 DATABASE SCHEMA UPDATE REQUIRED")
        print("="*60)
        print()
        print("The database is missing required columns for the photo management system.")
        print("Please apply the following SQL update in your Supabase SQL Editor:")
        print()
        print("1. Go to your Supabase dashboard")
        print("2. Navigate to SQL Editor")
        print("3. Copy and paste this SQL:")
        print()
        print("-" * 60)
        print(sql_content)
        print("-" * 60)
        print()
        print("4. Click 'Run' to execute the SQL")
        print("5. Verify the update completed successfully")
        print("6. Re-run the pipeline test")
        print()
        print("This will add the required columns for:")
        print("- ✅ Google Places integration (google_place_id)")
        print("- ✅ Photo management (primary_photo, all_photos, etc.)")
        print("- ✅ Collection covers (cover_photo)")
        print("- ✅ Enhanced POI metadata")
        print()
        
        return True
    
    def verify_schema_update(self) -> bool:
        """Verify that schema update was applied successfully"""
        logger.info("🔍 Verifying database schema update...")
        
        missing_columns = self.check_missing_columns()
        
        if 'error' in missing_columns:
            logger.error(f"Error verifying schema: {missing_columns['error']}")
            return False
        
        total_missing = sum(len(cols) for cols in missing_columns.values())
        
        if total_missing == 0:
            logger.info("✅ All required columns are present!")
            print("\n🎉 DATABASE SCHEMA UPDATE SUCCESSFUL!")
            print("The pipeline is now ready to run with full photo support.")
            return True
        else:
            logger.warning(f"❌ Still missing {total_missing} columns")
            for table, columns in missing_columns.items():
                if columns:
                    logger.warning(f"Missing in {table}: {', '.join(columns)}")
            return False
    
    def run_schema_check_and_guide(self):
        """Main method to check schema and provide guidance"""
        print("🔍 Checking database schema compatibility...")
        
        missing_columns = self.check_missing_columns()
        
        if 'error' in missing_columns:
            print(f"❌ Error checking database: {missing_columns['error']}")
            return False
        
        total_missing = sum(len(cols) for cols in missing_columns.values())
        
        if total_missing == 0:
            print("✅ Database schema is up to date!")
            print("All required columns are present for photo management.")
            return True
        else:
            print(f"\n⚠️ Found {total_missing} missing columns:")
            for table, columns in missing_columns.items():
                if columns:
                    print(f"  {table}: {', '.join(columns)}")
            
            print("\n📋 Schema update required...")
            return self.apply_schema_update_manually()

def main():
    """CLI interface for schema updates"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Schema Updater')
    parser.add_argument('--check', action='store_true', help='Check current schema')
    parser.add_argument('--update-guide', action='store_true', help='Show schema update instructions')
    parser.add_argument('--verify', action='store_true', help='Verify schema update was applied')
    
    args = parser.parse_args()
    
    updater = DatabaseSchemaUpdater()
    
    if args.verify:
        success = updater.verify_schema_update()
        return 0 if success else 1
    elif args.check:
        missing = updater.check_missing_columns()
        total_missing = sum(len(cols) for cols in missing.values()) if 'error' not in missing else -1
        print(f"Missing columns: {total_missing}")
        return 0 if total_missing == 0 else 1
    elif args.update_guide:
        updater.apply_schema_update_manually()
        return 0
    else:
        # Default behavior
        success = updater.run_schema_check_and_guide()
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())