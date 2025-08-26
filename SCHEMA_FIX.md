# 🔧 Database Schema Fix Guide

## Problem
Pipeline failing with error: `"Could not find the 'google_place_id' column of 'poi' in the schema cache"`

## Quick Fix (2 minutes)

### Step 1: Check What's Missing
```bash
python scripts/database_schema_updater.py --check
```

### Step 2: Get Schema Update SQL
```bash
python scripts/database_schema_updater.py --update-guide
```

### Step 3: Apply in Supabase
1. Go to your Supabase dashboard
2. Navigate to **SQL Editor**
3. Copy the SQL from the output above
4. Paste and click **Run**

### Step 4: Verify Fix
```bash
python scripts/database_schema_updater.py --verify
```

### Step 5: Test Pipeline
```bash
python run_pipeline.py --dry-run
```

## What the Schema Update Adds

### POI Table Columns:
- `google_place_id` - Google Places API ID
- `primary_photo` - Best photo file path  
- `primary_photo_quality` - Photo quality score (0-1)
- `all_photos` - JSON array of all photos
- `photos_updated_at` - Photo processing timestamp
- `last_google_sync` - Last Google API sync
- `business_status` - Open/closed status
- `opening_hours` - Operating hours JSON
- `is_open_now` - Current status boolean
- `photo_references` - Google photo references
- `google_types` - Google place types

### Collections Table:
- `cover_photo` - Collection cover image path

## Alternative: Manual SQL

If the script doesn't work, copy this SQL directly to Supabase:

```sql
-- Add missing columns to poi table
ALTER TABLE poi 
ADD COLUMN IF NOT EXISTS google_place_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS primary_photo TEXT,
ADD COLUMN IF NOT EXISTS primary_photo_quality DECIMAL(4,3),
ADD COLUMN IF NOT EXISTS all_photos JSONB,
ADD COLUMN IF NOT EXISTS photos_updated_at TIMESTAMP WITH TIME ZONE;

-- Add unique constraint
ALTER TABLE poi 
ADD CONSTRAINT poi_google_place_id_unique UNIQUE (google_place_id);

-- Add cover photo to collections
ALTER TABLE collections 
ADD COLUMN IF NOT EXISTS cover_photo TEXT;
```

## Verification

After applying the schema update, you should see:
```
✅ Column 'poi.google_place_id' exists
✅ Column 'poi.primary_photo' exists  
✅ Column 'collections.cover_photo' exists
🎉 DATABASE SCHEMA UPDATE SUCCESSFUL!
```

The pipeline will then work correctly with photo management!