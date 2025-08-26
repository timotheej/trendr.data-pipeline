-- Trendr Database Schema Update
-- Add missing columns for POI photo management and Google Places integration

-- Update POI table with missing columns
ALTER TABLE poi 
ADD COLUMN IF NOT EXISTS google_place_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS primary_photo TEXT,
ADD COLUMN IF NOT EXISTS primary_photo_quality DECIMAL(4,3),
ADD COLUMN IF NOT EXISTS all_photos JSONB,
ADD COLUMN IF NOT EXISTS photos_updated_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS last_google_sync TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS business_status VARCHAR(50),
ADD COLUMN IF NOT EXISTS opening_hours JSONB,
ADD COLUMN IF NOT EXISTS is_open_now BOOLEAN,
ADD COLUMN IF NOT EXISTS photo_references JSONB,
ADD COLUMN IF NOT EXISTS google_types JSONB;

-- Add unique constraint on google_place_id
ALTER TABLE poi 
ADD CONSTRAINT poi_google_place_id_unique UNIQUE (google_place_id);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS poi_google_place_id_idx ON poi (google_place_id);
CREATE INDEX IF NOT EXISTS poi_primary_photo_idx ON poi (primary_photo);
CREATE INDEX IF NOT EXISTS poi_photos_updated_at_idx ON poi (photos_updated_at);
CREATE INDEX IF NOT EXISTS poi_last_google_sync_idx ON poi (last_google_sync);

-- Update collections table for cover photos
ALTER TABLE collections 
ADD COLUMN IF NOT EXISTS cover_photo TEXT;

-- Add index for collections cover photo
CREATE INDEX IF NOT EXISTS collections_cover_photo_idx ON collections (cover_photo);

-- Update existing POIs to have google_place_id if missing (set to null initially)
-- This will be populated by the pipeline during normal operation

-- Add comments for documentation
COMMENT ON COLUMN poi.google_place_id IS 'Google Places API place ID for photo and detail retrieval';
COMMENT ON COLUMN poi.primary_photo IS 'File path to the best quality photo for this POI';
COMMENT ON COLUMN poi.primary_photo_quality IS 'Quality score (0-1) of the primary photo';
COMMENT ON COLUMN poi.all_photos IS 'JSON array of all photos with metadata';
COMMENT ON COLUMN poi.photos_updated_at IS 'Timestamp when photos were last processed';
COMMENT ON COLUMN poi.last_google_sync IS 'Last sync with Google Places API';
COMMENT ON COLUMN collections.cover_photo IS 'File path to cover photo for this collection';