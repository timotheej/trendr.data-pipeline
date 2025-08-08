# POI Photos Storage

This directory contains the downloaded and processed photos for POIs.

## Structure

```
photos/
├── README.md                    # This file
├── {poi_id}/                   # Directory for each POI
│   ├── 85_a1b2c3d4.jpg        # Quality score (85) + hash + extension
│   ├── 72_e5f6g7h8.jpg        # Lower quality photo
│   └── ...
└── collections/                # Collection covers (symlinks to best photos)
    ├── hot-new-spots-montreal.jpg
    └── ...
```

## Photo Naming Convention

Photos are named with the format: `{quality_score}_{hash}.jpg`

- **Quality Score**: 00-99, where 99 is the highest quality
- **Hash**: MD5 hash of the image data (first 8 characters)
- **Extension**: Always .jpg for consistency

## Quality Scoring

Photos are scored based on multiple factors:

1. **Resolution** (25%): Higher resolution is better
2. **Brightness** (20%): Optimal range is 80-180 (out of 255)
3. **Contrast** (20%): Good contrast improves visual appeal
4. **Saturation** (15%): Vibrant colors are preferred
5. **Composition** (10%): Aspect ratio preferences (16:9, 4:3, etc.)
6. **File Size** (10%): Reasonable file size (50KB-500KB optimal)

## API Usage

Photos are downloaded from Google Places Photo API:
- Maximum width: 1600px to balance quality and file size
- Rate limited: 1.5 seconds between requests
- Cached: Photos are stored locally to avoid re-downloading

## Management Commands

```bash
# Process photos for POIs without photos
python scripts/photo_processor.py --backfill --limit 50

# Refresh old photos
python scripts/photo_processor.py --refresh --days-old 30

# Analyze photo coverage
python scripts/photo_processor.py --analyze

# Test photo system
python scripts/photo_validation_test.py
```

## Integration with Collections

The best quality photo from each collection's POIs is automatically selected as the collection cover photo. This ensures visually appealing collection displays.

## Storage Considerations

- Photos are stored locally for fast access
- Old photos can be cleaned up with `photo_manager.cleanup_old_photos()`
- Consider cloud storage for production deployments
- Typical storage: ~100KB per photo, ~2-3 photos per POI

## Troubleshooting

**No photos downloaded:**
- Check Google Places API key configuration
- Verify POI has `google_place_id` field
- Some POIs may not have photos available

**Low quality scores:**
- Normal for some POIs (user-generated content varies)
- Algorithm prefers well-lit, high-contrast images
- Manual curation may be needed for premium collections