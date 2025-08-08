#!/usr/bin/env python3
"""
Photo System Validation Test
Quick test to validate the photo management system works correctly
"""
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.photo_manager import POIPhotoManager
from utils.database import SupabaseManager

def test_photo_system():
    """Test the photo management system"""
    print("🧪 Testing POI Photo Management System")
    print("=" * 50)
    
    # Initialize components
    try:
        photo_manager = POIPhotoManager("/tmp/trendr_photos_test")
        db = SupabaseManager()
        print("✅ Photo manager initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test 1: Get a sample POI
    try:
        pois = db.client.table('poi')\
            .select('id,google_place_id,name')\
            .limit(1)\
            .execute()
        
        if not pois.data:
            print("❌ No POIs found in database for testing")
            return False
        
        test_poi = pois.data[0]
        print(f"✅ Test POI: {test_poi['name']}")
        
    except Exception as e:
        print(f"❌ Failed to get test POI: {e}")
        return False
    
    # Test 2: Get photo references
    try:
        photo_refs = photo_manager.get_photo_references_from_poi(test_poi['google_place_id'])
        print(f"✅ Found {len(photo_refs)} photo references")
        
        if not photo_refs:
            print("⚠️ No photos found for this POI (this is normal for some POIs)")
            return True
        
    except Exception as e:
        print(f"❌ Failed to get photo references: {e}")
        return False
    
    # Test 3: Download and analyze first photo
    if photo_refs:
        try:
            first_photo = photo_refs[0]
            photo_reference = first_photo['photo_reference']
            
            print(f"📸 Downloading photo: {photo_reference[:20]}...")
            image_data = photo_manager.download_photo(photo_reference, 800)
            
            if image_data:
                print(f"✅ Downloaded photo: {len(image_data)} bytes")
                
                # Test photo analysis
                analysis = photo_manager.analyze_photo_quality(image_data, "test_photo.jpg")
                print(f"✅ Photo analysis complete:")
                print(f"   Quality score: {analysis['total_score']:.3f}")
                print(f"   Dimensions: {analysis.get('width', 'N/A')}x{analysis.get('height', 'N/A')}")
                print(f"   File size: {analysis.get('file_size', 0)} bytes")
                
                return True
            else:
                print("❌ Failed to download photo")
                return False
                
        except Exception as e:
            print(f"❌ Photo download/analysis failed: {e}")
            return False
    
    return True

def test_collection_photo_integration():
    """Test photo integration with collections"""
    print("\n🧪 Testing Collection Photo Integration")
    print("=" * 50)
    
    try:
        from ai.collection_generator import CollectionGenerator
        
        generator = CollectionGenerator()
        print("✅ Collection generator with photo support initialized")
        
        # Test getting best photo for a collection
        # Get some POIs with photos
        pois_with_photos = generator.db.client.table('poi')\
            .select('id')\
            .not_.is_('primary_photo', 'null')\
            .limit(3)\
            .execute()
        
        if pois_with_photos.data:
            poi_ids = [poi['id'] for poi in pois_with_photos.data]
            best_photo = generator.photo_manager.get_best_photo_for_collection(poi_ids)
            
            if best_photo:
                print(f"✅ Best photo selected for collection: {best_photo}")
            else:
                print("⚠️ No best photo found (normal if no photos processed yet)")
        else:
            print("⚠️ No POIs with photos found for collection test")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection photo integration test failed: {e}")
        return False

def main():
    """Run all photo system tests"""
    print("🚀 Trendr Photo System Validation")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Basic photo system
    result1 = test_photo_system()
    test_results.append(("Photo System", result1))
    
    # Test 2: Collection integration
    result2 = test_collection_photo_integration()
    test_results.append(("Collection Integration", result2))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Photo system is ready.")
    else:
        print("\n⚠️ Some tests failed. Check configuration and API keys.")
    
    print("\n💡 Next Steps:")
    print("1. Run photo backfill: python scripts/photo_processor.py --backfill --limit 10")
    print("2. Check photo coverage: python scripts/photo_processor.py --analyze")
    print("3. Generate collections with photos: python ai/collection_generator.py --city Montreal")

if __name__ == "__main__":
    main()