# Gatto POI Discovery Strategy - Problem Analysis & Solution

## 📋 Current Problem Statement

### Problem 1: Incomplete & Random Discovery
- **Current method**: `--seed --limit N` does random POI discovery
- **Issue**: Same POIs are re-ingested repeatedly (Maslow, Velvet Bar Paris seen multiple times)
- **Result**: Inefficient, non-exhaustive coverage of Paris POIs

### Problem 2: No Systematic Update Mechanism  
- **Current method**: No targeted way to update existing POIs
- **Issue**: Google Places API doesn't allow direct POI targeting without Place ID
- **Result**: Can't implement refresh cycles for existing POI data

### Problem 3: Detection of New POIs
- **Core mission**: Gatto needs to detect emerging/new POIs as they appear
- **Current gap**: No systematic approach to ensure new POIs are captured
- **Risk**: Missing trendy new spots that are Gatto's primary value proposition

## 🔍 Google Places API Capabilities Analysis

### Available Methods:

#### 1. Nearby Search
```python
nearby_search(location="48.8566,2.3522", radius=1000, type="restaurant")
```
- **Usage**: Find POIs within radius of coordinates
- **Limits**: 20 results/page, max 60 with pagination, max 50km radius
- **Best for**: Geographic systematic discovery

#### 2. Text Search  
```python
text_search(query="restaurant Paris 1er arrondissement")
```
- **Usage**: Keyword-based search in geographic area
- **Limits**: 20 results/page, max 60 with pagination
- **Best for**: Category-specific discovery

#### 3. Place Details
```python
place_details(place_id="ChIJN1t_tDeuEmsRUsoyG83frY4")
```
- **Usage**: Fetch complete details for known Place ID
- **Limits**: 1 POI per call
- **Best for**: Updating existing POIs

#### 4. Find Place
```python
find_place(input="Restaurant Name", inputtype="textquery")
```
- **Usage**: Find specific POI by name/address/phone
- **Limits**: Few most relevant results
- **Best for**: Targeted POI lookup

## 🗺️ Current Database Assets: urban_areas Table

### Available Geographic Data:
- **Total Paris zones**: 98
  - **20 arrondissements** (admin_level 9)
  - **78 quartiers** (admin_level 10) 
- **Each zone has**: name, type, admin_level, geometry_geojson
- **Coverage**: 100% of Paris with precise boundaries

### Geometric Challenge:
Using `nearby_search(center_point, radius=500)` on each quartier would create:
- **Overlapping coverage** → Duplicate POIs
- **Dead zones** at boundaries → Missed POIs

## 🎯 Recommended Solution: Multi-Layer Discovery Strategy

### Layer 1: Systematic Geographic Coverage (Base Layer)
**Method**: Text Search by administrative zones
```python
# For each quartier (78 zones)
for quartier in quartiers_paris:
    text_search(query=f"restaurant {quartier.name} Paris")
    text_search(query=f"bar {quartier.name} Paris")
    text_search(query=f"café {quartier.name} Paris")
```

**Advantages**:
- ✅ No geographic overlap issues
- ✅ Leverages Google's understanding of Paris neighborhoods  
- ✅ Systematic coverage guaranteed
- ✅ Uses existing urban_areas data

### Layer 2: Category-Specific Discovery (Completeness Layer)
**Method**: Text Search by arrondissement + category
```python
# For each arrondissement (20 zones) 
for arrondissement in arrondissements_paris:
    for category in ["restaurant", "bar", "café", "boulangerie"]:
        text_search(query=f"{category} {arrondissement.name}")
```

### Layer 3: Trend Detection (New POI Layer)
**Method**: Trend-based Text Search
```python
trend_queries = [
    "nouveau restaurant Paris 2024",
    "ouverture bar Paris",  
    "restaurant tendance Paris",
    "nouveau café Paris"
]
for query in trend_queries:
    text_search(query=query)
```

### Layer 4: Social Intelligence (Future Enhancement)
- Monitor social media mentions for new POI names
- Feed discovered names to `find_place` for validation
- Real-time detection of emerging spots

## 📊 Implementation Strategy

### Phase 1: Geographic Base Coverage
1. **Extract all quartiers** from urban_areas table
2. **Implement text_search** by quartier + POI type
3. **Track coverage** per quartier in database
4. **Deduplicate** POIs across searches

### Phase 2: Update Mechanism  
1. **Use Place IDs** stored in database
2. **Call place_details** for systematic updates
3. **Prioritize by**: recency, popularity, user engagement
4. **Schedule updates**: frequent for popular POIs, less for stable ones

### Phase 3: New POI Detection
1. **Implement trend queries** (Layer 3)
2. **Social media monitoring** integration
3. **ML-based** new POI detection from mentions

## 🚀 Next Steps

### Immediate Actions:
1. **Modify run_pipeline.py** to use geographic text search instead of random seed
2. **Create quartier coverage tracking** system
3. **Implement deduplication** logic for overlapping discoveries
4. **Add Place ID update** mechanism for existing POIs

### Success Metrics:
- **Coverage completeness**: % of Paris quartiers processed
- **Discovery efficiency**: New POIs found per API call  
- **Update freshness**: Average age of POI data
- **Duplicate reduction**: % reduction in duplicate processing

---

*Generated for Gatto Data Pipeline - POI Discovery Optimization*
*Date: 2025-09-16*