# Enhancement: Subcategories Intelligentes

## 🎯 Problème Identifié

### État Actuel (KISS V1)
- **Google Places API** renvoie des types génériques : `['restaurant', 'food', 'establishment']`
- **Google Maps Interface** affiche des labels plus précis : "Restaurant français", "Restaurant italien"
- **Notre implémentation** stocke les types bruts : `subcategories: ['food']`

### Exemples Concrets
```
L'Ami Jean:
- Google Maps: "Restaurant français" 
- API Types: ['establishment', 'food', 'point_of_interest', 'restaurant']
- Stocké: subcategories: ['food']
- Manque: 'français', 'bistrot', 'traditionnel'

Pierre Hermé:
- Google Maps: "Pâtisserie" 
- API Types: ['bakery', 'food', 'store']
- Stocké: subcategories: ['food', 'store']
- Manque: 'macarons', 'haut_de_gamme', 'pâtisserie_fine'
```

## 🚀 Solution Future: Classifier Intelligent

### Phase 1: Enrichissement par NLP/AI
**Sources de données à analyser :**
- **Nom du POI** : "L'Ami Jean" → français, bistrot
- **Description Google** : "Bistrot traditionnel servant une cuisine française"
- **Website content** : parsing automatique du site
- **Reviews keywords** : analyse des avis pour extraire des tags

### Phase 2: Classification Multi-Sources
**Algorithme proposé :**
```python
def classify_poi_intelligent(poi_data):
    # Base: types Google bruts
    base_categories = poi_data.subcategories  # ['food']
    
    # Enrichissement par sources
    enhanced_categories = []
    
    # 1. NLP sur le nom
    name_tags = extract_cuisine_from_name(poi_data.name)
    # "L'Ami Jean" → ['français', 'bistrot']
    
    # 2. Website analysis
    if poi_data.website:
        website_tags = analyze_website_content(poi_data.website)
    
    # 3. Reviews analysis (via mentions existantes)
    review_tags = analyze_mentions_keywords(poi_data.id)
    
    # 4. Geographic context
    location_tags = infer_from_location(poi_data.lat, poi_data.lng)
    # Quartier Latin → ['traditionnel', 'historique']
    
    return merge_and_rank_categories(base_categories, enhanced_categories)
```

### Phase 3: Machine Learning
**Training data :**
- POIs existants avec classifications manuelles
- Correlation nom/website/reviews → subcategories
- Modèle de recommandation de tags

## 📋 Implementation Plan

### Étape 1: Data Collection
- [ ] Scraper websites des POIs existants
- [ ] Analyser les mentions existantes pour keywords
- [ ] Constituer un dataset d'entraînement

### Étape 2: NLP Pipeline
- [ ] Implémentation extraction cuisine du nom
- [ ] Parser de contenu website
- [ ] Analyseur de sentiment/keywords des reviews

### Étape 3: Classification Engine
- [ ] Nouveau service `poi_intelligent_classifier`
- [ ] API endpoint `/classify/enhance-subcategories`
- [ ] Intégration dans le pipeline d'ingestion

### Étape 4: Validation & Training
- [ ] Interface admin pour validation manuelle
- [ ] Feedback loop pour améliorer l'algo
- [ ] A/B testing des classifications

## 🔧 Schema Changes Needed

### Nouvelle table: `poi_classification_history`
```sql
CREATE TABLE poi_classification_history (
    id uuid PRIMARY KEY,
    poi_id uuid REFERENCES poi(id),
    classification_method text, -- 'api_raw', 'nlp_enhanced', 'ml_predicted'
    subcategories jsonb,
    confidence_score float,
    created_at timestamp,
    validated_by uuid -- admin user who validated
);
```

### Extension table POI:
```sql
ALTER TABLE poi ADD COLUMN enhanced_subcategories jsonb;
ALTER TABLE poi ADD COLUMN classification_confidence float;
ALTER TABLE poi ADD COLUMN last_classified_at timestamp;
```

## 💡 Quick Wins (Phase 0)

### Mappings Statiques Améliorés
En attendant le classifier intelligent, améliorer les mappings :

```python
ENHANCED_SUBCATEGORY_MAP = {
    # Patterns dans les noms
    'ami_jean': ['français', 'bistrot', 'traditionnel'],
    'pierre_herme': ['macarons', 'pâtisserie_fine', 'haut_de_gamme'],
    'hemingway': ['cocktails', 'luxe', 'historique'],
    
    # Patterns génériques
    'brasserie': ['français', 'brasserie'],
    'bistrot': ['français', 'bistrot'],
    'trattoria': ['italien', 'traditionnel'],
    'sushi': ['japonais', 'sushi'],
}
```

## 🎯 Success Metrics

### KPIs à mesurer :
- **Précision** : % de subcategories correctes vs validation manuelle
- **Couverture** : % de POIs avec subcategories enrichies 
- **User Engagement** : utilisation des filtres par subcategories
- **API Cost** : réduction calls Details grâce à meilleure classification

### Target Goals :
- **Phase 1** : 80% des restaurants avec cuisine identifiée
- **Phase 2** : 90% des POIs avec 3+ subcategories pertinentes
- **Phase 3** : Classification automatique < 5% erreur

## 📚 Références

### APIs & Services à intégrer :
- **OpenAI/Claude** : classification par LLM
- **spaCy/NLTK** : NLP pour extraction keywords
- **Google Reviews API** : analyse des avis
- **Foursquare/Yelp APIs** : cross-reference categories

### Inspiration :
- Classification Foursquare (très granulaire)
- TripAdvisor tags automatiques
- Yelp category inference

---

**Note**: Cette enhancement va transformer notre système de subcategories basiques en un véritable moteur de découverte intelligent, permettant des recherches fines comme "restaurant français traditionnel 7ème arrondissement" ou "pâtisserie haut de gamme macarons".