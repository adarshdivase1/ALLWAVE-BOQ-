import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def prepare_training_data(self, products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for model training"""
        df = pd.DataFrame(products_data)
        
        # Create features for different models
        training_data = {
            'price_prediction': self._prepare_price_data(df),
            'category_classification': self._prepare_category_data(df),
            'compatibility_prediction': self._prepare_compatibility_data(df),
            'recommendation_system': self._prepare_recommendation_data(df)
        }
        
        return training_data
    
    def train_price_prediction_model(self, products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train price prediction model"""
        self.logger.info("Training price prediction model...")
        
        df = pd.DataFrame(products_data)
        
        # Filter products with valid prices
        df = df[df['price'] > 0]
        
        if len(df) < 10:
            raise ValueError("Insufficient data for training price model")
        
        # Create features
        features = []
        for _, row in df.iterrows():
            feature_vector = self._create_price_features(row)
            features.append(feature_vector)
        
        X = np.array(features)
        y = df['price'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Store model and scaler
        self.models['price_prediction'] = best_model
        self.scalers['price_prediction'] = scaler
        
        return {
            'model_type': 'price_prediction',
            'rmse': rmse,
            'r2_score': best_model.score(X_test_scaled, y_test),
            'best_params': grid_search.best_params_,
            'feature_importance': best_model.feature_importances_.tolist()
        }
    
    def train_category_classification_model(self, products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train product category classification model"""
        self.logger.info("Training category classification model...")
        
        df = pd.DataFrame(products_data)
        
        # Prepare text data
        descriptions = df['description'].fillna('')
        categories = df['category']
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(descriptions)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(categories)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store models
        self.models['category_classification'] = model
        self.vectorizers['category_classification'] = vectorizer
        self.label_encoders['category_classification'] = label_encoder
        
        return {
            'model_type': 'category_classification',
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=label_encoder.classes_,
                                                         output_dict=True)
        }
    
    def train_compatibility_model(self, products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train product compatibility model"""
        self.logger.info("Training compatibility model...")
        
        # Create compatibility pairs
        compatibility_data = self._create_compatibility_pairs(products_data)
        
        if len(compatibility_data) < 100:
            self.logger.warning("Insufficient compatibility data, using rule-based system")
            return {'model_type': 'compatibility', 'method': 'rule_based'}
        
        df_compat = pd.DataFrame(compatibility_data)
        
        # Create features for compatibility prediction
        feature_columns = ['cat1_display', 'cat1_audio', 'cat1_control', 'cat1_video',
                          'cat2_display', 'cat2_audio', 'cat2_control', 'cat2_video',
                          'common_interfaces', 'brand_match']
        
        X = df_compat[feature_columns]
        y = df_compat['compatible']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['compatibility'] = model
        
        return {
            'model_type': 'compatibility',
            'accuracy': accuracy,
            'method': 'ml_model'
        }
    
    def train_recommendation_model(self, products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train product recommendation model"""
        self.logger.info("Training recommendation model...")
        
        df = pd.DataFrame(products_data)
        
        # Create product embeddings using descriptions
        descriptions = df['description'].fillna('')
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        embeddings = vectorizer.fit_transform(descriptions)
        
        # Store for similarity calculations
        self.models['recommendation'] = {
            'embeddings': embeddings,
            'products_index': df.index.tolist(),
            'method': 'tfidf_similarity'
        }
        self.vectorizers['recommendation'] = vectorizer
        
        return {
            'model_type': 'recommendation',
            'embedding_dimension': embeddings.shape[1],
            'method': 'tfidf_similarity'
        }
    
    def _create_price_features(self, row: pd.Series) -> List[float]:
        """Create features for price prediction"""
        features = [
            len(row.get('description', '')),  # Description length
            len(row.get('specifications', {})),  # Number of specifications
            len(row.get('compatibility', [])),  # Number of compatibility options
            len(row.get('features', [])),  # Number of features
            1 if row.get('category') == 'display' else 0,
            1 if row.get('category') == 'audio' else 0,
            1 if row.get('category') == 'control' else 0,
            1 if row.get('category') == 'video' else 0,
            1 if 'premium' in row.get('description', '').lower() else 0,
            1 if 'professional' in row.get('description', '').lower() else 0,
        ]
        return features
    
    def _prepare_price_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for price prediction"""
        # Filter valid prices
        price_data = df[df['price'] > 0].copy()
        
        return {
            'features': [self._create_price_features(row) for _, row in price_data.iterrows()],
            'targets': price_data['price'].tolist(),
            'feature_names': ['desc_length', 'spec_count', 'compat_count', 'feature_count',
                            'is_display', 'is_audio', 'is_control', 'is_video', 
                            'is_premium', 'is_professional']
        }
    
    def _prepare_category_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for category classification"""
        return {
            'descriptions': df['description'].fillna('').tolist(),
            'categories': df['category'].tolist()
        }
    
    def _prepare_compatibility_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for compatibility prediction"""
        compatibility_pairs = self._create_compatibility_pairs(df.to_dict('records'))
        return {
            'pairs': compatibility_pairs
        }
    
    def _prepare_recommendation_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for recommendation system"""
        return {
            'descriptions': df['description'].fillna('').tolist(),
            'products': df.to_dict('records')
        }
    
    def _create_compatibility_pairs(self, products_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create compatibility training pairs"""
        pairs = []
        
        for i, product1 in enumerate(products_data):
            for j, product2 in enumerate(products_data[i+1:], i+1):
                # Create features for the pair
                pair_features = {
                    'product1_id': i,
                    'product2_id': j,
                    'cat1_display': 1 if product1['category'] == 'display' else 0,
                    'cat1_audio': 1 if product1['category'] == 'audio' else 0,
                    'cat1_control': 1 if product1['category'] == 'control' else 0,
                    'cat1_video': 1 if product1['category'] == 'video' else 0,
                    'cat2_display': 1 if product2['category'] == 'display' else 0,
                    'cat2_audio': 1 if product2['category'] == 'audio' else 0,
                    'cat2_control': 1 if product2['category'] == 'control' else 0,
                    'cat2_video': 1 if product2['category'] == 'video' else 0,
                    'common_interfaces': len(set(product1.get('compatibility', [])) & 
                                           set(product2.get('compatibility', []))),
                    'brand_match': 1 if product1.get('brand') == product2.get('brand') else 0,
                    'compatible': self._determine_compatibility(product1, product2)
                }
                
                pairs.append(pair_features)
                
                # Limit pairs to prevent memory issues
                if len(pairs) >= 10000:
                    break
            
            if len(pairs) >= 10000:
                break
        
        return pairs
    
    def _determine_compatibility(self, product1: Dict[str, Any], product2: Dict[str, Any]) -> int:
        """Determine if two products are compatible (simplified logic)"""
    def _determine_compatibility(self, product1: Dict[str, Any], product2: Dict[str, Any]) -> int:
        """Determine if two products are compatible (simplified logic)"""
        cat1 = product1.get('category', '')
        cat2 = product2.get('category', '')
        
        # Same category products are generally not compatible (no need to connect display to display)
        if cat1 == cat2:
            return 0
        
        # Get compatibility interfaces
        compat1 = set(product1.get('compatibility', []))
        compat2 = set(product2.get('compatibility', []))
        
        # If no common interfaces, not compatible
        common_interfaces = compat1 & compat2
        if not common_interfaces:
            return 0
        
        # Define compatibility rules
        compatible_pairs = {
            ('display', 'video'): 1,
            ('video', 'display'): 1,
            ('audio', 'video'): 1,
            ('video', 'audio'): 1,
            ('control', 'display'): 1,
            ('display', 'control'): 1,
            ('control', 'video'): 1,
            ('video', 'control'): 1,
            ('control', 'audio'): 1,
            ('audio', 'control'): 1,
        }
        
        # Check if category pair is in compatible pairs
        if (cat1, cat2) in compatible_pairs:
            return 1
        
        # Default to not compatible
        return 0
    
    def save_models(self, filepath: str) -> None:
        """Save all trained models to disk"""
        model_data = {
            'models': {},
            'vectorizers': {},
            'scalers': {},
            'label_encoders': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Save models
        for name, model in self.models.items():
            if name == 'recommendation':
                # Handle recommendation model separately (contains sparse matrices)
                model_data['models'][name] = {
                    'method': model['method'],
                    'products_index': model['products_index']
                }
                # Save embeddings separately
                joblib.dump(model['embeddings'], f"{filepath}_{name}_embeddings.pkl")
            else:
                model_data['models'][name] = joblib.dump(model, f"{filepath}_{name}_model.pkl")
        
        # Save vectorizers
        for name, vectorizer in self.vectorizers.items():
            joblib.dump(vectorizer, f"{filepath}_{name}_vectorizer.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{filepath}_{name}_scaler.pkl")
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, f"{filepath}_{name}_encoder.pkl")
        
        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk"""
        try:
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                model_data = json.load(f)
            
            # Load models
            for name in model_data['models']:
                if name == 'recommendation':
                    # Load recommendation model components
                    embeddings = joblib.load(f"{filepath}_{name}_embeddings.pkl")
                    self.models[name] = {
                        'embeddings': embeddings,
                        'method': model_data['models'][name]['method'],
                        'products_index': model_data['models'][name]['products_index']
                    }
                else:
                    self.models[name] = joblib.load(f"{filepath}_{name}_model.pkl")
            
            # Load vectorizers
            for name in model_data['vectorizers']:
                self.vectorizers[name] = joblib.load(f"{filepath}_{name}_vectorizer.pkl")
            
            # Load scalers  
            for name in model_data['scalers']:
                self.scalers[name] = joblib.load(f"{filepath}_{name}_scaler.pkl")
            
            # Load label encoders
            for name in model_data['label_encoders']:
                self.label_encoders[name] = joblib.load(f"{filepath}_{name}_encoder.pkl")
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except FileNotFoundError as e:
            self.logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        info = {
            'trained_models': list(self.models.keys()),
            'vectorizers': list(self.vectorizers.keys()),
            'scalers': list(self.scalers.keys()),
            'label_encoders': list(self.label_encoders.keys()),
            'model_details': {}
        }
        
        for name, model in self.models.items():
            if name == 'recommendation':
                info['model_details'][name] = {
                    'type': 'similarity_based',
                    'method': model.get('method', 'unknown'),
                    'embedding_shape': model['embeddings'].shape if 'embeddings' in model else None
                }
            elif hasattr(model, 'n_estimators'):
                info['model_details'][name] = {
                    'type': 'random_forest',
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth
                }
            else:
                info['model_details'][name] = {
                    'type': str(type(model).__name__)
                }
        
        return info
    
    def predict_price(self, product_features: Dict[str, Any]) -> float:
        """Predict price for a product"""
        if 'price_prediction' not in self.models:
            raise ValueError("Price prediction model not trained")
        
        # Create feature vector
        feature_vector = self._create_price_features(pd.Series(product_features))
        X = np.array([feature_vector])
        
        # Scale features
        scaler = self.scalers['price_prediction']
        X_scaled = scaler.transform(X)
        
        # Predict
        model = self.models['price_prediction']
        prediction = model.predict(X_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative price
    
    def classify_category(self, description: str) -> str:
        """Classify product category from description"""
        if 'category_classification' not in self.models:
            raise ValueError("Category classification model not trained")
        
        # Vectorize description
        vectorizer = self.vectorizers['category_classification']
        X = vectorizer.transform([description])
        
        # Predict
        model = self.models['category_classification']
        label_encoder = self.label_encoders['category_classification']
        
        prediction = model.predict(X)[0]
        category = label_encoder.inverse_transform([prediction])[0]
        
        return category
    
    def predict_compatibility(self, product1: Dict[str, Any], product2: Dict[str, Any]) -> float:
        """Predict compatibility between two products"""
        if 'compatibility' not in self.models:
            # Fall back to rule-based system
            return float(self._determine_compatibility(product1, product2))
        
        model_info = self.models['compatibility']
        if isinstance(model_info, dict) and model_info.get('method') == 'rule_based':
            return float(self._determine_compatibility(product1, product2))
        
        # Use ML model
        features = {
            'cat1_display': 1 if product1['category'] == 'display' else 0,
            'cat1_audio': 1 if product1['category'] == 'audio' else 0,
            'cat1_control': 1 if product1['category'] == 'control' else 0,
            'cat1_video': 1 if product1['category'] == 'video' else 0,
            'cat2_display': 1 if product2['category'] == 'display' else 0,
            'cat2_audio': 1 if product2['category'] == 'audio' else 0,
            'cat2_control': 1 if product2['category'] == 'control' else 0,
            'cat2_video': 1 if product2['category'] == 'video' else 0,
            'common_interfaces': len(set(product1.get('compatibility', [])) & 
                                   set(product2.get('compatibility', []))),
            'brand_match': 1 if product1.get('brand') == product2.get('brand') else 0,
        }
        
        feature_columns = ['cat1_display', 'cat1_audio', 'cat1_control', 'cat1_video',
                          'cat2_display', 'cat2_audio', 'cat2_control', 'cat2_video',
                          'common_interfaces', 'brand_match']
        
        X = np.array([[features[col] for col in feature_columns]])
        
        model = self.models['compatibility']
        prediction = model.predict_proba(X)[0][1]  # Probability of compatibility
        
        return prediction

        # Rule-based compatibility determination
