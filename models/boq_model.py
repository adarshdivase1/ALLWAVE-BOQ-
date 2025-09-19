import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from typing import List, Dict, Any, Optional

class BOQModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.product_embeddings = None
        self.products_data = None
    
    def train(self, products_data: List[Dict]):
        """Train the BOQ generation model"""
        self.products_data = products_data
        
        # Prepare text data for vectorization
        descriptions = [p['description'] for p in products_data]
        
        # Create TF-IDF embeddings
        self.product_embeddings = self.vectorizer.fit_transform(descriptions)
        
        # Train price prediction model
        self._train_price_model(products_data)
        
        return {"status": "Model trained successfully"}
    
    def _train_price_model(self, products_data: List[Dict]):
        """Train price prediction model"""
        # Create features for price prediction
        features = []
        prices = []
        
        for product in products_data:
            if product['price'] > 0:  # Only use products with valid prices
                feature_vector = [
                    len(product['description']),  # Description length
                    len(product.get('specifications', {})),  # Number of specs
                    len(product.get('compatibility', [])),  # Compatibility count
                    1 if product['category'] == 'display' else 0,
                    1 if product['category'] == 'audio' else 0,
                    1 if product['category'] == 'control' else 0,
                ]
                
                features.append(feature_vector)
                prices.append(product['price'])
        
        if features:
            X = np.array(features)
            y = np.array(prices)
            self.price_model.fit(X, y)
    
    def recommend_products(self, requirements: str, category: str = None, top_k: int = 5):
        """Recommend products based on requirements"""
        if self.product_embeddings is None:
            return []
        
        # Vectorize requirements
        req_vector = self.vectorizer.transform([requirements])
        
        # Calculate similarity scores
        similarities = cosine_similarity(req_vector, self.product_embeddings)[0]
        
        # Filter by category if specified
        if category:
            filtered_indices = [
                i for i, product in enumerate(self.products_data)
                if product['category'] == category
            ]
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
            sorted_products = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)
        else:
            sorted_products = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        
        # Get top-k recommendations
        recommendations = []
        for i, (idx, score) in enumerate(sorted_products[:top_k]):
            product = self.products_data[idx].copy()
            product['similarity_score'] = float(score)
            product['rank'] = i + 1
            recommendations.append(product)
        
        return recommendations
    
    def predict_price(self, product_features: Dict[str, Any]) -> float:
        """Predict price for a product based on its features"""
        feature_vector = [
            len(product_features.get('description', '')),
            len(product_features.get('specifications', {})),
            len(product_features.get('compatibility', [])),
            1 if product_features.get('category') == 'display' else 0,
            1 if product_features.get('category') == 'audio' else 0,
            1 if product_features.get('category') == 'control' else 0,
        ]
        
        X = np.array([feature_vector])
        predicted_price = self.price_model.predict(X)[0]
        
        return max(0, predicted_price)  # Ensure non-negative price
    
    def generate_boq(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Bill of Quantities based on requirements"""
        boq = {
            'project_name': requirements.get('project_name', 'Untitled Project'),
            'total_estimated_cost': 0,
            'items': [],
            'categories': {
                'display': {'items': [], 'subtotal': 0},
                'audio': {'items': [], 'subtotal': 0},
                'video': {'items': [], 'subtotal': 0},
                'control': {'items': [], 'subtotal': 0}
            },
            'summary': {}
        }
        
        # Process each category requirement
        for category, category_req in requirements.get('categories', {}).items():
            if category_req.get('required', False):
                quantity = category_req.get('quantity', 1)
                budget = category_req.get('budget', None)
                specifications = category_req.get('specifications', '')
                
                # Get recommendations for this category
                recommendations = self.recommend_products(
                    specifications, category=category, top_k=3
                )
                
                if recommendations:
                    # Select the best product (highest similarity score)
                    selected_product = recommendations[0]
                    
                    # Calculate costs
                    unit_price = selected_product['price']
                    total_price = unit_price * quantity
                    
                    # Check budget constraint
                    within_budget = budget is None or total_price <= budget
                    
                    boq_item = {
                        'id': selected_product['id'],
                        'name': selected_product['name'],
                        'category': category,
                        'description': selected_product['description'],
                        'quantity': quantity,
                        'unit_price': unit_price,
                        'total_price': total_price,
                        'within_budget': within_budget,
                        'similarity_score': selected_product['similarity_score'],
                        'specifications': selected_product.get('specifications', {}),
                        'alternatives': recommendations[1:3]  # Include top 2 alternatives
                    }
                    
                    boq['items'].append(boq_item)
                    boq['categories'][category]['items'].append(boq_item)
                    boq['categories'][category]['subtotal'] += total_price
                    boq['total_estimated_cost'] += total_price
        
        # Generate summary
        boq['summary'] = {
            'total_items': len(boq['items']),
            'categories_covered': len([cat for cat in boq['categories'] if boq['categories'][cat]['items']]),
            'budget_compliant': all(item['within_budget'] for item in boq['items']),
            'average_similarity_score': np.mean([item['similarity_score'] for item in boq['items']]) if boq['items'] else 0
        }
        
        return boq
    
    def optimize_boq(self, boq: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize BOQ based on constraints (budget, performance, etc.)"""
        max_budget = constraints.get('max_budget', float('inf'))
        min_performance = constraints.get('min_performance', 0)
        
        optimized_boq = boq.copy()
        
        # If over budget, try to find cheaper alternatives
        if boq['total_estimated_cost'] > max_budget:
            for i, item in enumerate(optimized_boq['items']):
                if optimized_boq['total_estimated_cost'] <= max_budget:
                    break
                
                # Look for cheaper alternatives
                for alternative in item['alternatives']:
                    alt_total = alternative['price'] * item['quantity']
                    savings = item['total_price'] - alt_total
                    
                    if savings > 0 and alternative.get('similarity_score', 0) >= min_performance:
                        # Replace with cheaper alternative
                        optimized_boq['items'][i].update({
                            'id': alternative['id'],
                            'name': alternative['name'],
                            'unit_price': alternative['price'],
                            'total_price': alt_total,
                            'similarity_score': alternative.get('similarity_score', 0),
                            'optimized': True
                        })
                        
                        optimized_boq['total_estimated_cost'] -= savings
                        
                        # Update category subtotal
                        category = item['category']
                        optimized_boq['categories'][category]['subtotal'] -= savings
                        break
        
        # Update summary
        optimized_boq['summary']['budget_compliant'] = optimized_boq['total_estimated_cost'] <= max_budget
        optimized_boq['summary']['optimized'] = True
        
        return optimized_boq
    
    def export_boq(self, boq: Dict[str, Any], format: str = 'json') -> str:
        """Export BOQ in specified format"""
        if format.lower() == 'json':
            return json.dumps(boq, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Convert to CSV format
            csv_data = []
            csv_data.append("Category,Item Name,Description,Quantity,Unit Price,Total Price,Within Budget")
            
            for item in boq['items']:
                csv_data.append(
                    f"{item['category']},{item['name']},\"{item['description']}\","
                    f"{item['quantity']},{item['unit_price']},{item['total_price']},{item['within_budget']}"
                )
            
            csv_data.append(f",,,,Total,{boq['total_estimated_cost']},")
            return "\n".join(csv_data)
        
        elif format.lower() == 'summary':
            # Generate text summary
            summary = f"Bill of Quantities - {boq['project_name']}\n"
            summary += "=" * 50 + "\n\n"
            
            for category, cat_data in boq['categories'].items():
                if cat_data['items']:
                    summary += f"{category.upper()} SYSTEMS:\n"
                    for item in cat_data['items']:
                        summary += f"  - {item['name']} (Qty: {item['quantity']}) - ${item['total_price']:,.2f}\n"
                    summary += f"  Subtotal: ${cat_data['subtotal']:,.2f}\n\n"
            
            summary += f"TOTAL ESTIMATED COST: ${boq['total_estimated_cost']:,.2f}\n"
            summary += f"Items: {boq['summary']['total_items']}\n"
            summary += f"Budget Compliant: {'Yes' if boq['summary']['budget_compliant'] else 'No'}\n"
            
            return summary
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'vectorizer': self.vectorizer,
            'price_model': self.price_model,
            'products_data': self.products_data
        }
        
        joblib.dump(model_data, filepath)
        
        # Save embeddings separately (sparse matrix)
        if self.product_embeddings is not None:
            joblib.dump(self.product_embeddings, f"{filepath}_embeddings.pkl")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.price_model = model_data['price_model']
        self.products_data = model_data['products_data']
        
        # Load embeddings
        try:
            self.product_embeddings = joblib.load(f"{filepath}_embeddings.pkl")
        except FileNotFoundError:
            print("Warning: Embeddings file not found. You may need to retrain the model.")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the trained model"""
        if self.products_data is None:
            return {"status": "Model not trained"}
        
        stats = {
            "total_products": len(self.products_data),
            "categories": {},
            "price_range": {
                "min": min(p['price'] for p in self.products_data if p['price'] > 0),
                "max": max(p['price'] for p in self.products_data if p['price'] > 0),
                "avg": np.mean([p['price'] for p in self.products_data if p['price'] > 0])
            },
            "vectorizer_features": self.vectorizer.max_features,
            "embedding_shape": self.product_embeddings.shape if self.product_embeddings is not None else None
        }
        
        # Category statistics
        for product in self.products_data:
            category = product['category']
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1
        
        return stats
