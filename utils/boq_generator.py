import json
import random
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

class BOQGenerator:
    def __init__(self, products_data: List[Dict[str, Any]]):
        self.products_data = products_data
        self.category_weights = {
            'display': 0.3,
            'audio': 0.25,
            'control': 0.2,
            'video': 0.15,
            'cable': 0.05,
            'mounting': 0.05
        }
    
    def generate_boq(self, project_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate BOQ based on project requirements"""
        
        # Analyze requirements
        required_categories = self._map_requirements_to_categories(
            project_requirements['requirements']
        )
        
        # Select products for each category
        selected_products = []
        total_cost = 0
        
        for category in required_categories:
            category_products = self._get_products_by_category(category)
            if category_products:
                # Use AI-like selection (for now, random selection with some logic)
                selected = self._smart_product_selection(
                    category_products, 
                    project_requirements
                )
                
                for product in selected:
                    quantity = self._calculate_quantity(product, project_requirements)
                    unit_cost = product.get('price', 0)
                    line_total = quantity * unit_cost
                    
                    boq_item = {
                        'category': category,
                        'description': product['description'],
                        'model_no': product['model_no'],
                        'company': product['company'],
                        'quantity': quantity,
                        'unit': self._get_unit(product),
                        'unit_cost': unit_cost,
                        'total_cost': line_total,
                        'specifications': product.get('specifications', {}),
                        'compatibility': product.get('compatibility', [])
                    }
                    
                    selected_products.append(boq_item)
                    total_cost += line_total
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(selected_products)
        
        # Generate BOQ summary
        boq_summary = {
            'project_name': project_requirements['project_name'],
            'client_name': project_requirements['client_name'],
            'project_type': project_requirements['project_type'],
            'generated_date': datetime.now().isoformat(),
            'items': selected_products,
            'total_cost': total_cost,
            'compliance_score': compliance_score,
            'recommendations': self._generate_recommendations(selected_products)
        }
        
        return boq_summary
    
    def _map_requirements_to_categories(self, requirements: List[str]) -> List[str]:
        """Map user requirements to product categories"""
        category_mapping = {
            'Display Systems': ['display', 'mounting'],
            'Audio Systems': ['audio'],
            'Control Systems': ['control'],
            'Video Conferencing': ['video', 'audio', 'display'],
            'Digital Signage': ['display', 'control'],
            'Lighting Control': ['control'],
            'Streaming Solutions': ['video', 'audio']
        }
        
        categories = set()
        for requirement in requirements:
            if requirement in category_mapping:
                categories.update(category_mapping[requirement])
        
        # Always include cables for any AV system
        categories.add('cable')
        
        return list(categories)
    
    def _get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get products by category"""
        return [p for p in self.products_data if p['category'] == category]
    
    def _smart_product_selection(self, products: List[Dict[str, Any]], 
                                requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI-powered product selection"""
        if not products:
            return []
        
        # Simple scoring algorithm (can be replaced with actual ML model)
        scored_products = []
        
        for product in products:
            score = 0
            
            # Price score (budget consideration)
            budget_multiplier = self._get_budget_multiplier(requirements['budget_range'])
            if product['price'] <= budget_multiplier:
                score += 3
            elif product['price'] <= budget_multiplier * 1.5:
                score += 1
            
            # Compatibility score
            compatibility_count = len(product.get('compatibility', []))
            score += min(compatibility_count, 3)
            
            # Brand reputation (simplified)
            if any(brand in product['company'].lower() for brand in ['sony', 'panasonic', 'crestron', 'extron']):
                score += 2
            
            scored_products.append((product, score))
        
        # Sort by score and select top products
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 1-3 products based on project size
        audience_size = requirements.get('audience_size', 20)
        num_products = min(3, max(1, audience_size // 20))
        
        return [p[0] for p in scored_products[:num_products]]
    
    def _calculate_quantity(self, product: Dict[str, Any], 
                          requirements: Dict[str, Any]) -> int:
        """Calculate quantity based on product type and room requirements"""
        audience_size = requirements.get('audience_size', 20)
        category = product['category']
        
        quantity_map = {
            'display': max(1, audience_size // 50),
            'audio': max(2, audience_size // 15),
            'control': 1,
            'video': max(1, audience_size // 30),
            'cable': max(5, audience_size // 5),
            'mounting': max(1, audience_size // 50)
        }
        
        return quantity_map.get(category, 1)
    
    def _get_unit(self, product: Dict[str, Any]) -> str:
        """Get unit of measurement for product"""
        category = product['category']
        
        unit_map = {
            'display': 'Each',
            'audio': 'Each',
            'control': 'Each',
            'video': 'Each',
            'cable': 'Meter',
            'mounting': 'Each'
        }
        
        return unit_map.get(category, 'Each')
    
    def _get_budget_multiplier(self, budget_range: str) -> float:
        """Get budget multiplier based on range"""
        multipliers = {
            "< $10,000": 200,
            "$10,000 - $50,000": 1000,
            "$50,000 - $100,000": 2500,
            "> $100,000": 5000
        }
        
        return multipliers.get(budget_range, 1000)
    
    def _calculate_compliance_score(self, products: List[Dict[str, Any]]) -> int:
        """Calculate AVIXA compliance score"""
        # Simplified compliance calculation
        score = 70  # Base score
        
        categories_present = set(p['category'] for p in products)
        
        # Bonus for having essential categories
        if 'display' in categories_present:
            score += 10
        if 'audio' in categories_present:
            score += 10
        if 'control' in categories_present:
            score += 5
        if 'cable' in categories_present:
            score += 5
        
        return min(100, score)
    
    def _generate_recommendations(self, products: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        total_cost = sum(p['total_cost'] for p in products)
        
        if total_cost > 50000:
            recommendations.append("Consider phased implementation to manage budget")
        
        categories = set(p['category'] for p in products)
        if 'control' not in categories:
            recommendations.append("Add control system for better system integration")
        
        if 'cable' not in categories:
            recommendations.append("Include proper cabling for system reliability")
        
        recommendations.append("Schedule regular maintenance for optimal performance")
        
        return recommendations
