import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional
import logging

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_data = []
        
    def clean_product_data(self, products_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and standardize product data"""
        cleaned_data = []
        
        for product in products_data:
            cleaned_product = self._clean_single_product(product)
            if cleaned_product:
                cleaned_data.append(cleaned_product)
        
        self.processed_data = cleaned_data
        return cleaned_data
    
    def _clean_single_product(self, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean individual product data"""
        try:
            # Skip products with missing essential data
            if not product.get('model_no') or not product.get('description'):
                return None
            
            cleaned = {
                'id': self._generate_product_id(product),
                'company': self._clean_text(product.get('company', '')),
                'model_no': self._clean_model_number(product.get('model_no', '')),
                'description': self._clean_description(product.get('description', '')),
                'category': self._standardize_category(product.get('category', 'other')),
                'price': self._clean_price(product.get('price', 0)),
                'specifications': self._clean_specifications(product.get('specifications', {})),
                'compatibility': self._clean_compatibility(product.get('compatibility', [])),
                'brand': self._extract_brand(product),
                'features': self._extract_features(product.get('description', '')),
                'power_requirements': self._extract_power_info(product),
                'dimensions': self._extract_dimensions(product),
                'warranty': self._extract_warranty(product)
            }
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error cleaning product {product.get('model_no', 'unknown')}: {str(e)}")
            return None
    
    def _generate_product_id(self, product: Dict[str, Any]) -> str:
        """Generate unique product ID"""
        company = product.get('company', 'unknown')
        model = product.get('model_no', 'unknown')
        return f"{company}_{model}".replace(' ', '_').lower()
    
    def _clean_text(self, text: str) -> str:
        """Clean and standardize text"""
        if not isinstance(text, str):
            return ''
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        return text.strip()
    
    def _clean_model_number(self, model_no: str) -> str:
        """Clean model number"""
        if not isinstance(model_no, str):
            return str(model_no)
        
        # Remove special characters but keep alphanumeric and common separators
        model_no = re.sub(r'[^\w\-\.]', '', model_no)
        return model_no.strip()
    
    def _clean_description(self, description: str) -> str:
        """Clean product description"""
        if not isinstance(description, str):
            return ''
        
        # Remove excessive punctuation and normalize
        description = re.sub(r'[^\w\s\-\.,()]', ' ', description)
        description = ' '.join(description.split())
        return description.strip()
    
    def _standardize_category(self, category: str) -> str:
        """Standardize product category"""
        if not isinstance(category, str):
            return 'other'
        
        category_map = {
            'display': ['display', 'monitor', 'screen', 'projector', 'tv'],
            'audio': ['audio', 'speaker', 'microphone', 'amplifier', 'mixer', 'sound'],
            'control': ['control', 'controller', 'processor', 'switch'],
            'video': ['video', 'camera', 'recorder', 'streaming'],
            'cable': ['cable', 'wire', 'connector', 'adapter'],
            'mounting': ['mount', 'bracket', 'stand', 'rack'],
            'lighting': ['light', 'lighting', 'led', 'lamp'],
            'network': ['network', 'ethernet', 'wifi', 'wireless']
        }
        
        category_lower = category.lower()
        
        for std_category, keywords in category_map.items():
            if any(keyword in category_lower for keyword in keywords):
                return std_category
        
        return 'other'
    
    def _clean_price(self, price: Any) -> float:
        """Clean and convert price to float"""
        if isinstance(price, (int, float)):
            return max(0.0, float(price))
        
        if isinstance(price, str):
            # Remove currency symbols and commas
            price_str = re.sub(r'[^\d\.]', '', price)
            try:
                return max(0.0, float(price_str))
            except ValueError:
                return 0.0
        
        return 0.0
    
    def _clean_specifications(self, specs: Any) -> Dict[str, Any]:
        """Clean specifications data"""
        if isinstance(specs, dict):
            return {k: str(v) for k, v in specs.items() if v is not None}
        elif isinstance(specs, str) and specs.strip():
            return {'description': specs.strip()}
        else:
            return {}
    
    def _clean_compatibility(self, compatibility: Any) -> List[str]:
        """Clean compatibility data"""
        if isinstance(compatibility, list):
            return [str(item).strip() for item in compatibility if item]
        elif isinstance(compatibility, str) and compatibility.strip():
            # Split by common separators
            items = re.split(r'[,;|]', compatibility)
            return [item.strip() for item in items if item.strip()]
        else:
            return []
    
    def _extract_brand(self, product: Dict[str, Any]) -> str:
        """Extract brand from product data"""
        # First check if brand is explicitly mentioned
        for key in ['brand', 'manufacturer', 'company']:
            if key in product and product[key]:
                return self._clean_text(str(product[key]))
        
        # Try to extract from model number or description
        description = product.get('description', '')
        model_no = product.get('model_no', '')
        
        # Common AV brands
        brands = [
            'Sony', 'Samsung', 'LG', 'Panasonic', 'Sharp', 'NEC', 'Epson',
            'Crestron', 'Extron', 'AMX', 'Biamp', 'QSC', 'Shure', 'Audio-Technica',
            'Sennheiser', 'Bose', 'JBL', 'Yamaha', 'Denon', 'Marantz',
            'Canon', 'Nikon', 'Polycom', 'Cisco', 'Logitech', 'Kramer'
        ]
        
        text_to_search = f"{description} {model_no}".lower()
        
        for brand in brands:
            if brand.lower() in text_to_search:
                return brand
        
        return product.get('company', 'Unknown')
    
    def _extract_features(self, description: str) -> List[str]:
        """Extract key features from description"""
        if not isinstance(description, str):
            return []
        
        features = []
        description_lower = description.lower()
        
        # Common AV features
        feature_keywords = {
            '4K': ['4k', 'uhd', '2160p'],
            'HD': ['hd', '1080p', 'full hd'],
            'Wireless': ['wireless', 'wifi', 'bluetooth'],
            'HDR': ['hdr', 'high dynamic range'],
            'Touch Screen': ['touch', 'touchscreen'],
            'Network': ['network', 'ethernet', 'ip'],
            'Remote Control': ['remote', 'ir control'],
            'USB': ['usb'],
            'HDMI': ['hdmi'],
            'Audio': ['audio', 'sound', 'speaker']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                features.append(feature)
        
        return features
    
    def _extract_power_info(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract power requirements"""
        power_info = {}
        
        # Look for power information in specifications or description
        text_sources = [
            product.get('description', ''),
            str(product.get('specifications', {}))
        ]
        
        full_text = ' '.join(text_sources).lower()
        
        # Extract voltage
        voltage_match = re.search(r'(\d+)v', full_text)
        if voltage_match:
            power_info['voltage'] = f"{voltage_match.group(1)}V"
        
        # Extract wattage
        wattage_match = re.search(r'(\d+)w', full_text)
        if wattage_match:
            power_info['power_consumption'] = f"{wattage_match.group(1)}W"
        
        return power_info
    
    def _extract_dimensions(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dimensions information"""
        dimensions = {}
        
        text_sources = [
            product.get('description', ''),
            str(product.get('specifications', {}))
        ]
        
        full_text = ' '.join(text_sources)
        
        # Look for dimension patterns
        dim_patterns = [
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*(mm|cm|inch)',
            r'(\d+\.?\d*)\s*(mm|cm|inch)\s*x\s*(\d+\.?\d*)\s*(mm|cm|inch)',
        ]
        
        for pattern in dim_patterns:
            match = re.search(pattern, full_text.lower())
            if match:
                dimensions['raw'] = match.group(0)
                break
        
        return dimensions
    
    def _extract_warranty(self, product: Dict[str, Any]) -> str:
        """Extract warranty information"""
        text_sources = [
            product.get('description', ''),
            str(product.get('specifications', {}))
        ]
        
        full_text = ' '.join(text_sources).lower()
        
        # Look for warranty patterns
        warranty_patterns = [
            r'(\d+)\s*year\s*warranty',
            r'warranty\s*(\d+)\s*year',
            r'(\d+)\s*month\s*warranty'
        ]
        
        for pattern in warranty_patterns:
            match = re.search(pattern, full_text)
            if match:
                return match.group(0)
        
        return 'Standard'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed data"""
        if not self.processed_data:
            return {}
        
        df = pd.DataFrame(self.processed_data)
        
        stats = {
            'total_products': len(self.processed_data),
            'categories': df['category'].value_counts().to_dict(),
            'companies': df['company'].value_counts().to_dict(),
            'brands': df['brand'].value_counts().to_dict(),
            'price_stats': {
                'mean': df['price'].mean(),
                'median': df['price'].median(),
                'min': df['price'].min(),
                'max': df['price'].max()
            },
            'missing_data': {
                'no_price': len(df[df['price'] == 0]),
                'no_specs': len(df[df['specifications'].apply(lambda x: len(x) == 0)]),
                'no_compatibility': len(df[df['compatibility'].apply(lambda x: len(x) == 0)])
            }
        }
        
        return stats
