import pandas as pd
import json
import os
from typing import List, Dict, Any

class CSVProcessor:
    """Process CSV files and convert to BOQ JSON format"""
    
    def __init__(self):
        self.category_mapping = {
            'Displays & Projectors': 'display',
            'Mounts & Racks': 'mounting',
            'Audio': 'audio',
            'Control Systems': 'control',
            'Video': 'video',
            'Cables': 'cable',
            'Speakers': 'audio',
            'Microphones': 'audio',
            'Amplifiers': 'audio',
            'Processors': 'control',
            'Switchers': 'control',
            'Cameras': 'video',
            'Streaming': 'video'
        }
        
        self.compatibility_mapping = {
            'samsung_magicinfo': 'Samsung MagicInfo',
            'vesa': 'VESA',
            'hdmi': 'HDMI',
            'usb': 'USB',
            'ethernet': 'Ethernet',
            'wifi': 'Wireless',
            'bluetooth': 'Bluetooth',
            'displayport': 'DisplayPort',
            'vga': 'VGA',
            'dvi': 'DVI'
        }
    
    def map_category(self, category_name: str) -> str:
        """Map CSV category to BOQ category"""
        if category_name in self.category_mapping:
            return self.category_mapping[category_name]
        
        # Fallback keyword matching
        category_lower = str(category_name).lower()
        if 'display' in category_lower or 'projector' in category_lower:
            return 'display'
        elif 'audio' in category_lower or 'speaker' in category_lower or 'mic' in category_lower:
            return 'audio'
        elif 'control' in category_lower or 'processor' in category_lower:
            return 'control'
        elif 'video' in category_lower or 'camera' in category_lower:
            return 'video'
        elif 'cable' in category_lower or 'wire' in category_lower:
            return 'cable'
        elif 'mount' in category_lower or 'rack' in category_lower:
            return 'mounting'
        else:
            return 'other'
    
    def parse_compatibility(self, compatibility_tags: str) -> List[str]:
        """Parse compatibility tags into standardized list"""
        if pd.isna(compatibility_tags):
            return []
        
        tags = str(compatibility_tags).replace(' ', '').split(',')
        mapped_tags = []
        
        for tag in tags:
            tag = tag.lower().strip()
            if tag in self.compatibility_mapping:
                mapped_tags.append(self.compatibility_mapping[tag])
            elif tag:  # Keep non-empty tags
                mapped_tags.append(tag.title())
        
        return mapped_tags
    
    def parse_use_cases(self, use_case_tags: str) -> List[str]:
        """Parse use case tags"""
        if pd.isna(use_case_tags):
            return []
        
        return [tag.strip() for tag in str(use_case_tags).split(',') if tag.strip()]
    
    def process_csv_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process pandas DataFrame to BOQ format"""
        processed_products = []
        
        for index, row in df.iterrows():
            # Skip rows with missing essential data
            if pd.isna(row.get('name', '')) or pd.isna(row.get('brand', '')):
                continue
            
            # Extract and clean data
            company = str(row['brand']).strip()
            name = str(row['name']).strip()
            category = str(row['category']) if 'category' in row else 'other'
            price = float(row['price']) if not pd.isna(row.get('price', 0)) else 0.0
            
            # Create product dictionary
            product = {
                'company': company,
                'model_no': f"{company}-{index:04d}",  # Generate model number
                'description': name,
                'category': self.map_category(category),
                'price': price,
                'specifications': {
                    'features': str(row['features']) if not pd.isna(row.get('features', '')) else '',
                    'tier': str(row['tier']) if not pd.isna(row.get('tier', '')) else 'Standard',
                    'use_cases': self.parse_use_cases(row.get('use_case_tags', ''))
                },
                'compatibility': self.parse_compatibility(row.get('compatibility_tags', ''))
            }
            
            processed_products.append(product)
        
        return processed_products
    
    def process_csv_file(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file from file path"""
        try:
            df = pd.read_csv(csv_file_path)
            return self.process_csv_dataframe(df)
        except Exception as e:
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def save_to_json(self, products_data: List[Dict[str, Any]], 
                     output_file: str = 'data/processed_products.json') -> bool:
        """Save processed data to JSON file"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(products_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
    
    def get_processing_summary(self, products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of processed data"""
        if not products_data:
            return {'total_products': 0}
        
        # Category breakdown
        categories = {}
        companies = {}
        price_stats = []
        
        for product in products_data:
            # Categories
            cat = product.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Companies
            company = product.get('company', 'Unknown')
            companies[company] = companies.get(company, 0) + 1
            
            # Prices
            price = product.get('price', 0)
            if price > 0:
                price_stats.append(price)
        
        summary = {
            'total_products': len(products_data),
            'categories': categories,
            'top_companies': dict(sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]),
            'price_stats': {
                'min_price': min(price_stats) if price_stats else 0,
                'max_price': max(price_stats) if price_stats else 0,
                'avg_price': sum(price_stats) / len(price_stats) if price_stats else 0
            }
        }
        
        return summary
