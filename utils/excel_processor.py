import pandas as pd
import json
import logging
from typing import List, Dict, Any
import streamlit as st

class ExcelProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_excel(self, file) -> List[Dict[str, Any]]:
        """Process Excel file with multiple sheets"""
        try:
            excel_file = pd.ExcelFile(file)
            all_products = []
            
            progress_bar = st.progress(0)
            total_sheets = len(excel_file.sheet_names)
            
            for idx, sheet_name in enumerate(excel_file.sheet_names):
                try:
                    df = pd.read_excel(file, sheet_name=sheet_name)
                    
                    # Clean and process the sheet
                    products = self._process_sheet(df, sheet_name)
                    all_products.extend(products)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / total_sheets)
                    
                except Exception as e:
                    st.warning(f"Error processing sheet {sheet_name}: {str(e)}")
                    continue
            
            return all_products
            
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    def _process_sheet(self, df: pd.DataFrame, sheet_name: str) -> List[Dict[str, Any]]:
        """Process individual sheet"""
        products = []
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        for _, row in df.iterrows():
            if pd.isna(row.get('model_no')) or pd.isna(row.get('description')):
                continue
                
            product = {
                'company': sheet_name,
                'model_no': str(row.get('model_no', '')).strip(),
                'description': str(row.get('description', '')).strip(),
                'category': self._categorize_product(str(row.get('description', ''))),
                'price': self._extract_price(row),
                'specifications': self._extract_specifications(row),
                'compatibility': self._determine_compatibility(str(row.get('description', '')))
            }
            
            products.append(product)
        
        return products
    
    def _categorize_product(self, description: str) -> str:
        """Categorize product based on description"""
        description_lower = description.lower()
        
        categories = {
            'display': ['monitor', 'display', 'screen', 'projector', 'tv'],
            'audio': ['speaker', 'microphone', 'amplifier', 'mixer', 'sound'],
            'control': ['controller', 'control', 'switch', 'processor'],
            'video': ['camera', 'recorder', 'video', 'streaming'],
            'cable': ['cable', 'wire', 'connector', 'adapter'],
            'mounting': ['mount', 'bracket', 'stand', 'rack']
        }
        
        for category, keywords in categories.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _extract_price(self, row: pd.Series) -> float:
        """Extract price from row"""
        price_columns = ['price', 'cost', 'amount', 'value']
        
        for col in price_columns:
            if col in row.index and not pd.isna(row[col]):
                try:
                    price_str = str(row[col]).replace('$', '').replace(',', '')
                    return float(price_str)
                except:
                    continue
        
        return 0.0
    
    def _extract_specifications(self, row: pd.Series) -> Dict[str, Any]:
        """Extract specifications from row"""
        specs = {}
        
        spec_columns = ['specifications', 'specs', 'features', 'details']
        
        for col in spec_columns:
            if col in row.index and not pd.isna(row[col]):
                specs[col] = str(row[col])
        
        return specs
    
    def _determine_compatibility(self, description: str) -> List[str]:
        """Determine product compatibility"""
        compatibility = []
        
        # Simple keyword-based compatibility determination
        if 'hdmi' in description.lower():
            compatibility.append('HDMI')
        if 'usb' in description.lower():
            compatibility.append('USB')
        if 'ethernet' in description.lower():
            compatibility.append('Ethernet')
        if 'wireless' in description.lower() or 'wifi' in description.lower():
            compatibility.append('Wireless')
        
        return compatibility
    
    def save_to_json(self, data: List[Dict[str, Any]], filename: str):
        """Save processed data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
