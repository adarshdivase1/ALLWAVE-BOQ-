import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import random
from collections import defaultdict
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent BOQ Generator",
    page_icon="🧠",
    layout="wide"
)

class IntelligentBOQGenerator:
    def __init__(self, products_data):
        self.products_data = products_data
        self.df = pd.DataFrame(products_data)
        
        # Clean and validate data
        self._clean_data()
        
        # Intelligent category mapping based on your data
        self.category_keywords = {
            'display': ['display', 'projector', 'monitor', 'screen', 'led', 'lcd'],
            'audio': ['speaker', 'microphone', 'amplifier', 'sound', 'audio'],
            'video': ['camera', 'video', 'conferencing', 'codec'],
            'control': ['controller', 'control', 'processor', 'switch'],
            'mounting': ['mount', 'bracket', 'rack', 'stand'],
            'cable': ['cable', 'connector', 'wire', 'hdmi', 'usb'],
            'installation': ['installation', 'service', 'setup', 'commissioning']
        }
        
        # Room size to equipment mapping
        self.room_equipment_logic = {
            'huddle_room': {'max_size': 6, 'display_size': '32-55"', 'audio': 'compact'},
            'small_room': {'max_size': 12, 'display_size': '55-65"', 'audio': 'standard'},
            'medium_room': {'max_size': 25, 'display_size': '65-75"', 'audio': 'enhanced'},
            'boardroom': {'max_size': 16, 'display_size': '75-86"', 'audio': 'premium'},
            'meeting_room': {'max_size': 20, 'display_size': '65-75"', 'audio': 'standard'},
            'large_room': {'max_size': 50, 'display_size': '75+"', 'audio': 'premium'}
        }

    def _clean_data(self):
        """Clean and validate the data"""
        if self.df.empty:
            return
        
        # Convert price to numeric, handling errors
        if 'price' in self.df.columns:
            # Remove currency symbols and convert to numeric
            self.df['price'] = pd.to_numeric(
                self.df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True), 
                errors='coerce'
            )
            # Fill NaN values with 0
            self.df['price'] = self.df['price'].fillna(0)
        
        # Ensure required columns exist
        required_columns = ['name', 'category', 'brand', 'price']
        for col in required_columns:
            if col not in self.df.columns:
                if col == 'name':
                    self.df[col] = 'Unknown Product'
                elif col == 'category':
                    self.df[col] = 'General'
                elif col == 'brand':
                    self.df[col] = 'Generic'
                elif col == 'price':
                    self.df[col] = 0.0
        
        # Clean string columns
        string_columns = ['name', 'category', 'brand', 'tier', 'features', 'use_case_tags']
        for col in string_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').astype(str)
        
        # Set default tier if missing
        if 'tier' not in self.df.columns or self.df['tier'].isna().all():
            self.df['tier'] = 'Standard'
        
        # Set default features if missing
        if 'features' not in self.df.columns:
            self.df['features'] = 'Standard features'
        
        # Set default use_case_tags if missing
        if 'use_case_tags' not in self.df.columns:
            self.df['use_case_tags'] = 'general'

    def analyze_data_structure(self):
        """Analyze the uploaded data to understand its structure"""
        if self.df.empty:
            return None
            
        analysis = {
            'total_products': len(self.df),
            'columns': list(self.df.columns),
            'categories': self.df['category'].unique().tolist() if 'category' in self.df.columns else [],
            'brands': self.df['brand'].unique().tolist() if 'brand' in self.df.columns else [],
            'price_range': {
                'min': float(self.df['price'].min()) if 'price' in self.df.columns else 0,
                'max': float(self.df['price'].max()) if 'price' in self.df.columns else 0
            },
            'use_cases': self._extract_use_cases(),
            'tiers': self.df['tier'].unique().tolist() if 'tier' in self.df.columns else []
        }
        return analysis

    def _extract_use_cases(self):
        """Extract use cases from tags"""
        use_cases = set()
        if 'use_case_tags' in self.df.columns:
            for tags in self.df['use_case_tags'].dropna():
                if isinstance(tags, str) and tags.strip():
                    use_cases.update([tag.strip() for tag in tags.split(',')])
        return list(use_cases)

    def generate_smart_boq(self, requirements):
        """Generate intelligent BOQ based on requirements and data analysis"""
        
        # Analyze requirements
        room_type = self._determine_room_type(requirements)
        equipment_needs = self._analyze_equipment_needs(requirements, room_type)
        budget_constraints = self._get_budget_constraints(requirements)
        
        selected_products = []
        total_budget_used = 0
        
        # Smart product selection for each category
        for category, needs in equipment_needs.items():
            if needs['required']:
                products = self._get_smart_product_selection(
                    category, needs, requirements, budget_constraints
                )
                
                for product in products:
                    boq_item = self._create_smart_boq_item(product, needs, requirements)
                    selected_products.append(boq_item)
                    total_budget_used += boq_item['Total Cost']

        # Add complementary products and services
        selected_products.extend(self._add_complementary_items(requirements, selected_products))
        
        # Calculate final totals
        equipment_total = sum(item['Total Cost'] for item in selected_products)
        installation_cost = self._calculate_installation_cost(selected_products, requirements)
        contingency = equipment_total * 0.10
        
        # Add installation
        selected_products.append({
            'Category': 'Services',
            'Brand': 'Professional Services',
            'Description': 'Complete Installation & Configuration',
            'Model No': 'INSTALL-PRO',
            'Quantity': 1,
            'Unit': 'Lump Sum',
            'Unit Cost': installation_cost,
            'Total Cost': installation_cost,
            'Tier': 'Standard',
            'Features': 'Full setup, testing, training, documentation'
        })
        
        # Add contingency
        selected_products.append({
            'Category': 'Project Management',
            'Brand': 'Project Controls',
            'Description': 'Project Contingency & Risk Management',
            'Model No': 'CONTINGENCY',
            'Quantity': 1,
            'Unit': 'Percentage',
            'Unit Cost': contingency,
            'Total Cost': contingency,
            'Tier': 'Standard',
            'Features': '10% contingency for unforeseen requirements'
        })

        final_total = equipment_total + installation_cost + contingency
        budget_max = self._get_budget_max(requirements['budget_range'])
        
        return {
            'project_info': {
                'name': requirements['project_name'],
                'client': requirements['client_name'],
                'room_type': room_type,
                'audience_size': requirements['audience_size'],
                'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'analysis': {
                'room_classification': room_type,
                'equipment_categories': len(equipment_needs),
                'budget_utilization': (final_total / budget_max * 100) if budget_max > 0 else 0,
                'total_items': len(selected_products)
            },
            'costs': {
                'equipment_subtotal': equipment_total,
                'installation': installation_cost,
                'contingency': contingency,
                'total': final_total
            },
            'items': selected_products
        }

    def _determine_room_type(self, requirements):
        """Intelligently determine room type based on requirements"""
        audience_size = requirements['audience_size']
        project_type = requirements.get('project_type', '').lower()
        requirements_list = [req.lower() for req in requirements.get('requirements', [])]
        
        # Room classification logic
        if audience_size <= 6:
            return 'huddle_room'
        elif audience_size <= 12:
            return 'small_room'
        elif audience_size <= 16 and ('boardroom' in project_type or 'board' in project_type):
            return 'boardroom'
        elif audience_size <= 25:
            return 'medium_room'
        elif audience_size <= 50:
            return 'meeting_room'
        else:
            return 'large_room'

    def _analyze_equipment_needs(self, requirements, room_type):
        """Analyze what equipment is needed based on requirements"""
        needs = defaultdict(lambda: {'required': False, 'quantity': 0, 'priority': 'low'})
        
        # Parse requirements
        req_list = [req.lower() for req in requirements.get('requirements', [])]
        audience_size = requirements['audience_size']
        
        # Display needs
        if any(x in ' '.join(req_list) for x in ['display', 'presentation', 'screen', 'visual']):
            needs['Displays & Projectors']['required'] = True
            needs['Displays & Projectors']['quantity'] = 1 if audience_size <= 25 else 2
            needs['Displays & Projectors']['priority'] = 'high'
            
            # Mounting is required for displays
            needs['Mounts & Racks']['required'] = True
            needs['Mounts & Racks']['quantity'] = needs['Displays & Projectors']['quantity']
            needs['Mounts & Racks']['priority'] = 'medium'

        # Video conferencing needs
        if any(x in ' '.join(req_list) for x in ['video', 'conferencing', 'meeting', 'collaboration']):
            needs['Video Conferencing']['required'] = True
            needs['Video Conferencing']['quantity'] = 1
            needs['Video Conferencing']['priority'] = 'high'

        # Audio needs
        if any(x in ' '.join(req_list) for x in ['audio', 'sound', 'speaker', 'microphone']) or needs['Video Conferencing']['required']:
            needs['Audio Systems'] = {'required': True, 'quantity': max(2, audience_size // 10), 'priority': 'high'}

        # Connectivity needs
        needs['Cables & Connectivity']['required'] = True
        needs['Cables & Connectivity']['quantity'] = 1
        needs['Cables & Connectivity']['priority'] = 'medium'

        # Control systems for larger rooms
        if audience_size > 12 or any(x in ' '.join(req_list) for x in ['control', 'automation']):
            needs['Control Systems'] = {'required': True, 'quantity': 1, 'priority': 'medium'}

        return needs

    def _get_smart_product_selection(self, category, needs, requirements, budget_constraints):
        """Smart product selection based on multiple criteria"""
        
        # Filter products by category
        category_products = self.df[self.df['category'] == category].copy()
        if category_products.empty:
            # Try to find similar categories
            similar_categories = [cat for cat in self.df['category'].unique() 
                                if any(keyword in cat.lower() for keyword in self.category_keywords.get(category.lower().split()[0], []))]
            if similar_categories:
                category_products = self.df[self.df['category'].isin(similar_categories)].copy()
        
        if category_products.empty:
            return []

        # Apply intelligent filtering
        filtered_products = self._apply_intelligent_filters(
            category_products, requirements, needs, category
        )
        
        if filtered_products.empty:
            filtered_products = category_products  # Fallback to all category products
        
        # Score and rank products
        scored_products = self._score_products(filtered_products, requirements, needs, category)
        
        # Select top products based on quantity needed
        quantity_needed = needs.get('quantity', 1)
        selected = scored_products.head(quantity_needed)
        
        return selected.to_dict('records')

    def _apply_intelligent_filters(self, products, requirements, needs, category):
        """Apply intelligent filters based on use cases, compatibility, and tier"""
        
        filtered = products.copy()
        audience_size = requirements['audience_size']
        
        # Filter by use case tags
        if 'use_case_tags' in filtered.columns:
            room_type = self._determine_room_type(requirements)
            use_case_filter = filtered['use_case_tags'].str.contains(
                room_type.replace('_', '_'), case=False, na=False
            )
            if use_case_filter.any():
                filtered = filtered[use_case_filter]
        
        # Filter by tier based on budget and requirements
        if 'tier' in filtered.columns:
            budget_range = requirements.get('budget_range', '')
            if budget_range in ['< $10,000', '$10,000 - $50,000']:
                # Prefer Economy and Standard for lower budgets
                tier_filter = filtered['tier'].isin(['Economy', 'Standard'])
                if tier_filter.any():
                    filtered = filtered[tier_filter]
            
        # Category-specific filtering
        if category == 'Displays & Projectors':
            # Filter by room size appropriateness
            room_info = self.room_equipment_logic.get(self._determine_room_type(requirements), {})
            display_size = room_info.get('display_size', '')
            
            if display_size and 'name' in filtered.columns:
                # Extract size from product names (looking for inch measurements)
                try:
                    size_matches = filtered['name'].str.extract(r'(\d+)"', expand=False).astype(float, errors='ignore')
                    if not size_matches.isna().all():
                        # Filter based on appropriate size ranges
                        if '32-55' in display_size:
                            filtered = filtered[size_matches.between(32, 55, na=True)]
                        elif '55-65' in display_size:
                            filtered = filtered[size_matches.between(55, 65, na=True)]
                        elif '65-75' in display_size:
                            filtered = filtered[size_matches.between(65, 75, na=True)]
                        elif '75-86' in display_size:
                            filtered = filtered[size_matches.between(75, 86, na=True)]
                except:
                    pass  # If size extraction fails, continue without filtering

        return filtered

    def _score_products(self, products, requirements, needs, category):
        """Score products based on multiple criteria"""
        
        if products.empty:
            return products
            
        scored = products.copy()
        scored['score'] = 0
        
        # Price scoring (sweet spot preference) - with error handling
        if 'price' in scored.columns and not scored['price'].isna().all():
            try:
                # Ensure price column is numeric
                scored['price'] = pd.to_numeric(scored['price'], errors='coerce').fillna(0)
                
                if scored['price'].sum() > 0:  # Only score if we have valid prices
                    price_median = scored['price'].median()
                    if price_median > 0:
                        price_scores = 100 - abs(scored['price'] - price_median) / price_median * 50
                        scored['score'] += price_scores.clip(0, 100) * 0.3
            except Exception as e:
                st.warning(f"Price scoring error: {str(e)}")
        
        # Tier scoring
        if 'tier' in scored.columns:
            tier_scores = scored['tier'].map({'Economy': 60, 'Standard': 100, 'Premium': 80})
            scored['score'] += tier_scores.fillna(50) * 0.2
        
        # Use case relevance scoring
        if 'use_case_tags' in scored.columns:
            room_type = self._determine_room_type(requirements)
            try:
                use_case_scores = scored['use_case_tags'].str.count(room_type.replace('_', '|'), na=0) * 20
                scored['score'] += use_case_scores * 0.3
            except:
                pass
        
        # Brand preference (diversification)
        if 'brand' in scored.columns:
            try:
                brand_counts = scored['brand'].value_counts()
                brand_diversity_scores = scored['brand'].map(lambda x: 100 - brand_counts[x] * 5).clip(50, 100)
                scored['score'] += brand_diversity_scores * 0.1
            except:
                pass
        
        # Feature richness scoring
        if 'features' in scored.columns:
            try:
                feature_scores = scored['features'].str.len().fillna(0) / 10
                scored['score'] += feature_scores.clip(0, 20) * 0.1
            except:
                pass
        
        return scored.sort_values('score', ascending=False)

    def _create_smart_boq_item(self, product, needs, requirements):
        """Create BOQ item with smart quantity calculation"""
        
        base_quantity = needs.get('quantity', 1)
        category = product.get('category', '')
        
        # Smart quantity adjustment based on category and room size
        audience_size = requirements['audience_size']
        
        if category == 'Cables & Connectivity':
            # Cables need more quantity
            quantity = max(base_quantity, audience_size // 5 + 5)
        elif category == 'Audio Systems':
            # Audio scales with room size
            quantity = max(base_quantity, audience_size // 15 + 1)
        else:
            quantity = base_quantity
            
        # Ensure price is numeric
        unit_cost = 0.0
        try:
            unit_cost = float(product.get('price', 0))
        except (ValueError, TypeError):
            unit_cost = 0.0
        
        return {
            'Category': product.get('category', 'Unknown'),
            'Brand': product.get('brand', 'TBD'),
            'Description': product.get('name', 'Product Description'),
            'Model No': str(product.get('name', '')).split()[-1] if product.get('name') else 'TBD',
            'Quantity': quantity,
            'Unit': self._get_unit_for_category(category),
            'Unit Cost': unit_cost,
            'Total Cost': round(quantity * unit_cost, 2),
            'Tier': product.get('tier', 'Standard'),
            'Features': str(product.get('features', 'Standard features'))[:100] + '...' if len(str(product.get('features', ''))) > 100 else str(product.get('features', ''))
        }

    def _get_unit_for_category(self, category):
        """Get appropriate unit for category"""
        unit_map = {
            'Cables & Connectivity': 'Set',
            'Displays & Projectors': 'Each', 
            'Video Conferencing': 'Each',
            'Audio Systems': 'Each',
            'Control Systems': 'Each',
            'Mounts & Racks': 'Each',
            'Installation & Services': 'Lump Sum'
        }
        return unit_map.get(category, 'Each')

    def _add_complementary_items(self, requirements, selected_products):
        """Add complementary items based on selected products"""
        complementary = []
        
        # Check if we have video conferencing but no control system
        has_video = any('Video Conferencing' in item['Category'] for item in selected_products)
        has_control = any('Control' in item['Category'] for item in selected_products)
        
        if has_video and not has_control and requirements['audience_size'] > 8:
            # Add a basic control solution
            complementary.append({
                'Category': 'Control Systems',
                'Brand': 'Generic',
                'Description': 'Basic Room Control Solution',
                'Model No': 'CTRL-BASIC',
                'Quantity': 1,
                'Unit': 'Each',
                'Unit Cost': 800.00,
                'Total Cost': 800.00,
                'Tier': 'Standard',
                'Features': 'Volume control, source switching, room controls'
            })
        
        return complementary

    def _calculate_installation_cost(self, products, requirements):
        """Calculate installation cost based on complexity"""
        equipment_cost = sum(item['Total Cost'] for item in products 
                           if item['Category'] not in ['Services', 'Project Management'])
        
        # Base installation percentage
        install_percentage = 0.15
        
        # Adjust based on complexity
        if requirements['audience_size'] > 50:
            install_percentage += 0.05
        if len([p for p in products if 'Video Conferencing' in p['Category']]) > 0:
            install_percentage += 0.03
        if len([p for p in products if 'Control' in p['Category']]) > 0:
            install_percentage += 0.02
            
        return round(equipment_cost * install_percentage, 2)

    def _get_budget_constraints(self, requirements):
        """Get budget constraints"""
        budget_map = {
            "< $10,000": 10000,
            "$10,000 - $50,000": 50000, 
            "$50,000 - $100,000": 100000,
            "> $100,000": 250000
        }
        return budget_map.get(requirements.get('budget_range', ''), 50000)

    def _get_budget_max(self, budget_range):
        """Get maximum budget"""
        return self._get_budget_constraints({'budget_range': budget_range})

# --- Data Loading Functions ---
def load_product_data():
    """Load product data from JSON file"""
    try:
        if os.path.exists('data/products.json'):
            with open('data/products.json', 'r') as f:
                data = json.load(f)
                return data
        else:
            return []
    except Exception as e:
        st.error(f"Error loading product data: {str(e)}")
        return []

def save_product_data(data):
    """Save product data to JSON file"""
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/products.json', 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving product data: {str(e)}")
        return False

# --- Main BOQ Generator Page ---
def show_intelligent_boq_page():
    st.header("🧠 Intelligent BOQ Generator")
    
    products_data = load_product_data()
    
    if not products_data:
        st.warning("⚠️ No product data loaded. Please upload your data first.")
        
        # Quick upload section
        st.subheader("📤 Quick Data Upload")
        uploaded_file = st.file_uploader("Upload your CSV data", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} products!")
                
                # Preview the data
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10))
                
                if st.button("💾 Save Data", type="primary"):
                    data_list = df.to_dict('records')
                    if save_product_data(data_list):
                        st.success("✅ Data saved successfully!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
        return
    
    # Initialize the intelligent generator
    try:
        generator = IntelligentBOQGenerator(products_data)
        analysis = generator.analyze_data_structure()
    except Exception as e:
        st.error(f"Error initializing generator: {str(e)}")
        return
    
    # Show data analysis
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", analysis['total_products'])
    with col2:
        st.metric("Categories", len(analysis['categories']))
    with col3:
        st.metric("Brands", len(analysis['brands']))
    with col4:
        st.metric("Use Cases", len(analysis['use_cases']))
    
    # Smart BOQ Generation Form
    with st.form("intelligent_boq_form"):
        st.subheader("🎯 Project Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input("Project Name*", placeholder="Executive Boardroom AV System")
            client_name = st.text_input("Client Name*", placeholder="Corporate Client Inc.")
            audience_size = st.number_input("Room Capacity*", min_value=1, max_value=500, value=12)
            
        with col2:
            project_type = st.selectbox("Room Type*", [
                "Conference Room", "Boardroom", "Huddle Room", 
                "Training Room", "Auditorium", "Multi-Purpose Room"
            ])
            budget_range = st.selectbox("Budget Range*", [
                "< $10,000", "$10,000 - $50,000", 
                "$50,000 - $100,000", "> $100,000"
            ])
            priority = st.selectbox("Quality Tier Preference", [
                "Economy (Cost-focused)", "Standard (Balanced)", 
                "Premium (Quality-focused)", "Mixed (Best Value)"
            ])
        
        st.subheader("🛠️ Required Systems")
        
        # Dynamic requirements based on available categories
        available_systems = []
        if any('display' in cat.lower() or 'projector' in cat.lower() for cat in analysis['categories']):
            available_systems.append("Display & Presentation")
        if any('video' in cat.lower() or 'conferencing' in cat.lower() for cat in analysis['categories']):
            available_systems.append("Video Conferencing")
        if any('audio' in cat.lower() or 'speaker' in cat.lower() for cat in analysis['categories']):
            available_systems.append("Audio Systems")
        if any('control' in cat.lower() for cat in analysis['categories']):
            available_systems.append("Room Control")
        
        # Add generic options
        available_systems.extend(["Connectivity & Cables", "Professional Installation"])
        
        requirements = st.multiselect("Select Required Systems*", available_systems)
        
        special_notes = st.text_area("Special Requirements", 
                                   placeholder="Integration with existing systems, specific brand preferences, accessibility needs...")
        
        submitted = st.form_submit_button("🚀 Generate Intelligent BOQ", type="primary")
        
        if submitted:
            if not project_name or not client_name or not requirements:
                st.error("❌ Please fill in all required fields")
            else:
                boq_requirements = {
                    'project_name': project_name,
                    'client_name': client_name,
                    'project_type': project_type,
                    'requirements': requirements,
                    'audience_size': audience_size,
                    'budget_range': budget_range,
                    'priority': priority,
                    'special_notes': special_notes
                }
                
                try:
                    with st.spinner("🤖 AI is analyzing your requirements and generating optimal BOQ..."):
                        boq_data = generator.generate_smart_boq(boq_requirements)
                        st.session_state['current_boq'] = boq_data
                        st.success("✅ Intelligent BOQ Generated!")
                        st.balloons()
                except Exception as e:
                    st.error(f"Error generating BOQ: {str(e)}")

    # Display Generated BOQ
    if 'current_boq' in st.session_state:
        boq = st.session_state['current_boq']
        
        st.markdown("---")
        st.header(f"📋 {boq['project_info']['name']} - BOQ")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", boq['analysis']['total_items'])
        with col2:
            st.metric("Equipment Cost", f"${boq['costs']['equipment_subtotal']:,.2f}")
        with col3:
            st.metric("Total Investment", f"${boq['costs']['total']:,.2f}")
        with col4:
            st.metric("Budget Efficiency", f"{boq['analysis']['budget_utilization']:.1f}%")
        
        # Analysis insights
        st.subheader("🔍 AI Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Room Classification**: {boq['analysis']['room_classification'].replace('_', ' ').title()}
            
            **Equipment Categories**: {boq['analysis']['equipment_categories']}
            
            **Budget Utilization**: {boq['analysis']['budget_utilization']:.1f}%
            """)
        
        with col2:
            if boq['analysis']['budget_utilization'] < 80:
                st.success("💚 **Efficient Design** - Great value within budget")
            elif boq['analysis']['budget_utilization'] < 95:
                st.warning("⚡ **Optimized Design** - Well-utilized budget")
            else:
                st.error("🔴 **Over Budget** - Consider adjusting requirements")
        
       # Detailed BOQ Table (continuation from where it was cut off)
        df_boq = pd.DataFrame(boq['items'])
        
        # Format the dataframe for better display
        df_display = df_boq.copy()
        df_display['Unit Cost'] = df_display['Unit Cost'].apply(lambda x: f"${x:,.2f}")
        df_display['Total Cost'] = df_display['Total Cost'].apply(lambda x: f"${x:,.2f}")
        
        # Display the table
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Brand": st.column_config.TextColumn("Brand", width="small"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Model No": st.column_config.TextColumn("Model No", width="small"),
                "Quantity": st.column_config.NumberColumn("Qty", width="small"),
                "Unit": st.column_config.TextColumn("Unit", width="small"),
                "Unit Cost": st.column_config.TextColumn("Unit Cost", width="small"),
                "Total Cost": st.column_config.TextColumn("Total Cost", width="small"),
                "Tier": st.column_config.TextColumn("Tier", width="small"),
                "Features": st.column_config.TextColumn("Features", width="large")
            }
        )
        
        # Cost Breakdown
        st.subheader("💰 Cost Breakdown")
        
        # Create cost breakdown chart
        cost_data = {
            'Component': ['Equipment', 'Installation', 'Contingency'],
            'Amount': [boq['costs']['equipment_subtotal'], 
                      boq['costs']['installation'], 
                      boq['costs']['contingency']]
        }
        
        fig = px.pie(cost_data, values='Amount', names='Component', 
                    title='Cost Distribution',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Category-wise cost breakdown
        category_costs = df_boq.groupby('Category')['Total Cost'].sum().reset_index()
        category_costs = category_costs.sort_values('Total Cost', ascending=True)
        
        fig_bar = px.bar(category_costs, x='Total Cost', y='Category', 
                        orientation='h', title='Cost by Category',
                        color='Total Cost', color_continuous_scale='viridis')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Export options
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as CSV
            csv_data = df_boq.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"{boq['project_info']['name']}_BOQ_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export as JSON
            json_data = json.dumps(boq, indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"{boq['project_info']['name']}_BOQ_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with col3:
            # Generate PDF Report button
            if st.button("📄 Generate PDF Report"):
                st.info("PDF generation feature can be implemented using libraries like reportlab or weasyprint")
        
        # Project Summary
        st.subheader("📋 Project Summary")
        
        summary_info = f"""
        **Project**: {boq['project_info']['name']}
        **Client**: {boq['project_info']['client']}
        **Room Type**: {boq['project_info']['room_type'].replace('_', ' ').title()}
        **Capacity**: {boq['project_info']['audience_size']} people
        **Generated**: {boq['project_info']['generated_date']}
        
        **Equipment Total**: ${boq['costs']['equipment_subtotal']:,.2f}
        **Installation**: ${boq['costs']['installation']:,.2f}
        **Contingency**: ${boq['costs']['contingency']:,.2f}
        **Grand Total**: ${boq['costs']['total']:,.2f}
        """
        
        st.markdown(summary_info)
        
        # Recommendations
        st.subheader("💡 AI Recommendations")
        
        recommendations = []
        
        # Budget-based recommendations
        if boq['analysis']['budget_utilization'] < 70:
            recommendations.append("💰 Consider upgrading to premium tier components for better performance and longevity")
        elif boq['analysis']['budget_utilization'] > 95:
            recommendations.append("⚠️ Budget is tight - consider phasing the project or reducing scope")
        
        # Room size recommendations
        if boq['project_info']['audience_size'] > 25:
            recommendations.append("🔊 Large room detected - ensure adequate audio coverage with multiple speakers")
        
        # Category-specific recommendations
        has_video_conf = any('Video Conferencing' in item['Category'] for item in boq['items'])
        has_control = any('Control' in item['Category'] for item in boq['items'])
        
        if has_video_conf and not has_control:
            recommendations.append("🎮 Consider adding room control system for better user experience")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ Well-balanced configuration with no major concerns")


# --- Additional Helper Functions ---
def show_data_management():
    """Data management page for uploading and managing product data"""
    st.header("📊 Data Management")
    
    # Load existing data
    products_data = load_product_data()
    
    if products_data:
        st.success(f"✅ Currently loaded: {len(products_data)} products")
        
        # Show data preview
        df = pd.DataFrame(products_data)
        st.subheader("📋 Current Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", len(df))
        with col2:
            st.metric("Categories", df['category'].nunique() if 'category' in df.columns else 0)
        with col3:
            st.metric("Brands", df['brand'].nunique() if 'brand' in df.columns else 0)
    else:
        st.info("No product data currently loaded")
    
    st.markdown("---")
    
    # Upload new data
    st.subheader("📤 Upload Product Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file with product information including columns like: name, category, brand, price, tier, features, use_case_tags"
    )
    
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(new_df)} products from CSV")
            
            # Show preview of new data
            st.subheader("📊 New Data Preview")
            st.dataframe(new_df.head(10), use_container_width=True)
            
            # Show column mapping
            st.subheader("🔄 Column Mapping")
            st.info("Make sure your CSV has these columns: name, category, brand, price, tier, features, use_case_tags")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Available Columns:**")
                st.write(list(new_df.columns))
            
            with col2:
                st.write("**Expected Columns:**")
                st.write(["name", "category", "brand", "price", "tier", "features", "use_case_tags"])
            
            # Save options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Replace Existing Data", type="primary"):
                    data_list = new_df.to_dict('records')
                    if save_product_data(data_list):
                        st.success("✅ Data replaced successfully!")
                        st.rerun()
            
            with col2:
                if st.button("➕ Append to Existing Data"):
                    if products_data:
                        combined_df = pd.concat([pd.DataFrame(products_data), new_df], ignore_index=True)
                        data_list = combined_df.to_dict('records')
                    else:
                        data_list = new_df.to_dict('records')
                    
                    if save_product_data(data_list):
                        st.success("✅ Data appended successfully!")
                        st.rerun()
                        
        except Exception as e:
            st.error(f"❌ Error loading CSV file: {str(e)}")


# --- Main App ---
def main():
    st.title("🏗️ Intelligent BOQ Generator")
    st.markdown("*AI-Powered Bill of Quantities for AV Systems*")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🧠 BOQ Generator", "📊 Data Management", "ℹ️ About"]
    )
    
    if page == "🧠 BOQ Generator":
        show_intelligent_boq_page()
    elif page == "📊 Data Management":
        show_data_management()
    elif page == "ℹ️ About":
        st.header("ℹ️ About")
        st.markdown("""
        ## Intelligent BOQ Generator
        
        This application uses AI to generate optimized Bill of Quantities (BOQ) for AV systems based on:
        
        - **Room Analysis**: Automatically classifies rooms and determines optimal equipment
        - **Smart Selection**: Uses multiple criteria to select the best products
        - **Budget Optimization**: Ensures efficient use of budget while meeting requirements
        - **Industry Experience**: Built-in logic based on AV industry best practices
        
        ### Features:
        - 🎯 Intelligent product selection based on room size and use case
        - 💰 Budget optimization and cost analysis
        - 📊 Visual analytics and reporting
        - 📤 Multiple export formats (CSV, JSON)
        - 🔄 Easy data management
        
        ### How to Use:
        1. Upload your product database via **Data Management**
        2. Fill in project requirements in **BOQ Generator**
        3. Review the AI-generated BOQ and analysis
        4. Export results for your project
        """)


if __name__ == "__main__":
    main()aFrame(boq['items
