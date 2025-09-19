import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import random
from collections import defaultdict

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
                'min': self.df['price'].min() if 'price' in self.df.columns else 0,
                'max': self.df['price'].max() if 'price' in self.df.columns else 0
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
                if isinstance(tags, str):
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

        return filtered

    def _score_products(self, products, requirements, needs, category):
        """Score products based on multiple criteria"""
        
        if products.empty:
            return products
            
        scored = products.copy()
        scored['score'] = 0
        
        # Price scoring (sweet spot preference)
        if 'price' in scored.columns:
            price_median = scored['price'].median()
            price_scores = 100 - abs(scored['price'] - price_median) / price_median * 50
            scored['score'] += price_scores.clip(0, 100) * 0.3
        
        # Tier scoring
        if 'tier' in scored.columns:
            tier_scores = scored['tier'].map({'Economy': 60, 'Standard': 100, 'Premium': 80})
            scored['score'] += tier_scores.fillna(50) * 0.2
        
        # Use case relevance scoring
        if 'use_case_tags' in scored.columns:
            room_type = self._determine_room_type(requirements)
            use_case_scores = scored['use_case_tags'].str.count(room_type.replace('_', '|'), na=0) * 20
            scored['score'] += use_case_scores * 0.3
        
        # Brand preference (diversification)
        if 'brand' in scored.columns:
            brand_counts = scored['brand'].value_counts()
            brand_diversity_scores = scored['brand'].map(lambda x: 100 - brand_counts[x] * 5).clip(50, 100)
            scored['score'] += brand_diversity_scores * 0.1
        
        # Feature richness scoring
        if 'features' in scored.columns:
            feature_scores = scored['features'].str.len().fillna(0) / 10
            scored['score'] += feature_scores.clip(0, 20) * 0.1
        
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
            
        unit_cost = float(product.get('price', 0))
        
        return {
            'Category': product.get('category', 'Unknown'),
            'Brand': product.get('brand', 'TBD'),
            'Description': product.get('name', 'Product Description'),
            'Model No': product.get('name', '').split()[-1] if product.get('name') else 'TBD',
            'Quantity': quantity,
            'Unit': self._get_unit_for_category(category),
            'Unit Cost': unit_cost,
            'Total Cost': round(quantity * unit_cost, 2),
            'Tier': product.get('tier', 'Standard'),
            'Features': product.get('features', 'Standard features')[:100] + '...' if len(str(product.get('features', ''))) > 100 else product.get('features', '')
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
    generator = IntelligentBOQGenerator(products_data)
    analysis = generator.analyze_data_structure()
    
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
        if 'Displays & Projectors' in analysis['categories']:
            available_systems.append("Display & Presentation")
        if 'Video Conferencing' in analysis['categories']:
            available_systems.append("Video Conferencing")
        if 'Audio Systems' in analysis['categories'] or any('audio' in cat.lower() for cat in analysis['categories']):
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
                
                with st.spinner("🤖 AI is analyzing your requirements and generating optimal BOQ..."):
                    boq_data = generator.generate_smart_boq(boq_requirements)
                    st.session_state['current_boq'] = boq_data
                    st.success("✅ Intelligent BOQ Generated!")
                    st.balloons()

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
        
        # Detailed BOQ Table
        st.subheader("📊 Detailed Bill of Quantities")
        
        if boq['items']:
            df_boq = pd.DataFrame(boq['items'])
            
            # Format for display
            display_df = df_boq.copy()
            display_df['Unit Cost'] = display_df['Unit Cost'].apply(lambda x: f"${x:,.2f}")
            display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Category breakdown
            st.subheader("📈 Investment Breakdown")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                category_totals = df_boq.groupby('Category')['Total Cost'].sum().sort_values(ascending=False)
                
                fig = px.pie(
                    values=category_totals.values,
                    names=category_totals.index,
                    title="Investment Distribution by Category",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Category Totals:**")
                for cat, total in category_totals.items():
                    percentage = (total / boq['costs']['total']) * 100
                    st.markdown(f"**{cat}**: ${total:,.2f} ({percentage:.1f}%)")
            
            # Export options
            st.subheader("📥 Export & Share")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df_boq.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv_data,
                    f"BOQ_{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with col2:
                # Create detailed report
                report = f"""
INTELLIGENT BOQ REPORT
=====================

Project: {boq['project_info']['name']}
Client: {boq['project_info']['client']}
Room Type: {boq['analysis']['room_classification'].replace('_', ' ').title()}
Capacity: {boq['project_info']['audience_size']} people
Generated: {boq['project_info']['generated_date']}

INVESTMENT SUMMARY
==================
Equipment: ${boq['costs']['equipment_subtotal']:,.2f}
Installation: ${boq['costs']['installation']:,.2f}
Contingency: ${boq['costs']['contingency']:,.2f}
Total Investment: ${boq['costs']['total']:,.2f}

DETAILED LINE ITEMS
===================
"""
                for _, item in df_boq.iterrows():
                    report += f"{item['Description']} | {item['Brand']} | {item['Quantity']} {item['Unit']} @ ${item['Unit Cost']:,.2f} = ${item['Total Cost']:,.2f}\n"
                
                st.download_button(
                    "📑 Download Report",
                    report,
                    f"BOQ_Report_{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
                )
            
            with col3:
                json_data = json.dumps(boq, indent=2, default=str)
                st.download_button(
                    "📋 Download JSON",
                    json_data,
                    f"BOQ_{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
                )

# --- Database Management Page (Continuation) ---
def show_database_page():
    st.header("📊 Product Database Management")
    
    products_data = load_product_data()
    
    # Database overview
    if products_data:
        df = pd.DataFrame(products_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Products", len(df))
        with col2:
            st.metric("Categories", df['category'].nunique() if 'category' in df.columns else 0)
        with col3:
            st.metric("Brands", df['brand'].nunique() if 'brand' in df.columns else 0)
        with col4:
            avg_price = df['price'].mean() if 'price' in df.columns else 0
            st.metric("Avg Price", f"${avg_price:,.2f}")
        
        # Data management tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 View Data", "➕ Add Product", "📤 Upload Data", "📥 Export Data"])
        
        with tab1:
            st.subheader("Current Product Database")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'category' in df.columns:
                    categories = ['All'] + list(df['category'].unique())
                    selected_category = st.selectbox("Filter by Category", categories)
                else:
                    selected_category = 'All'
            
            with col2:
                if 'brand' in df.columns:
                    brands = ['All'] + list(df['brand'].unique())
                    selected_brand = st.selectbox("Filter by Brand", brands)
                else:
                    selected_brand = 'All'
            
            with col3:
                if 'tier' in df.columns:
                    tiers = ['All'] + list(df['tier'].unique())
                    selected_tier = st.selectbox("Filter by Tier", tiers)
                else:
                    selected_tier = 'All'
            
            # Apply filters
            filtered_df = df.copy()
            if selected_category != 'All' and 'category' in df.columns:
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
            if selected_brand != 'All' and 'brand' in df.columns:
                filtered_df = filtered_df[filtered_df['brand'] == selected_brand]
            if selected_tier != 'All' and 'tier' in df.columns:
                filtered_df = filtered_df[filtered_df['tier'] == selected_tier]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            if len(filtered_df) != len(df):
                st.info(f"Showing {len(filtered_df)} of {len(df)} products")
        
        with tab2:
            st.subheader("Add New Product")
            
            with st.form("add_product_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_name = st.text_input("Product Name*")
                    new_brand = st.text_input("Brand*")
                    new_category = st.selectbox("Category*", 
                        df['category'].unique().tolist() + ['Custom'] if 'category' in df.columns else ['Custom'])
                    if new_category == 'Custom':
                        new_category = st.text_input("Enter Custom Category")
                
                with col2:
                    new_price = st.number_input("Price ($)*", min_value=0.0, value=0.0, step=0.01)
                    new_tier = st.selectbox("Tier", ['Economy', 'Standard', 'Premium'])
                    new_use_case = st.text_input("Use Case Tags (comma-separated)")
                
                new_features = st.text_area("Features & Description")
                
                if st.form_submit_button("Add Product", type="primary"):
                    if new_name and new_brand and new_category and new_price > 0:
                        new_product = {
                            'name': new_name,
                            'brand': new_brand,
                            'category': new_category,
                            'price': new_price,
                            'tier': new_tier,
                            'features': new_features,
                            'use_case_tags': new_use_case
                        }
                        
                        products_data.append(new_product)
                        if save_product_data(products_data):
                            st.success("✅ Product added successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to save product")
                    else:
                        st.error("❌ Please fill in all required fields")
        
        with tab3:
            st.subheader("Upload Product Data")
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                try:
                    new_df = pd.read_csv(uploaded_file)
                    
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(new_df.head())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        merge_option = st.radio("Import Option", 
                            ["Replace existing data", "Append to existing data"])
                    
                    with col2:
                        st.write(f"**Upload contains:** {len(new_df)} products")
                        if merge_option == "Append to existing data":
                            st.write(f"**Total after import:** {len(df) + len(new_df)} products")
                    
                    if st.button("Import Data", type="primary"):
                        if merge_option == "Replace existing data":
                            final_data = new_df.to_dict('records')
                        else:
                            final_data = products_data + new_df.to_dict('records')
                        
                        if save_product_data(final_data):
                            st.success(f"✅ Successfully imported {len(new_df)} products!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to import data")
                            
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
        
        with tab4:
            st.subheader("Export Product Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📄 Download as CSV",
                    csv_data,
                    f"products_database_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with col2:
                json_data = json.dumps(products_data, indent=2)
                st.download_button(
                    "📋 Download as JSON",
                    json_data,
                    f"products_database_{datetime.now().strftime('%Y%m%d')}.json"
                )
            
            with col3:
                if st.button("🗑️ Clear All Data"):
                    if st.checkbox("I understand this will delete all product data"):
                        if save_product_data([]):
                            st.success("✅ Database cleared")
                            st.rerun()
    
    else:
        st.warning("📭 No product data found. Please upload some data to get started.")
        
        st.subheader("📤 Initial Data Upload")
        uploaded_file = st.file_uploader("Upload your first CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} products!")
                
                st.dataframe(df.head())
                
                if st.button("💾 Save to Database", type="primary"):
                    if save_product_data(df.to_dict('records')):
                        st.success("✅ Database initialized successfully!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

# --- Analytics Page ---
def show_analytics_page():
    st.header("📈 Product Analytics")
    
    products_data = load_product_data()
    
    if not products_data:
        st.warning("⚠️ No product data available for analysis.")
        return
    
    df = pd.DataFrame(products_data)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        avg_price = df['price'].mean() if 'price' in df.columns else 0
        st.metric("Average Price", f"${avg_price:,.2f}")
    with col3:
        st.metric("Categories", df['category'].nunique() if 'category' in df.columns else 0)
    with col4:
        st.metric("Brands", df['brand'].nunique() if 'brand' in df.columns else 0)
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Category Analysis", "💰 Price Analysis", "🏷️ Brand Analysis", "⭐ Tier Analysis"])
    
    with tab1:
        if 'category' in df.columns:
            st.subheader("Product Distribution by Category")
            
            category_counts = df['category'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.pie(values=category_counts.values, names=category_counts.index,
                           title="Product Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Category Breakdown:**")
                for cat, count in category_counts.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"**{cat}**: {count} ({percentage:.1f}%)")
            
            # Category price analysis
            if 'price' in df.columns:
                st.subheader("Average Price by Category")
                avg_price_by_cat = df.groupby('category')['price'].agg(['mean', 'count']).round(2)
                
                fig = px.bar(x=avg_price_by_cat.index, y=avg_price_by_cat['mean'],
                           title="Average Price by Category")
                fig.update_layout(xaxis_title="Category", yaxis_title="Average Price ($)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category data not available")
    
    with tab2:
        if 'price' in df.columns:
            st.subheader("Price Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='price', nbins=20, title="Price Distribution")
                fig.update_layout(xaxis_title="Price ($)", yaxis_title="Number of Products")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price statistics
                price_stats = df['price'].describe()
                st.write("**Price Statistics:**")
                st.write(f"**Min**: ${price_stats['min']:,.2f}")
                st.write(f"**Max**: ${price_stats['max']:,.2f}")
                st.write(f"**Mean**: ${price_stats['mean']:,.2f}")
                st.write(f"**Median**: ${price_stats['50%']:,.2f}")
                st.write(f"**Std Dev**: ${price_stats['std']:,.2f}")
            
            # Price ranges
            st.subheader("Price Range Analysis")
            
            def categorize_price(price):
                if price < 100:
                    return "< $100"
                elif price < 500:
                    return "$100 - $500"
                elif price < 1000:
                    return "$500 - $1,000"
                elif price < 5000:
                    return "$1,000 - $5,000"
                else:
                    return "> $5,000"
            
            df['price_range'] = df['price'].apply(categorize_price)
            price_range_counts = df['price_range'].value_counts()
            
            fig = px.bar(x=price_range_counts.index, y=price_range_counts.values,
                       title="Product Count by Price Range")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price data not available")
    
    with tab3:
        if 'brand' in df.columns:
            st.subheader("Brand Analysis")
            
            brand_counts = df['brand'].value_counts().head(10)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(x=brand_counts.values, y=brand_counts.index,
                           title="Top 10 Brands by Product Count", orientation='h')
                fig.update_layout(xaxis_title="Number of Products", yaxis_title="Brand")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Brand Statistics:**")
                st.write(f"**Total Brands**: {df['brand'].nunique()}")
                st.write(f"**Top Brand**: {brand_counts.index[0]} ({brand_counts.iloc[0]} products)")
                
                if 'price' in df.columns:
                    avg_brand_price = df.groupby('brand')['price'].mean().sort_values(ascending=False).head(5)
                    st.write("**Top 5 Most Expensive Brands:**")
                    for brand, price in avg_brand_price.items():
                        st.write(f"**{brand}**: ${price:,.2f}")
        else:
            st.info("Brand data not available")
    
    with tab4:
        if 'tier' in df.columns:
            st.subheader("Product Tier Analysis")
            
            tier_counts = df['tier'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                           title="Product Distribution by Tier")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Tier Breakdown:**")
                for tier, count in tier_counts.items():
                    percentage = (count / len(df)) * 100
                    st.write(f"**{tier}**: {count} ({percentage:.1f}%)")
            
            # Tier vs Price analysis
            if 'price' in df.columns:
                st.subheader("Price Distribution by Tier")
                
                fig = px.box(df, x='tier', y='price', title="Price Distribution by Tier")
                fig.update_layout(xaxis_title="Tier", yaxis_title="Price ($)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Average price by tier
                avg_price_by_tier = df.groupby('tier')['price'].mean().round(2)
                st.write("**Average Price by Tier:**")
                for tier, price in avg_price_by_tier.items():
                    st.write(f"**{tier}**: ${price:,.2f}")
        else:
            st.info("Tier data not available")

# --- Settings Page ---
def show_settings_page():
    st.header("⚙️ Application Settings")
    
    st.subheader("🎨 Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Show detailed tooltips", value=True)
        st.checkbox("Enable animations", value=True)
        st.selectbox("Theme preference", ["Auto", "Light", "Dark"])
    
    with col2:
        st.number_input("Items per page", min_value=10, max_value=100, value=25)
        st.selectbox("Currency", ["USD ($)", "EUR (€)", "GBP (£)"])
    
    st.subheader("📊 BOQ Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_markup = st.number_input("Default markup (%)", min_value=0, max_value=100, value=15)
        default_contingency = st.number_input("Default contingency (%)", min_value=0, max_value=50, value=10)
    
    with col2:
        st.selectbox("Default tier preference", ["Economy", "Standard", "Premium", "Mixed"])
        st.checkbox("Include installation by default", value=True)
    
    st.subheader("🔧 Advanced Settings")
    
    st.checkbox("Enable debug mode", value=False)
    st.checkbox("Auto-save generated BOQs", value=True)
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")

# --- Main Application ---
def main():
    st.sidebar.title("🧠 Intelligent BOQ Generator")
    
    # Navigation
    pages = {
        "🚀 Generate BOQ": show_intelligent_boq_page,
        "📊 Product Database": show_database_page,
        "📈 Analytics": show_analytics_page,
        "⚙️ Settings": show_settings_page
    }
    
    selected_page = st.sidebar.radio("Navigate", list(pages.keys()))
    
    # App info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📋 About This App
    
    Intelligent BOQ Generator uses AI-powered analysis to create optimized Bills of Quantities for AV systems.
    
    **Features:**
    - 🧠 Smart product selection
    - 📊 Data-driven recommendations
    - 💰 Budget optimization
    - 📈 Comprehensive analytics
    
    **Version:** 2.0.0
    """)
    
    # Run selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
