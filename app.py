import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent BOQ Generator",
    page_icon="🧠",
    layout="wide"
)

# --- Core Logic Class ---
class IntelligentBOQGenerator:
    """
    Handles all the backend logic for analyzing product data and generating
    intelligent Bills of Quantities (BOQ).
    """

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
        """Analyze the uploaded data to understand its structure."""
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
        """Extract use cases from tags."""
        use_cases = set()
        if 'use_case_tags' in self.df.columns:
            for tags in self.df['use_case_tags'].dropna():
                if isinstance(tags, str):
                    use_cases.update([tag.strip() for tag in tags.split(',')])
        return list(use_cases)

    def generate_smart_boq(self, requirements):
        """Generate intelligent BOQ based on requirements and data analysis."""
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
        """Intelligently determine room type based on requirements."""
        audience_size = requirements['audience_size']
        project_type = requirements.get('project_type', '').lower()
        
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
        """Analyze what equipment is needed based on requirements."""
        needs = defaultdict(lambda: {'required': False, 'quantity': 0, 'priority': 'low'})
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
        """Smart product selection based on multiple criteria."""
        category_products = self.df[self.df['category'] == category].copy()
        if category_products.empty:
            return []

        # Apply intelligent filtering
        filtered_products = self._apply_intelligent_filters(
            category_products, requirements, needs, category
        )
        if filtered_products.empty:
            filtered_products = category_products  # Fallback

        # Score and rank products
        scored_products = self._score_products(filtered_products, requirements, needs, category)

        # Select top products based on quantity needed
        quantity_needed = needs.get('quantity', 1)
        selected = scored_products.head(quantity_needed)
        return selected.to_dict('records')

    def _apply_intelligent_filters(self, products, requirements, needs, category):
        """Apply intelligent filters based on use cases, compatibility, and tier."""
        filtered = products.copy()
        room_type = self._determine_room_type(requirements)

        # Filter by use case tags
        if 'use_case_tags' in filtered.columns:
            use_case_filter = filtered['use_case_tags'].str.contains(
                room_type.replace('_', ' '), case=False, na=False
            )
            if use_case_filter.any():
                filtered = filtered[use_case_filter]

        # Filter by tier based on budget
        if 'tier' in filtered.columns:
            budget_range = requirements.get('budget_range', '')
            if budget_range in ['< $10,000', '$10,000 - $50,000']:
                tier_filter = filtered['tier'].isin(['Economy', 'Standard'])
                if tier_filter.any():
                    filtered = filtered[tier_filter]

        # Category-specific filtering for display size
        if category == 'Displays & Projectors':
            room_info = self.room_equipment_logic.get(room_type, {})
            display_size = room_info.get('display_size', '')
            if display_size and 'name' in filtered.columns:
                size_matches = filtered['name'].str.extract(r'(\d+)"', expand=False).astype(float, errors='ignore')
                if not size_matches.isna().all():
                    if '32-55' in display_size:
                        filtered = filtered[size_matches.between(32, 55, inclusive='both')]
                    elif '55-65' in display_size:
                        filtered = filtered[size_matches.between(55, 65, inclusive='both')]
                    elif '65-75' in display_size:
                        filtered = filtered[size_matches.between(65, 75, inclusive='both')]
                    elif '75-86' in display_size:
                        filtered = filtered[size_matches.between(75, 86, inclusive='both')]
        return filtered

    def _score_products(self, products, requirements, needs, category):
        """Score products based on price, tier, use case, and other factors."""
        if products.empty:
            return products

        scored = products.copy()
        scored['score'] = 0

        # Price scoring (prefer median)
        if 'price' in scored.columns and scored['price'].median() > 0:
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

        # Brand diversity scoring
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
        """Create a BOQ item with intelligent quantity calculation."""
        base_quantity = needs.get('quantity', 1)
        category = product.get('category', '')
        audience_size = requirements['audience_size']
        quantity = base_quantity

        # Smart quantity adjustment
        if category == 'Cables & Connectivity':
            quantity = max(base_quantity, audience_size // 5 + 5)
        elif category == 'Audio Systems':
            quantity = max(base_quantity, audience_size // 15 + 1)

        unit_cost = float(product.get('price', 0))
        features = str(product.get('features', ''))

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
            'Features': (features[:100] + '...') if len(features) > 100 else features
        }

    def _get_unit_for_category(self, category):
        """Get the appropriate unit for a given product category."""
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
        """Add complementary items like control systems if needed."""
        complementary = []
        has_video = any('Video Conferencing' in item['Category'] for item in selected_products)
        has_control = any('Control' in item['Category'] for item in selected_products)

        if has_video and not has_control and requirements['audience_size'] > 8:
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
        """Calculate installation cost based on project complexity."""
        equipment_cost = sum(
            item['Total Cost'] for item in products
            if item['Category'] not in ['Services', 'Project Management']
        )
        install_percentage = 0.15  # Base percentage

        # Adjust for complexity
        if requirements['audience_size'] > 50:
            install_percentage += 0.05
        if any('Video Conferencing' in p['Category'] for p in products):
            install_percentage += 0.03
        if any('Control' in p['Category'] for p in products):
            install_percentage += 0.02

        return round(equipment_cost * install_percentage, 2)

    def _get_budget_constraints(self, requirements):
        """Get the maximum budget value from a range string."""
        budget_map = {
            "< $10,000": 10000,
            "$10,000 - $50,000": 50000,
            "$50,000 - $100,000": 100000,
            "> $100,000": 250000
        }
        return budget_map.get(requirements.get('budget_range', ''), 50000)

    def _get_budget_max(self, budget_range):
        """Helper to get max budget."""
        return self._get_budget_constraints({'budget_range': budget_range})

# --- Data Loading and Saving Functions ---
def load_product_data():
    """Load product data from the local JSON file."""
    try:
        if os.path.exists('data/products.json'):
            with open('data/products.json', 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading product data: {e}")
        return []

def save_product_data(data):
    """Save product data to the local JSON file."""
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/products.json', 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving product data: {e}")
        return False

# --- UI Page Functions ---
def show_intelligent_boq_page():
    """Displays the main BOQ generation interface."""
    st.header("🧠 Intelligent BOQ Generator")

    products_data = load_product_data()
    if not products_data:
        st.warning("⚠️ No product data found. Please upload a CSV on the 'Product Database' page.")
        return

    # Initialize generator and display data overview
    generator = IntelligentBOQGenerator(products_data)
    analysis = generator.analyze_data_structure()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", analysis['total_products'])
    with col2:
        st.metric("Categories", len(analysis['categories']))
    with col3:
        st.metric("Brands", len(analysis['brands']))
    with col4:
        st.metric("Use Cases", len(analysis['use_cases']))

    # Project requirements form
    with st.form("intelligent_boq_form"):
        st.subheader("🎯 Project Requirements")
        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("Project Name*", placeholder="Executive Boardroom AV Upgrade")
            client_name = st.text_input("Client Name*", placeholder="Global Corp Inc.")
            audience_size = st.number_input("Room Capacity*", min_value=1, max_value=500, value=12)
        with col2:
            project_type = st.selectbox("Room Type*", [
                "Conference Room", "Boardroom", "Huddle Room", "Training Room", "Auditorium"
            ])
            budget_range = st.selectbox("Budget Range*", [
                "< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"
            ])
            priority = st.selectbox("Quality Tier Preference", [
                "Economy (Cost-focused)", "Standard (Balanced)", "Premium (Quality-focused)"
            ])

        st.subheader("🛠️ Required Systems")
        available_systems = [
            "Display & Presentation", "Video Conferencing", "Audio Systems",
            "Room Control", "Connectivity & Cables"
        ]
        requirements = st.multiselect("Select Required Systems*", available_systems, default=available_systems[:3])
        special_notes = st.text_area("Special Requirements", placeholder="e.g., specific brand preferences, BYOD support...")
        
        submitted = st.form_submit_button("🚀 Generate Intelligent BOQ", type="primary")

    if submitted:
        if not all([project_name, client_name, requirements]):
            st.error("❌ Please fill in all required fields.")
        else:
            boq_requirements = {
                'project_name': project_name, 'client_name': client_name,
                'project_type': project_type, 'requirements': requirements,
                'audience_size': audience_size, 'budget_range': budget_range,
                'priority': priority, 'special_notes': special_notes
            }
            with st.spinner("🤖 AI is analyzing requirements and building your BOQ..."):
                boq_data = generator.generate_smart_boq(boq_requirements)
                st.session_state['current_boq'] = boq_data
            st.success("✅ Intelligent BOQ Generated!")
            st.balloons()

    # Display the generated BOQ
    if 'current_boq' in st.session_state:
        boq = st.session_state['current_boq']
        st.markdown("---")
        st.header(f"📋 BOQ: {boq['project_info']['name']}")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Investment", f"${boq['costs']['total']:,.2f}")
        with col2:
            st.metric("Equipment Cost", f"${boq['costs']['equipment_subtotal']:,.2f}")
        with col3:
            st.metric("Total Items", boq['analysis']['total_items'])
        with col4:
            st.metric("Budget Utilization", f"{boq['analysis']['budget_utilization']:.1f}%")

        # Detailed BOQ table
        st.subheader("📊 Detailed Bill of Quantities")
        if boq['items']:
            df_boq = pd.DataFrame(boq['items'])
            display_df = df_boq.copy()
            display_df['Unit Cost'] = display_df['Unit Cost'].apply(lambda x: f"${x:,.2f}")
            display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Investment breakdown chart
            st.subheader("📈 Investment Breakdown")
            col1, col2 = st.columns([2, 1])
            with col1:
                category_totals = df_boq.groupby('Category')['Total Cost'].sum().sort_values(ascending=False)
                fig = px.pie(
                    values=category_totals.values, names=category_totals.index,
                    title="Investment Distribution by Category",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Category Totals:**")
                for cat, total in category_totals.items():
                    percentage = (total / boq['costs']['total']) * 100
                    st.markdown(f"**{cat}**: ${total:,.2f} `({percentage:.1f}%)`")

            # Export options
            st.subheader("📥 Export & Share")
            col1, col2 = st.columns(2)
            with col1:
                csv_data = df_boq.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV", csv_data,
                    f"BOQ_{boq['project_info']['name'].replace(' ', '_')}.csv", "text/csv"
                )
            with col2:
                json_data = json.dumps(boq, indent=2, default=str)
                st.download_button(
                    "📋 Download JSON", json_data,
                    f"BOQ_{boq['project_info']['name'].replace(' ', '_')}.json", "application/json"
                )

def show_database_page():
    """Displays the interface for managing the product database."""
    st.header("📦 Product Database Management")
    
    products_data = load_product_data()

    if not products_data:
        st.warning("📭 No product data found. Upload a CSV file to get started.")
        uploaded_file = st.file_uploader("Upload your product database (CSV)", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                if st.button("💾 Save to Database", type="primary"):
                    if save_product_data(df.to_dict('records')):
                        st.success("✅ Database initialized successfully!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
        return

    df = pd.DataFrame(products_data)
    tab1, tab2, tab3 = st.tabs(["📋 View & Filter Data", "➕ Add/Upload Products", "📥 Export Data"])

    with tab1:
        st.subheader("Current Product Database")
        col1, col2, col3 = st.columns(3)
        with col1:
            categories = ['All'] + sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
        with col2:
            brands = ['All'] + sorted(df['brand'].unique().tolist())
            selected_brand = st.selectbox("Filter by Brand", brands)
        with col3:
            tiers = ['All'] + sorted(df['tier'].unique().tolist())
            selected_tier = st.selectbox("Filter by Tier", tiers)

        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if selected_brand != 'All':
            filtered_df = filtered_df[filtered_df['brand'] == selected_brand]
        if selected_tier != 'All':
            filtered_df = filtered_df[filtered_df['tier'] == selected_tier]
        
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        st.info(f"Showing {len(filtered_df)} of {len(df)} total products.")

    with tab2:
        st.subheader("➕ Add a New Product")
        with st.form("add_product_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Product Name*")
                new_brand = st.text_input("Brand*")
                new_category = st.selectbox("Category*", df['category'].unique().tolist())
            with col2:
                new_price = st.number_input("Price ($)*", min_value=0.0, step=1.00, format="%.2f")
                new_tier = st.selectbox("Tier*", ['Economy', 'Standard', 'Premium'])
                new_use_case = st.text_input("Use Case Tags (comma-separated)")
            new_features = st.text_area("Features")
            
            if st.form_submit_button("Add Product", type="primary"):
                if all([new_name, new_brand, new_category, new_price > 0]):
                    new_product = {
                        'name': new_name, 'brand': new_brand, 'category': new_category,
                        'price': new_price, 'tier': new_tier, 'features': new_features,
                        'use_case_tags': new_use_case
                    }
                    products_data.append(new_product)
                    if save_product_data(products_data):
                        st.success("✅ Product added successfully!")
                        st.rerun()
                else:
                    st.error("❌ Please fill in all required fields.")

        st.markdown("---")
        st.subheader("📤 Upload from CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="db_upload")
        if uploaded_file:
            new_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(new_df.head())
            
            merge_option = st.radio("Import Option", ["Append to existing data", "Replace existing data"])
            if st.button("Import Data"):
                final_data = products_data + new_df.to_dict('records') if merge_option == "Append to existing data" else new_df.to_dict('records')
                if save_product_data(final_data):
                    st.success("✅ Data imported successfully!")
                    st.rerun()

    # CORRECTED BLOCK: Changed 'with tab4:' to 'with tab3:'
    with tab3:
        st.subheader("Export Product Data")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download as CSV", df.to_csv(index=False),
                f"products_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv"
            )
        with col2:
            st.download_button(
                "📋 Download as JSON", json.dumps(products_data, indent=2),
                f"products_{datetime.now().strftime('%Y%m%d')}.json", "application/json"
            )

def show_analytics_page():
    """Displays analytics and visualizations of the product data."""
    st.header("📈 Product Analytics")
    
    products_data = load_product_data()
    if not products_data:
        st.warning("⚠️ No product data available for analysis.")
        return
        
    df = pd.DataFrame(products_data)

    tab1, tab2, tab3 = st.tabs(["📊 Category Analysis", "💰 Price Analysis", "🏷️ Brand & Tier Analysis"])

    with tab1:
        st.subheader("Product Distribution by Category")
        category_counts = df['category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index, title="Product Count by Category")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Average Price by Category")
        avg_price_by_cat = df.groupby('category')['price'].mean().round(2).sort_values(ascending=False)
        fig2 = px.bar(avg_price_by_cat, x=avg_price_by_cat.index, y=avg_price_by_cat.values, title="Average Price per Category")
        fig2.update_layout(xaxis_title="Category", yaxis_title="Average Price ($)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Product Price Distribution")
        fig = px.histogram(df, x='price', nbins=30, title="Price Distribution of All Products")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Price Distribution by Tier")
        fig2 = px.box(df, x='tier', y='price', title="Price Spread per Tier", points="all")
        fig2.update_layout(xaxis_title="Tier", yaxis_title="Price ($)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Top 10 Brands by Product Count")
        brand_counts = df['brand'].value_counts().head(10)
        fig = px.bar(brand_counts, y=brand_counts.index, x=brand_counts.values, orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Product Distribution by Tier")
        tier_counts = df['tier'].value_counts()
        fig2 = px.pie(values=tier_counts.values, names=tier_counts.index, title="Product Count by Tier")
        st.plotly_chart(fig2, use_container_width=True)

# --- Main Application Runner ---
def main():
    """Main function to run the Streamlit app."""
    st.sidebar.title("Intelligent BOQ Generator")
    st.sidebar.markdown("---")
    
    pages = {
        "🚀 Generate BOQ": show_intelligent_boq_page,
        "📦 Product Database": show_database_page,
        "📈 Analytics": show_analytics_page,
    }
    
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app uses AI to create optimized Bills of Quantities for AV systems. "
        "Navigate using the options above."
    )
    st.sidebar.markdown("**Version:** 2.0.2")

    # Run the selected page function
    pages[selected_page]()

if __name__ == "__main__":
    main()
