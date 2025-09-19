import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="BOQ Generator",
    page_icon="📊",
    layout="wide"
)

class SimpleBOQGenerator:
    def __init__(self, products_data):
        self.products_data = products_data
        
    def generate_boq(self, requirements):
        """Generate BOQ based on project requirements"""
        selected_products = []
        
        # Map requirements to categories
        category_mapping = {
            'Display Systems': ['display'],
            'Audio Systems': ['audio'], 
            'Control Systems': ['control'],
            'Video Conferencing': ['video', 'audio'],
            'Digital Signage': ['display', 'control'],
            'Lighting Control': ['control'],
            'Streaming Solutions': ['video']
        }
        
        required_categories = set()
        for req in requirements['requirements']:
            if req in category_mapping:
                required_categories.update(category_mapping[req])
        
        # Always add cables for any AV system
        required_categories.add('cable')
        
        # Select products for each category
        for category in required_categories:
            category_products = [p for p in self.products_data if p.get('category') == category]
            
            if category_products:
                # Simple selection: pick products based on price range
                budget_max = self._get_budget_max(requirements['budget_range'])
                suitable_products = [p for p in category_products if p.get('price', 0) <= budget_max * 0.3]
                
                if suitable_products:
                    # Select top 2 products for the category
                    selected = suitable_products[:2]
                    
                    for product in selected:
                        quantity = self._calculate_quantity(category, requirements['audience_size'])
                        unit_cost = product.get('price', 0)
                        
                        boq_item = {
                            'category': category.title(),
                            'description': product.get('description', 'N/A'),
                            'model_no': product.get('model_no', 'N/A'),
                            'company': product.get('company', 'N/A'),
                            'quantity': quantity,
                            'unit': 'Each' if category != 'cable' else 'Meter',
                            'unit_cost': unit_cost,
                            'total_cost': quantity * unit_cost
                        }
                        selected_products.append(boq_item)
        
        total_cost = sum(item['total_cost'] for item in selected_products)
        
        return {
            'project_name': requirements['project_name'],
            'client_name': requirements['client_name'],
            'project_type': requirements['project_type'],
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'items': selected_products,
            'total_cost': total_cost,
            'item_count': len(selected_products)
        }
    
    def _get_budget_max(self, budget_range):
        budget_map = {
            "< $10,000": 10000,
            "$10,000 - $50,000": 50000,
            "$50,000 - $100,000": 100000,
            "> $100,000": 200000
        }
        return budget_map.get(budget_range, 50000)
    
    def _calculate_quantity(self, category, audience_size):
        quantity_map = {
            'display': max(1, audience_size // 50),
            'audio': max(2, audience_size // 20),
            'control': 1,
            'video': max(1, audience_size // 30),
            'cable': max(5, audience_size // 10),
            'mounting': max(1, audience_size // 50)
        }
        return quantity_map.get(category, 1)

def load_product_data():
    """Load product data from JSON file"""
    try:
        if os.path.exists('data/processed_products.json'):
            with open('data/processed_products.json', 'r') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        st.error(f"Error loading product data: {str(e)}")
        return []

def save_product_data(data):
    """Save product data to JSON file"""
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/processed_products.json', 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving product data: {str(e)}")
        return False

def create_sample_data():
    """Create sample product data for testing"""
    sample_products = [
        {
            "company": "DisplayTech Pro",
            "model_no": "DTP-75-4K",
            "description": "75\" 4K Interactive Display",
            "category": "display",
            "price": 4500.00,
            "specifications": {"resolution": "4K", "size": "75\""},
            "compatibility": ["HDMI", "USB"]
        },
        {
            "company": "AudioPro Systems", 
            "model_no": "APS-SPKR-100",
            "description": "Ceiling Speaker 100W",
            "category": "audio",
            "price": 450.00,
            "specifications": {"power": "100W", "frequency": "50Hz-20kHz"},
            "compatibility": ["Amplifier"]
        },
        {
            "company": "ControlTech",
            "model_no": "CT-TOUCH-10",
            "description": "10\" Touch Control Panel",
            "category": "control", 
            "price": 1200.00,
            "specifications": {"size": "10\"", "connectivity": "Ethernet"},
            "compatibility": ["Ethernet", "IR"]
        },
        {
            "company": "CableTech",
            "model_no": "CBL-HDMI-15",
            "description": "HDMI Cable 15m",
            "category": "cable",
            "price": 25.00,
            "specifications": {"length": "15m", "type": "HDMI 2.1"},
            "compatibility": ["HDMI"]
        }
    ]
    return sample_products

def process_csv_data(df):
    """Process CSV data to JSON format"""
    
    def map_category(category_name):
        """Map CSV categories to BOQ categories"""
        category_mapping = {
            'Displays & Projectors': 'display',
            'Mounts & Racks': 'mounting',
            'Audio': 'audio',
            'Control Systems': 'control',
            'Video': 'video',
            'Cables': 'cable'
        }
        
        if category_name in category_mapping:
            return category_mapping[category_name]
        
        # Fallback keyword matching
        category_lower = str(category_name).lower()
        if 'display' in category_lower or 'projector' in category_lower:
            return 'display'
        elif 'audio' in category_lower or 'speaker' in category_lower:
            return 'audio'
        elif 'control' in category_lower:
            return 'control'
        elif 'video' in category_lower or 'camera' in category_lower:
            return 'video'
        elif 'cable' in category_lower:
            return 'cable'
        elif 'mount' in category_lower or 'rack' in category_lower:
            return 'mounting'
        else:
            return 'other'
    
    def parse_compatibility(compatibility_tags):
        """Parse compatibility tags"""
        if pd.isna(compatibility_tags):
            return []
        
        tags = str(compatibility_tags).replace(' ', '').split(',')
        compatibility_map = {
            'samsung_magicinfo': 'Samsung MagicInfo',
            'vesa': 'VESA',
            'hdmi': 'HDMI',
            'usb': 'USB',
            'ethernet': 'Ethernet'
        }
        
        mapped_tags = []
        for tag in tags:
            tag = tag.lower().strip()
            if tag in compatibility_map:
                mapped_tags.append(compatibility_map[tag])
            elif tag:
                mapped_tags.append(tag.title())
        
        return mapped_tags
    
    processed_products = []
    
    for index, row in df.iterrows():
        if pd.isna(row['name']) or pd.isna(row['brand']):
            continue
        
        product = {
            'company': str(row['brand']).strip(),
            'model_no': f"{row['brand']}-{index:04d}",
            'description': str(row['name']).strip(),
            'category': map_category(str(row['category'])),
            'price': float(row['price']) if not pd.isna(row['price']) else 0.0,
            'specifications': {
                'features': str(row['features']) if not pd.isna(row['features']) else '',
                'tier': str(row['tier']) if not pd.isna(row['tier']) else '',
                'use_cases': str(row['use_case_tags']).split(',') if not pd.isna(row['use_case_tags']) else []
            },
            'compatibility': parse_compatibility(row.get('compatibility_tags', ''))
        }
        
        processed_products.append(product)
    
    return processed_products

def main():
    st.title("🚀 Simple BOQ Generator")
    st.markdown("Generate Bills of Quantities from your product database")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Choose a page", [
            "BOQ Generator",
            "Product Database", 
            "Analytics"
        ])
    
    if page == "Product Database":
        show_product_database_page()
    elif page == "Analytics":
        show_analytics_page()
    else:
        show_boq_generator_page()

def show_product_database_page():
    st.header("📁 Product Database Management")
    
    # Load existing data
    products_data = load_product_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Product Data")
        
        # Option 1: Upload JSON
        uploaded_json = st.file_uploader("Upload processed_products.json", type=['json'])
        
        if uploaded_json:
            try:
                new_products = json.load(uploaded_json)
                st.success(f"Loaded {len(new_products)} products from JSON!")
                
                if st.button("Save JSON to Database"):
                    if save_product_data(new_products):
                        st.success("Data saved successfully!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading JSON file: {str(e)}")
        
        st.markdown("---")
        
        # Option 2: Upload CSV (your format)
        uploaded_csv = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_csv:
            try:
                # Process CSV
                df = pd.read_csv(uploaded_csv)
                st.success(f"Loaded CSV with {len(df)} rows!")
                
                # Show preview
                st.write("**CSV Preview:**")
                st.dataframe(df.head())
                
                if st.button("Process CSV to Database"):
                    processed_products = process_csv_data(df)
                    if save_product_data(processed_products):
                        st.success(f"Processed and saved {len(processed_products)} products!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
        
        if st.button("Create Sample Data"):
            sample_data = create_sample_data()
            if save_product_data(sample_data):
                st.success("Sample data created!")
                st.rerun()
    
    with col2:
        st.subheader("Database Status")
        if products_data:
            st.success(f"✅ {len(products_data)} products in database")
            
            # Category breakdown
            if products_data:
                categories = {}
                for product in products_data:
                    cat = product.get('category', 'unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                
                st.write("**Categories:**")
                for cat, count in categories.items():
                    st.write(f"- {cat.title()}: {count} products")
        else:
            st.warning("No products in database")
    
    # Show product data
    if products_data:
        st.subheader("Product Data Preview")
        df = pd.DataFrame(products_data)
        st.dataframe(df, use_container_width=True)

def show_boq_generator_page():
    st.header("📊 BOQ Generation")
    
    # Load product data
    products_data = load_product_data()
    
    if not products_data:
        st.warning("No product data available. Please add products in the Product Database page first.")
        if st.button("Create Sample Data Now"):
            sample_data = create_sample_data()
            if save_product_data(sample_data):
                st.success("Sample data created!")
                st.rerun()
        return
    
    st.success(f"✅ {len(products_data)} products available")
    
    # BOQ Generation Form
    with st.form("boq_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Project Details")
            project_name = st.text_input("Project Name*")
            client_name = st.text_input("Client Name*")
            project_type = st.selectbox("Project Type", [
                "Conference Room", "Auditorium", "Classroom", 
                "Control Room", "Retail Space", "Corporate Office"
            ])
            audience_size = st.number_input("Expected Audience Size", min_value=1, value=20)
        
        with col2:
            st.subheader("Requirements")
            budget_range = st.selectbox("Budget Range", [
                "< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"
            ])
            
            requirements = st.multiselect("Select Requirements", [
                "Display Systems", "Audio Systems", "Control Systems",
                "Video Conferencing", "Digital Signage", "Lighting Control",
                "Streaming Solutions"
            ])
            
            special_requirements = st.text_area("Special Requirements")
        
        submitted = st.form_submit_button("Generate BOQ", type="primary")
        
        if submitted:
            if project_name and client_name and requirements:
                # Generate BOQ
                generator = SimpleBOQGenerator(products_data)
                
                boq_requirements = {
                    'project_name': project_name,
                    'client_name': client_name,
                    'project_type': project_type,
                    'requirements': requirements,
                    'audience_size': audience_size,
                    'budget_range': budget_range,
                    'special_requirements': special_requirements
                }
                
                with st.spinner("Generating BOQ..."):
                    boq_data = generator.generate_boq(boq_requirements)
                    
                    # Store in session state
                    st.session_state['current_boq'] = boq_data
                    
                    st.success("BOQ Generated Successfully!")
            else:
                st.error("Please fill in all required fields (marked with *)")
    
    # Display generated BOQ
    if 'current_boq' in st.session_state:
        boq_data = st.session_state['current_boq']
        
        st.subheader("Generated BOQ")
        
        # Project info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", boq_data['item_count'])
        with col2:
            st.metric("Total Cost", f"${boq_data['total_cost']:,.2f}")
        with col3:
            st.metric("Generated", boq_data['generated_date'])
        
        # BOQ Items table
        if boq_data['items']:
            df_boq = pd.DataFrame(boq_data['items'])
            st.dataframe(df_boq, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv_data = df_boq.to_csv(index=False)
                st.download_button(
                    "Download as CSV",
                    csv_data,
                    f"BOQ_{boq_data['project_name']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with col2:
                json_data = json.dumps(boq_data, indent=2)
                st.download_button(
                    "Download as JSON",
                    json_data,
                    f"BOQ_{boq_data['project_name']}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json"
                )
        else:
            st.warning("No suitable products found for your requirements. Try adjusting your budget range or requirements.")

def show_analytics_page():
    st.header("📈 Analytics")
    
    products_data = load_product_data()
    
    if not products_data:
        st.warning("No product data available for analytics.")
        return
    
    df = pd.DataFrame(products_data)
    
    # Category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values, 
                names=category_counts.index, 
                title="Product Categories Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'price' in df.columns:
            fig_hist = px.histogram(
                df, 
                x='price', 
                title="Price Distribution",
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Company distribution
    if 'company' in df.columns:
        company_counts = df['company'].value_counts().head(10)
        fig_bar = px.bar(
            x=company_counts.values,
            y=company_counts.index,
            orientation='h',
            title="Top 10 Companies by Product Count"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Data summary
    st.subheader("Database Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        if 'price' in df.columns:
            avg_price = df['price'].mean()
            st.metric("Average Price", f"${avg_price:,.2f}")
    with col3:
        if 'company' in df.columns:
            unique_companies = df['company'].nunique()
            st.metric("Companies", unique_companies)
    with col4:
        if 'category' in df.columns:
            unique_categories = df['category'].nunique()
            st.metric("Categories", unique_categories)

if __name__ == "__main__":
    main()
