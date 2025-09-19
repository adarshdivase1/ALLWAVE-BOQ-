import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="BOQ Generator",
    page_icon="📊",
    layout="wide"
)

# --- Core BOQ Generation Logic ---
class SimpleBOQGenerator:
    def __init__(self, products_data):
        self.products_data = products_data

    def generate_boq(self, requirements):
        """Generate BOQ based on project requirements."""
        selected_products = []

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

        # Always add cables and mounting for any AV system
        required_categories.add('cable')
        required_categories.add('mounting')

        # Select products for each category
        for category in required_categories:
            category_products = [p for p in self.products_data if p.get('category') == category]

            if category_products:
                budget_max = self._get_budget_max(requirements['budget_range'])
                # Simple logic: find suitable products within a fraction of the budget
                suitable_products = [p for p in category_products if p.get('price', 0) <= budget_max * 0.3]

                if suitable_products:
                    # Select top 2 products for variety, sorted by price
                    selected = sorted(suitable_products, key=lambda p: p['price'], reverse=True)[:2]

                    for product in selected:
                        quantity = self._calculate_quantity(category, requirements['audience_size'])
                        unit_cost = product.get('price', 0)

                        boq_item = {
                            'Category': category.title(),
                            'Description': product.get('description', 'N/A'),
                            'Model No': product.get('model_no', 'N/A'),
                            'Company': product.get('company', 'N/A'),
                            'Quantity': quantity,
                            'Unit': 'Each' if category != 'cable' else 'Meter',
                            'Unit Cost': unit_cost,
                            'Total Cost': quantity * unit_cost
                        }
                        selected_products.append(boq_item)

        total_cost = sum(item['Total Cost'] for item in selected_products)

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
            'cable': max(50, audience_size * 2), # Assuming 2m cable per person
            'mounting': max(1, audience_size // 50) # Match display count
        }
        return quantity_map.get(category, 1)

# --- Data Handling Functions ---
def load_product_data():
    """Load product data from JSON file."""
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
    """Save product data to JSON file."""
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/processed_products.json', 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving product data: {str(e)}")
        return False

def create_sample_data():
    """Create sample product data for testing."""
    return [
        {"company": "DisplayTech Pro", "model_no": "DTP-75-4K", "description": "75\" 4K Interactive Display", "category": "display", "price": 4500.00, "specifications": {"resolution": "4K", "size": "75\""}, "compatibility": ["HDMI", "USB"]},
        {"company": "AudioPro Systems", "model_no": "APS-SPKR-100", "description": "Ceiling Speaker 100W", "category": "audio", "price": 450.00, "specifications": {"power": "100W", "frequency": "50Hz-20kHz"}, "compatibility": ["Amplifier"]},
        {"company": "ControlTech", "model_no": "CT-TOUCH-10", "description": "10\" Touch Control Panel", "category": "control", "price": 1200.00, "specifications": {"size": "10\"", "connectivity": "Ethernet"}, "compatibility": ["Ethernet", "IR"]},
        {"company": "VideoLink", "model_no": "VL-CAM-PTZ12", "description": "12x PTZ Conference Camera", "category": "video", "price": 1800.00, "specifications": {"zoom": "12x Optical", "resolution": "1080p"}, "compatibility": ["USB", "HDMI"]},
        {"company": "Mounts Inc.", "model_no": "MI-WM-80", "description": "Universal Wall Mount for 55-80\" Displays", "category": "mounting", "price": 150.00, "specifications": {"max_weight": "150lbs"}, "compatibility": ["VESA"]},
        {"company": "CableTech", "model_no": "CBL-HDMI-15", "description": "HDMI Cable 15m", "category": "cable", "price": 25.00, "specifications": {"length": "15m", "type": "HDMI 2.1"}, "compatibility": ["HDMI"]}
    ]

def process_csv_data(df):
    """Process CSV data from the user's file into the app's JSON format."""
    def map_category(category_name):
        category_lower = str(category_name).lower()
        if 'display' in category_lower or 'projector' in category_lower: return 'display'
        if 'audio' in category_lower or 'speaker' in category_lower or 'mic' in category_lower: return 'audio'
        if 'control' in category_lower or 'process' in category_lower: return 'control'
        if 'video' in category_lower or 'camera' in category_lower: return 'video'
        if 'cable' in category_lower or 'connect' in category_lower: return 'cable'
        if 'mount' in category_lower or 'rack' in category_lower: return 'mounting'
        return 'other'

    processed_products = []
    for index, row in df.iterrows():
        if pd.isna(row.get('name')) or pd.isna(row.get('brand')):
            continue
        
        try:
            price = float(row.get('price', 0.0))
        except (ValueError, TypeError):
            price = 0.0

        product = {
            'company': str(row.get('brand', '')).strip(),
            'model_no': f"{row.get('brand', 'GEN')}-{index:04d}",
            'description': str(row.get('name', '')).strip(),
            'category': map_category(row.get('category', '')),
            'price': price,
            'specifications': {
                'features': str(row.get('features', '')) if pd.notna(row.get('features')) else '',
                'tier': str(row.get('tier', '')) if pd.notna(row.get('tier')) else '',
                'use_cases': str(row.get('use_case_tags', '')).split(',') if pd.notna(row.get('use_case_tags')) else []
            },
            'compatibility': str(row.get('compatibility_tags', '')).split(',') if pd.notna(row.get('compatibility_tags')) else []
        }
        processed_products.append(product)
    return processed_products

# --- Streamlit Page Functions ---
def show_boq_generator_page():
    st.header("📊 BOQ Generation")
    products_data = load_product_data()

    if not products_data:
        st.warning("No product data available. Please add products on the 'Product Database' page first.")
        if st.button("Create Sample Data Now"):
            if save_product_data(create_sample_data()):
                st.success("Sample data created!")
                st.rerun()
        return

    st.success(f"✅ Using a database of {len(products_data)} products.")

    with st.form("boq_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Project Details")
            project_name = st.text_input("Project Name*")
            client_name = st.text_input("Client Name*")
            project_type = st.selectbox("Project Type", ["Conference Room", "Auditorium", "Classroom", "Control Room", "Retail Space", "Corporate Office"])
            audience_size = st.number_input("Expected Audience Size", min_value=1, value=20)
        with col2:
            st.subheader("Requirements")
            budget_range = st.selectbox("Budget Range", ["< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"])
            requirements = st.multiselect("Select Requirements", ["Display Systems", "Audio Systems", "Control Systems", "Video Conferencing", "Digital Signage", "Lighting Control", "Streaming Solutions"])
            special_requirements = st.text_area("Special Requirements (optional)")

        submitted = st.form_submit_button("Generate BOQ", type="primary")

        if submitted:
            if not project_name or not client_name or not requirements:
                st.error("Please fill in all required fields (marked with *) and select at least one requirement.")
            else:
                generator = SimpleBOQGenerator(products_data)
                boq_requirements = {
                    'project_name': project_name, 'client_name': client_name,
                    'project_type': project_type, 'requirements': requirements,
                    'audience_size': audience_size, 'budget_range': budget_range,
                    'special_requirements': special_requirements
                }
                with st.spinner("Generating BOQ..."):
                    boq_data = generator.generate_boq(boq_requirements)
                    st.session_state['current_boq'] = boq_data
                    st.success("BOQ Generated Successfully!")

    if 'current_boq' in st.session_state:
        boq_data = st.session_state['current_boq']
        st.subheader(f"Generated BOQ for: {boq_data['project_name']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Items", boq_data['item_count'])
        col2.metric("Total Cost", f"${boq_data['total_cost']:,.2f}")
        col3.metric("Generated On", boq_data['generated_date'].split(' ')[0])

        if boq_data['items']:
            df_boq = pd.DataFrame(boq_data['items'])
            st.dataframe(df_boq, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            csv_data = df_boq.to_csv(index=False).encode('utf-8')
            col1.download_button("Download as CSV", csv_data, f"BOQ_{boq_data['project_name']}.csv", "text/csv")

            json_data = json.dumps(boq_data, indent=2)
            col2.download_button("Download as JSON", json_data, f"BOQ_{boq_data['project_name']}.json", "application/json")
        else:
            st.warning("No suitable products found. Try adjusting the budget or requirements.")

def show_product_database_page():
    st.header("📁 Product Database Management")
    products_data = load_product_data()

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("Upload & Process Data")
        uploaded_csv = st.file_uploader("Upload product catalog (CSV)", type=['csv'])
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                st.success(f"Loaded CSV with {len(df)} rows!")
                st.dataframe(df.head())

                if st.button("Process CSV and Add to Database"):
                    processed_products = process_csv_data(df)
                    if processed_products:
                        if save_product_data(processed_products):
                            st.success(f"Successfully processed and saved {len(processed_products)} products!")
                            st.rerun()
                    else:
                        st.error("No valid products were processed. Check your CSV format.")
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")

        st.markdown("---")
        if st.button("Load Sample Data (will overwrite existing data)"):
            if save_product_data(create_sample_data()):
                st.success("Sample data loaded!")
                st.rerun()

    with col2:
        st.subheader("Database Status")
        if products_data:
            st.success(f"✅ {len(products_data)} products in database.")
            categories = pd.Series([p.get('category', 'unknown') for p in products_data]).value_counts()
            st.dataframe(categories.rename("Product Count"))
        else:
            st.warning("No products in database.")

    if products_data:
        st.subheader("Product Data Preview")
        st.dataframe(pd.DataFrame(products_data), use_container_width=True, height=400)

def show_analytics_page():
    st.header("📈 Analytics Dashboard")
    products_data = load_product_data()

    if not products_data:
        st.warning("No product data available for analytics. Please upload data on the 'Product Database' page.")
        return

    df = pd.DataFrame(products_data).dropna(subset=['price', 'category', 'company'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", len(df))
    col2.metric("Unique Companies", df['company'].nunique())
    col3.metric("Average Price", f"${df['price'].mean():,.2f}")

    col1, col2 = st.columns(2)
    with col1:
        category_counts = df['category'].value_counts()
        fig_pie = px.pie(values=category_counts.values, names=category_counts.index, title="Product Distribution by Category")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        company_counts = df['company'].value_counts().head(10)
        fig_bar = px.bar(x=company_counts.index, y=company_counts.values, title="Top 10 Companies by Product Count", labels={'x': 'Company', 'y': 'Count'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader("Price Analysis by Category")
    fig_box = px.box(df, x='category', y='price', title="Price Range by Category")
    st.plotly_chart(fig_box, use_container_width=True)

# --- Main App Router ---
def main():
    st.title("🚀 Simple BOQ Generator")
    st.markdown("An intelligent tool to generate Bills of Quantities from your product database.")

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Choose a page", ["BOQ Generator", "Product Database", "Analytics"])

    if page == "Product Database":
        show_product_database_page()
    elif page == "Analytics":
        show_analytics_page()
    else:
        show_boq_generator_page()

if __name__ == "__main__":
    main()
