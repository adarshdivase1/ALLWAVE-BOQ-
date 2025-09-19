import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="Professional BOQ Generator",
    page_icon="📊",
    layout="wide"
)

# --- Enhanced BOQ Generation Logic ---
class ProfessionalBOQGenerator:
    def __init__(self, products_data):
        self.products_data = products_data
        
        # Define logical product relationships
        self.product_relationships = {
            'display': {'requires': ['mounting', 'cable'], 'optional': ['control']},
            'audio': {'requires': ['cable'], 'optional': ['control', 'mounting']},
            'video': {'requires': ['cable', 'mounting'], 'optional': ['control']},
            'control': {'requires': ['cable'], 'optional': []},
            'mounting': {'requires': [], 'optional': []},
            'cable': {'requires': [], 'optional': []}
        }
        
        # Define quantity logic based on room type and audience
        self.quantity_rules = {
            'Conference Room': {
                'display': lambda size: 1 if size <= 20 else 2,
                'audio': lambda size: max(2, size // 10),
                'video': lambda size: 1 if size <= 30 else 2,
                'control': lambda size: 1,
                'mounting': lambda size: 1 if size <= 20 else 2,
                'cable': lambda size: max(20, size * 1.5)
            },
            'Auditorium': {
                'display': lambda size: max(1, size // 100),
                'audio': lambda size: max(4, size // 25),
                'video': lambda size: max(2, size // 50),
                'control': lambda size: max(1, size // 100),
                'mounting': lambda size: max(1, size // 100),
                'cable': lambda size: max(50, size * 3)
            },
            'Classroom': {
                'display': lambda size: 1,
                'audio': lambda size: max(2, size // 15),
                'video': lambda size: 1,
                'control': lambda size: 1,
                'mounting': lambda size: 1,
                'cable': lambda size: max(15, size * 1.2)
            }
        }

    def generate_boq(self, requirements):
        """Generate a logical and production-ready BOQ."""
        selected_products = []
        
        # Map requirements to product categories
        category_mapping = {
            'Display Systems': ['display'],
            'Audio Systems': ['audio'],
            'Control Systems': ['control'],
            'Video Conferencing': ['video', 'audio'],
            'Digital Signage': ['display', 'control'],
            'Lighting Control': ['control'],
            'Streaming Solutions': ['video', 'audio']
        }
        
        required_categories = set()
        for req in requirements['requirements']:
            if req in category_mapping:
                required_categories.update(category_mapping[req])
        
        # Add dependent categories based on relationships
        additional_categories = set()
        for category in required_categories:
            if category in self.product_relationships:
                additional_categories.update(self.product_relationships[category]['requires'])
        
        required_categories.update(additional_categories)
        
        # Get budget constraints
        budget_max = self._get_budget_max(requirements['budget_range'])
        allocated_budget = self._allocate_budget_by_category(budget_max, required_categories)
        
        # Generate BOQ items for each category
        for category in required_categories:
            category_products = self._get_products_by_category(category)
            
            if not category_products:
                # Add a placeholder item if no products available
                selected_products.append(self._create_placeholder_item(category, requirements))
                continue
            
            # Filter products within budget allocation for this category
            category_budget = allocated_budget.get(category, budget_max * 0.1)
            suitable_products = self._filter_products_by_budget(
                category_products, category_budget, requirements['audience_size']
            )
            
            if suitable_products:
                selected_product = self._select_best_product(suitable_products, requirements)
                quantity = self._calculate_quantity(category, requirements)
                
                boq_item = self._create_boq_item(selected_product, category, quantity)
                selected_products.append(boq_item)
        
        # Add installation and miscellaneous items
        selected_products.extend(self._add_installation_items(requirements, selected_products))
        
        # Calculate totals and add contingency
        total_cost = sum(item['Total Cost'] for item in selected_products)
        contingency = total_cost * 0.1  # 10% contingency
        
        # Add contingency as a line item
        selected_products.append({
            'Category': 'Miscellaneous',
            'Description': 'Project Contingency (10%)',
            'Model No': 'CONTINGENCY',
            'Company': 'Project Management',
            'Quantity': 1,
            'Unit': 'Lump Sum',
            'Unit Cost': contingency,
            'Total Cost': contingency
        })
        
        total_cost_with_contingency = total_cost + contingency
        
        return {
            'project_name': requirements['project_name'],
            'client_name': requirements['client_name'],
            'project_type': requirements['project_type'],
            'audience_size': requirements['audience_size'],
            'budget_range': requirements['budget_range'],
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'items': selected_products,
            'subtotal': total_cost,
            'contingency': contingency,
            'total_cost': total_cost_with_contingency,
            'item_count': len(selected_products),
            'budget_utilization': (total_cost_with_contingency / budget_max) * 100 if budget_max > 0 else 0
        }
    
    def _get_products_by_category(self, category):
        """Get all products for a specific category."""
        return [p for p in self.products_data if p.get('category', '').lower() == category.lower()]
    
    def _filter_products_by_budget(self, products, category_budget, audience_size):
        """Filter products that fit within the category budget."""
        suitable_products = []
        
        for product in products:
            price = product.get('price', 0)
            if price == 0:
                continue
                
            # Estimate total cost for this product including quantity
            estimated_quantity = self._estimate_quantity_for_budget(product.get('category'), audience_size)
            estimated_total = price * estimated_quantity
            
            if estimated_total <= category_budget:
                suitable_products.append(product)
        
        return suitable_products
    
    def _estimate_quantity_for_budget(self, category, audience_size):
        """Estimate quantity needed for budget calculation."""
        base_quantities = {
            'display': max(1, audience_size // 50),
            'audio': max(2, audience_size // 20),
            'video': max(1, audience_size // 30),
            'control': 1,
            'mounting': max(1, audience_size // 50),
            'cable': max(20, audience_size * 2)
        }
        return base_quantities.get(category, 1)
    
    def _select_best_product(self, products, requirements):
        """Select the best product based on requirements and value."""
        if not products:
            return None
        
        # Sort by value (features vs price) - for now, use price as primary factor
        # In production, you'd want more sophisticated scoring
        budget_max = self._get_budget_max(requirements['budget_range'])
        
        # Prefer products in the middle price range (not cheapest, not most expensive)
        scored_products = []
        for product in products:
            price = product.get('price', 0)
            # Score based on price positioning (sweet spot around 30-60% of budget allocation)
            price_score = self._calculate_price_score(price, budget_max)
            scored_products.append((product, price_score))
        
        # Sort by score and return best product
        scored_products.sort(key=lambda x: x[1], reverse=True)
        return scored_products[0][0]
    
    def _calculate_price_score(self, price, budget_max):
        """Calculate a score based on price positioning."""
        if budget_max == 0:
            return 0
        
        price_ratio = price / (budget_max * 0.1)  # Assuming 10% of budget for this category
        
        # Optimal price range is 0.3 to 0.7 of category budget
        if 0.3 <= price_ratio <= 0.7:
            return 100
        elif price_ratio < 0.3:
            return 70  # Too cheap might mean lower quality
        else:
            return max(0, 100 - (price_ratio - 0.7) * 100)  # Too expensive
    
    def _create_boq_item(self, product, category, quantity):
        """Create a properly formatted BOQ item."""
        unit_cost = product.get('price', 0)
        
        return {
            'Category': category.title(),
            'Description': product.get('description', 'Product Description Not Available'),
            'Model No': product.get('model_no', f"MODEL-{random.randint(1000, 9999)}"),
            'Company': product.get('company', 'Manufacturer TBD'),
            'Quantity': quantity,
            'Unit': self._get_unit_for_category(category),
            'Unit Cost': round(unit_cost, 2),
            'Total Cost': round(quantity * unit_cost, 2)
        }
    
    def _create_placeholder_item(self, category, requirements):
        """Create a placeholder item when no products are available in the database."""
        quantity = self._calculate_quantity(category, requirements)
        estimated_cost = self._get_estimated_cost_for_category(category)
        
        return {
            'Category': category.title(),
            'Description': f'{category.title()} System - To Be Specified',
            'Model No': 'TBD',
            'Company': 'To Be Determined',
            'Quantity': quantity,
            'Unit': self._get_unit_for_category(category),
            'Unit Cost': estimated_cost,
            'Total Cost': round(quantity * estimated_cost, 2)
        }
    
    def _get_unit_for_category(self, category):
        """Get appropriate unit for each category."""
        unit_map = {
            'cable': 'Meter',
            'display': 'Each',
            'audio': 'Each',
            'video': 'Each',
            'control': 'Each',
            'mounting': 'Each'
        }
        return unit_map.get(category.lower(), 'Each')
    
    def _get_estimated_cost_for_category(self, category):
        """Get estimated costs for categories when no products are available."""
        cost_estimates = {
            'display': 3000,
            'audio': 500,
            'video': 1500,
            'control': 1000,
            'mounting': 200,
            'cable': 15
        }
        return cost_estimates.get(category.lower(), 1000)
    
    def _calculate_quantity(self, category, requirements):
        """Calculate logical quantity based on project type and audience size."""
        project_type = requirements.get('project_type', 'Conference Room')
        audience_size = requirements.get('audience_size', 20)
        
        if project_type in self.quantity_rules:
            rule = self.quantity_rules[project_type].get(category.lower())
            if rule:
                return rule(audience_size)
        
        # Fallback to basic rules
        basic_rules = {
            'display': max(1, audience_size // 50),
            'audio': max(2, audience_size // 20),
            'video': max(1, audience_size // 30),
            'control': 1,
            'mounting': max(1, audience_size // 50),
            'cable': max(20, audience_size * 2)
        }
        
        return basic_rules.get(category.lower(), 1)
    
    def _allocate_budget_by_category(self, total_budget, categories):
        """Allocate budget across different categories based on typical AV project distributions."""
        budget_allocation = {
            'display': 0.40,  # 40% for displays (usually the largest expense)
            'audio': 0.25,    # 25% for audio systems
            'video': 0.15,    # 15% for video equipment
            'control': 0.10,  # 10% for control systems
            'mounting': 0.05, # 5% for mounting hardware
            'cable': 0.05     # 5% for cables and connectivity
        }
        
        allocated_budget = {}
        for category in categories:
            allocation_percentage = budget_allocation.get(category.lower(), 0.1)
            allocated_budget[category] = total_budget * allocation_percentage
        
        return allocated_budget
    
    def _add_installation_items(self, requirements, existing_items):
        """Add installation and project management items."""
        installation_items = []
        
        # Calculate installation cost as percentage of equipment cost
        equipment_cost = sum(item['Total Cost'] for item in existing_items)
        installation_percentage = 0.15  # 15% of equipment cost
        
        installation_items.append({
            'Category': 'Installation',
            'Description': 'Professional Installation and Configuration',
            'Model No': 'INSTALL-001',
            'Company': 'Installation Team',
            'Quantity': 1,
            'Unit': 'Lump Sum',
            'Unit Cost': round(equipment_cost * installation_percentage, 2),
            'Total Cost': round(equipment_cost * installation_percentage, 2)
        })
        
        # Add project management
        pm_cost = equipment_cost * 0.05  # 5% for project management
        installation_items.append({
            'Category': 'Project Management',
            'Description': 'Project Management and Coordination',
            'Model No': 'PM-001',
            'Company': 'Project Management Office',
            'Quantity': 1,
            'Unit': 'Lump Sum',
            'Unit Cost': round(pm_cost, 2),
            'Total Cost': round(pm_cost, 2)
        })
        
        return installation_items
    
    def _get_budget_max(self, budget_range):
        """Get maximum budget value from range."""
        budget_map = {
            "< $10,000": 10000,
            "$10,000 - $50,000": 50000,
            "$50,000 - $100,000": 100000,
            "> $100,000": 250000
        }
        return budget_map.get(budget_range, 50000)

# --- Data Handling Functions (keeping existing ones but improved) ---
def load_product_data():
    """Load product data from JSON file."""
    try:
        if os.path.exists('data/processed_products.json'):
            with open('data/processed_products.json', 'r') as f:
                data = json.load(f)
                # Validate data structure
                valid_data = []
                for item in data:
                    if isinstance(item, dict) and 'description' in item:
                        # Ensure price is numeric
                        try:
                            item['price'] = float(item.get('price', 0))
                        except (ValueError, TypeError):
                            item['price'] = 0
                        valid_data.append(item)
                return valid_data
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
    """Create realistic sample product data."""
    return [
        {"company": "Samsung", "model_no": "QM75R", "description": "75\" 4K UHD Commercial Display", "category": "display", "price": 2800.00},
        {"company": "LG", "model_no": "86UN7300", "description": "86\" 4K Smart Commercial Display", "category": "display", "price": 3500.00},
        {"company": "Sony", "model_no": "VPL-FHZ75", "description": "Laser Projector 7000 Lumens", "category": "display", "price": 8500.00},
        
        {"company": "Bose", "model_no": "EdgeMax EM90", "description": "Premium Ceiling Speaker", "category": "audio", "price": 400.00},
        {"company": "JBL", "model_no": "Control 26CT", "description": "Ceiling Speaker with Transformer", "category": "audio", "price": 180.00},
        {"company": "Shure", "model_no": "MXA910", "description": "Ceiling Array Microphone", "category": "audio", "price": 1200.00},
        
        {"company": "Crestron", "model_no": "CP4N", "description": "4-Series Control System", "category": "control", "price": 2500.00},
        {"company": "AMX", "model_no": "NX-2200", "description": "NetLinx Control Processor", "category": "control", "price": 1800.00},
        {"company": "Extron", "model_no": "TLP 720M", "description": "7\" TouchLink Pro Touchpanel", "category": "control", "price": 1100.00},
        
        {"company": "Logitech", "model_no": "Rally", "description": "Premium ConferenceCam", "category": "video", "price": 1200.00},
        {"company": "Poly", "model_no": "Studio X50", "description": "All-in-One Video Bar", "category": "video", "price": 2200.00},
        {"company": "Cisco", "model_no": "Webex Room Kit", "description": "Smart Video Collaboration", "category": "video", "price": 3500.00},
        
        {"company": "Chief", "model_no": "LTM1U", "description": "Large Tilt Wall Mount", "category": "mounting", "price": 180.00},
        {"company": "Peerless", "model_no": "ST680", "description": "Tilting Wall Mount", "category": "mounting", "price": 150.00},
        {"company": "Middle Atlantic", "model_no": "ERK-2025", "description": "20RU Equipment Rack", "category": "mounting", "price": 850.00},
        
        {"company": "Kramer", "model_no": "C-HM/HM-15", "description": "HDMI Cable 15ft", "category": "cable", "price": 45.00},
        {"company": "Belden", "model_no": "1694A", "description": "HD-SDI Video Cable (per ft)", "category": "cable", "price": 3.50},
        {"company": "Mogami", "model_no": "2534", "description": "Balanced Audio Cable (per ft)", "category": "cable", "price": 2.80}
    ]

# --- Enhanced BOQ Generator Page ---
def show_boq_generator_page():
    st.header("📊 Professional BOQ Generation")
    products_data = load_product_data()

    if not products_data:
        st.warning("⚠️ No product data available. Please add products on the 'Product Database' page first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Sample Data", type="primary"):
                if save_product_data(create_sample_data()):
                    st.success("Sample data created!")
                    st.rerun()
        with col2:
            st.info("💡 Upload your product catalog in the 'Product Database' section for accurate BOQ generation.")
        return

    st.success(f"✅ Using database with {len(products_data)} products across {len(set(p.get('category', 'unknown') for p in products_data))} categories")

    # Enhanced form with better validation and help text
    with st.form("professional_boq_form"):
        st.subheader("📋 Project Information")
        
        col1, col2 = st.columns(2)
        with col1:
            project_name = st.text_input("Project Name*", placeholder="e.g., Main Conference Room AV Upgrade")
            client_name = st.text_input("Client Name*", placeholder="e.g., ABC Corporation")
            project_location = st.text_input("Project Location", placeholder="e.g., New York, NY")
            
        with col2:
            project_type = st.selectbox("Project Type*", [
                "Conference Room", "Auditorium", "Classroom", 
                "Control Room", "Retail Space", "Corporate Office",
                "Training Room", "Boardroom", "Multi-Purpose Room"
            ])
            audience_size = st.number_input("Expected Audience Size*", min_value=1, max_value=1000, value=20, 
                                          help="This affects equipment quantities and specifications")
            budget_range = st.selectbox("Budget Range*", [
                "< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"
            ])

        st.subheader("🎯 System Requirements")
        
        col1, col2 = st.columns(2)
        with col1:
            requirements = st.multiselect("Select Required Systems*", [
                "Display Systems", "Audio Systems", "Control Systems",
                "Video Conferencing", "Digital Signage", "Lighting Control",
                "Streaming Solutions"
            ], help="Select all systems needed for your project")
            
        with col2:
            priority_level = st.selectbox("Project Priority", [
                "Standard", "High", "Critical"
            ], help="Affects equipment selection and redundancy")
            
        special_requirements = st.text_area("Special Requirements", 
                                          placeholder="e.g., Must integrate with existing Crestron system, requires wireless presentation, needs recording capability")

        submitted = st.form_submit_button("🚀 Generate Professional BOQ", type="primary")

        if submitted:
            # Enhanced validation
            errors = []
            if not project_name: errors.append("Project Name is required")
            if not client_name: errors.append("Client Name is required")
            if not requirements: errors.append("Please select at least one system requirement")
            if audience_size <= 0: errors.append("Audience size must be greater than 0")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                generator = ProfessionalBOQGenerator(products_data)
                boq_requirements = {
                    'project_name': project_name,
                    'client_name': client_name,
                    'project_type': project_type,
                    'requirements': requirements,
                    'audience_size': audience_size,
                    'budget_range': budget_range,
                    'special_requirements': special_requirements,
                    'priority_level': priority_level,
                    'project_location': project_location
                }
                
                with st.spinner("🔧 Generating professional BOQ..."):
                    boq_data = generator.generate_boq(boq_requirements)
                    st.session_state['current_boq'] = boq_data
                    st.success("✅ Professional BOQ Generated Successfully!")
                    st.balloons()

    # Enhanced BOQ Display
    if 'current_boq' in st.session_state:
        boq_data = st.session_state['current_boq']
        
        st.markdown("---")
        st.header(f"📊 Bill of Quantities: {boq_data['project_name']}")
        
        # Project summary cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", boq_data['item_count'])
        with col2:
            st.metric("Equipment Cost", f"${boq_data['subtotal']:,.2f}")
        with col3:
            st.metric("Total Cost", f"${boq_data['total_cost']:,.2f}")
        with col4:
            budget_util = boq_data.get('budget_utilization', 0)
            st.metric("Budget Utilization", f"{budget_util:.1f}%")

        # BOQ Table with better formatting
        if boq_data['items']:
            df_boq = pd.DataFrame(boq_data['items'])
            
            # Format currency columns
            df_display = df_boq.copy()
            df_display['Unit Cost'] = df_display['Unit Cost'].apply(lambda x: f"${x:,.2f}")
            df_display['Total Cost'] = df_display['Total Cost'].apply(lambda x: f"${x:,.2f}")
            
            st.subheader("📋 Detailed Line Items")
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Category breakdown
            st.subheader("📊 Cost Breakdown by Category")
            category_totals = df_boq.groupby('Category')['Total Cost'].sum().sort_values(ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_pie = px.pie(
                    values=category_totals.values, 
                    names=category_totals.index,
                    title="Cost Distribution by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.write("**Category Totals:**")
                for category, total in category_totals.items():
                    st.write(f"• {category}: ${total:,.2f}")

            # Export options
            st.subheader("📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df_boq.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📄 Download CSV", 
                    csv_data, 
                    f"BOQ_{boq_data['project_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

            with col2:
                json_data = json.dumps(boq_data, indent=2, default=str)
                st.download_button(
                    "📋 Download JSON", 
                    json_data, 
                    f"BOQ_{boq_data['project_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json"
                )
            
            with col3:
                # Create a formatted report
                report_text = f"""
BILL OF QUANTITIES
===================

Project: {boq_data['project_name']}
Client: {boq_data['client_name']}
Project Type: {boq_data['project_type']}
Generated: {boq_data['generated_date']}

SUMMARY
-------
Total Items: {boq_data['item_count']}
Equipment Subtotal: ${boq_data['subtotal']:,.2f}
Contingency: ${boq_data['contingency']:,.2f}
Total Project Cost: ${boq_data['total_cost']:,.2f}

DETAILED BREAKDOWN
------------------
"""
                for _, item in df_boq.iterrows():
                    report_text += f"{item['Description']} | {item['Company']} | {item['Quantity']} {item['Unit']} | ${item['Total Cost']:,.2f}\n"
                
                st.download_button(
                    "📑 Download Report", 
                    report_text, 
                    f"BOQ_Report_{boq_data['project_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    "text/plain"
                )
        else:
            st.warning("⚠️ No suitable products found. Try adjusting the budget range or requirements.")

# Keep existing database and analytics functions but use the improved BOQ generator
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
                    from your_existing_code import process_csv_data  # Import your existing function
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

# Missing components for the Streamlit BOQ Generator

# 1. Complete the analytics page function
def show_analytics_page():
    st.header("📈 Analytics Dashboard")
    products_data = load_product_data()

    if not products_data:
        st.warning("No product data available for analytics. Please upload data on the 'Product Database' page.")
        return

    df = pd.DataFrame(products_data)
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution by Category")
        if 'price' in df.columns and 'category' in df.columns:
            fig_box = px.box(df, x='category', y='price', title="Product Prices by Category")
            st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.subheader("Product Count by Category")
        category_counts = df['category'].value_counts()
        fig_bar = px.bar(x=category_counts.index, y=category_counts.values, 
                        title="Products per Category")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Company analysis
    st.subheader("Company Analysis")
    if 'company' in df.columns:
        company_stats = df.groupby('company').agg({
            'price': ['count', 'mean', 'min', 'max']
        }).round(2)
        company_stats.columns = ['Product Count', 'Avg Price', 'Min Price', 'Max Price']
        st.dataframe(company_stats)

# 2. CSV Processing function
def process_csv_data(df):
    """Process uploaded CSV data and convert to standard format."""
    processed_products = []
    
    # Map common column names to our standard format
    column_mapping = {
        'company': ['company', 'manufacturer', 'brand', 'vendor'],
        'model_no': ['model_no', 'model', 'part_number', 'sku', 'product_code'],
        'description': ['description', 'product_name', 'name', 'title'],
        'category': ['category', 'type', 'product_type', 'class'],
        'price': ['price', 'cost', 'unit_price', 'retail_price', 'list_price']
    }
    
    # Find matching columns
    mapped_columns = {}
    for standard_col, possible_names in column_mapping.items():
        for col_name in df.columns:
            if col_name.lower().strip() in [name.lower() for name in possible_names]:
                mapped_columns[standard_col] = col_name
                break
    
    if not mapped_columns:
        st.error("Could not find recognizable columns in CSV. Please ensure columns are named appropriately.")
        return []
    
    for _, row in df.iterrows():
        try:
            product = {}
            
            # Extract data using mapped columns
            for standard_col, csv_col in mapped_columns.items():
                if csv_col in row:
                    value = row[csv_col]
                    if pd.isna(value):
                        value = ""
                    product[standard_col] = str(value).strip()
            
            # Clean and validate price
            if 'price' in product:
                try:
                    # Remove currency symbols and convert to float
                    price_str = str(product['price']).replace('$', '').replace(',', '').strip()
                    product['price'] = float(price_str) if price_str else 0.0
                except (ValueError, TypeError):
                    product['price'] = 0.0
            else:
                product['price'] = 0.0
            
            # Ensure required fields exist
            required_fields = ['company', 'description', 'category']
            if all(product.get(field, '').strip() for field in required_fields):
                processed_products.append(product)
                
        except Exception as e:
            st.warning(f"Error processing row: {e}")
            continue
    
    return processed_products

# 3. Main application setup and navigation
def main():
    # Sidebar navigation
    st.sidebar.title("🎛️ AV BOQ Generator")
    st.sidebar.markdown("Professional Audio-Visual Bill of Quantities Generator")
    
    page = st.sidebar.selectbox("Select Page", [
        "BOQ Generator", 
        "Product Database", 
        "Analytics",
        "Help & Documentation"
    ])
    
    if page == "BOQ Generator":
        show_boq_generator_page()
    elif page == "Product Database":
        show_product_database_page()
    elif page == "Analytics":
        show_analytics_page()
    elif page == "Help & Documentation":
        show_help_page()

# 4. Help page function
def show_help_page():
    st.header("📚 Help & Documentation")
    
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        ### Quick Start Guide
        1. **Upload Product Data**: Go to 'Product Database' and upload your CSV file with product information
        2. **Generate BOQ**: Use the 'BOQ Generator' to create professional quotes
        3. **Review Analytics**: Check the 'Analytics' page for insights into your product database
        
        ### CSV Format Requirements
        Your CSV should include these columns (names are flexible):
        - **Company/Manufacturer**: Brand name
        - **Model/Part Number**: Product identifier
        - **Description/Name**: Product description
        - **Category/Type**: Product category (display, audio, video, control, mounting, cable)
        - **Price/Cost**: Unit price (numbers only, currency symbols will be removed)
        """)
    
    with st.expander("💡 BOQ Generation Tips"):
        st.markdown("""
        ### Best Practices
        - **Accurate Audience Size**: This directly affects equipment quantities
        - **Realistic Budget**: Set appropriate budget ranges for better product selection
        - **Multiple Requirements**: Select all systems needed for comprehensive coverage
        - **Special Requirements**: Include integration needs and specific brand requirements
        
        ### Budget Guidelines
        - **< $10,000**: Small meeting rooms, basic setups
        - **$10,000 - $50,000**: Standard conference rooms, classrooms
        - **$50,000 - $100,000**: Large conference rooms, small auditoriums
        - **> $100,000**: Large auditoriums, complex installations
        """)
    
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        ### System Categories
        - **Display Systems**: Monitors, projectors, LED walls
        - **Audio Systems**: Speakers, microphones, amplifiers
        - **Control Systems**: Touch panels, processors, automation
        - **Video Conferencing**: Cameras, codecs, collaboration tools
        - **Digital Signage**: Display management, content systems
        - **Mounting**: Brackets, racks, installation hardware
        - **Cables**: HDMI, audio, network, power cables
        
        ### Quantity Calculations
        The system automatically calculates quantities based on:
        - Room type and expected audience size
        - Industry best practices for AV installations
        - Logical dependencies between systems
        """)

    # Sample CSV download
    st.subheader("📥 Sample CSV Template")
    sample_data = pd.DataFrame([
        {"company": "Samsung", "model_no": "QM65R", "description": "65\" 4K Commercial Display", 
         "category": "display", "price": 2200.00},
        {"company": "Bose", "model_no": "EdgeMax EM180", "description": "Ceiling Speaker", 
         "category": "audio", "price": 320.00},
        {"company": "Crestron", "model_no": "CP4", "description": "Control Processor", 
         "category": "control", "price": 2800.00}
    ])
    
    csv_sample = sample_data.to_csv(index=False)
    st.download_button(
        "Download Sample CSV Template",
        csv_sample,
        "sample_products.csv",
        "text/csv"
    )

# 5. Enhanced product database management
def show_product_database_page():
    st.header("📁 Product Database Management")
    products_data = load_product_data()

    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    if products_data:
        df = pd.DataFrame(products_data)
        with col1:
            st.metric("Total Products", len(products_data))
        with col2:
            st.metric("Categories", df['category'].nunique() if 'category' in df.columns else 0)
        with col3:
            st.metric("Companies", df['company'].nunique() if 'company' in df.columns else 0)
        with col4:
            avg_price = df['price'].mean() if 'price' in df.columns else 0
            st.metric("Avg Price", f"${avg_price:,.0f}")

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("📤 Upload & Process Data")
        uploaded_csv = st.file_uploader("Upload product catalog (CSV)", type=['csv'])
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                st.success(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns!")
                
                # Preview data
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column mapping interface
                st.subheader("🔗 Column Mapping")
                st.info("Map your CSV columns to the standard format:")
                
                mapping = {}
                standard_fields = ['company', 'model_no', 'description', 'category', 'price']
                
                for field in standard_fields:
                    selected_col = st.selectbox(
                        f"Select column for {field.replace('_', ' ').title()}:",
                        ['-- Select --'] + list(df.columns),
                        key=f"mapping_{field}"
                    )
                    if selected_col != '-- Select --':
                        mapping[field] = selected_col

                if st.button("🔄 Process and Add to Database", type="primary"):
                    if len(mapping) >= 3:  # At least 3 fields mapped
                        with st.spinner("Processing CSV data..."):
                            processed_products = process_csv_with_mapping(df, mapping)
                            if processed_products:
                                # Merge with existing data
                                existing_data = load_product_data()
                                all_products = existing_data + processed_products
                                
                                if save_product_data(all_products):
                                    st.success(f"✅ Successfully added {len(processed_products)} products to database!")
                                    st.rerun()
                            else:
                                st.error("❌ No valid products were processed. Check your data and mapping.")
                    else:
                        st.error("❌ Please map at least Company, Description, and Category fields.")
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")

    with col2:
        st.subheader("🎛️ Database Controls")
        
        if st.button("📊 Load Sample Data", help="This will replace existing data"):
            if save_product_data(create_sample_data()):
                st.success("✅ Sample data loaded!")
                st.rerun()
        
        if products_data:
            st.markdown("---")
            if st.button("🗑️ Clear Database", help="This will delete all products"):
                if save_product_data([]):
                    st.success("✅ Database cleared!")
                    st.rerun()
                    
            # Export current database
            st.markdown("---")
            st.subheader("📥 Export Database")
            
            df_export = pd.DataFrame(products_data)
            csv_export = df_export.to_csv(index=False)
            
            st.download_button(
                "📄 Download Current Database (CSV)",
                csv_export,
                f"product_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

    # Current database view
    if products_data:
        st.markdown("---")
        st.subheader("📊 Current Database")
        
        # Filter options
        df = pd.DataFrame(products_data)
        col1, col2 = st.columns(2)
        
        with col1:
            if 'category' in df.columns:
                categories = ['All'] + sorted(df['category'].unique().tolist())
                selected_category = st.selectbox("Filter by Category:", categories)
        
        with col2:
            if 'company' in df.columns:
                companies = ['All'] + sorted(df['company'].unique().tolist())
                selected_company = st.selectbox("Filter by Company:", companies)
        
        # Apply filters
        filtered_df = df.copy()
        if 'category' in df.columns and selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if 'company' in df.columns and selected_company != 'All':
            filtered_df = filtered_df[filtered_df['company'] == selected_company]
        
        st.dataframe(filtered_df, use_container_width=True, height=400)
        st.info(f"Showing {len(filtered_df)} of {len(df)} products")

def process_csv_with_mapping(df, mapping):
    """Process CSV with user-defined column mapping."""
    processed_products = []
    
    for _, row in df.iterrows():
        try:
            product = {}
            
            # Map columns based on user selection
            for standard_field, csv_column in mapping.items():
                if csv_column in row:
                    value = row[csv_column]
                    if pd.isna(value):
                        value = ""
                    product[standard_field] = str(value).strip()
            
            # Handle price conversion
            if 'price' in product:
                try:
                    price_str = str(product['price']).replace('$', '').replace(',', '').strip()
                    product['price'] = float(price_str) if price_str else 0.0
                except (ValueError, TypeError):
                    product['price'] = 0.0
            else:
                product['price'] = 0.0
            
            # Validate required fields
            if product.get('description', '').strip() and product.get('category', '').strip():
                processed_products.append(product)
                
        except Exception as e:
            continue
    
    return processed_products

# 6. Run the application
if __name__ == "__main__":
    main()
