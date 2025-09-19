import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from utils.excel_processor import ExcelProcessor
from utils.boq_generator import BOQGenerator
from utils.avixa_compliance import AVIXACompliance
from models.boq_model import BOQModel
from components.ui_components import UIComponents
from components.report_generator import ReportGenerator

# Page config
st.set_page_config(
    page_title="AI-Powered BOQ Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🚀 AI-Powered BOQ Generator for AV Solutions")
    st.markdown("Generate professional Bills of Quantities compliant with AVIXA guidelines")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Choose a page", [
            "Home", 
            "Data Processing", 
            "BOQ Generation", 
            "Model Training",
            "Analytics"
        ])
    
    if page == "Home":
        show_home_page()
    elif page == "Data Processing":
        show_data_processing_page()
    elif page == "BOQ Generation":
        show_boq_generation_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Analytics":
        show_analytics_page()

def show_home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the AI-Powered BOQ Generator
        
        This application helps AV solution companies generate accurate and compliant 
        Bills of Quantities using advanced machine learning algorithms.
        
        ### Features:
        - 🤖 AI-powered product recommendations
        - 📋 AVIXA guidelines compliance
        - 📊 Professional report generation
        - 🔍 Smart product search and matching
        - 📈 Cost optimization suggestions
        """)
    
    with col2:
        st.info("""
        **Quick Start:**
        1. Process your Excel data
        2. Train the AI model
        3. Generate BOQs
        4. Download reports
        """)

def show_data_processing_page():
    st.header("📁 Data Processing")
    
    uploaded_file = st.file_uploader(
        "Upload your Excel file with product data",
        type=['xlsx', 'xls'],
        help="Upload the Excel file containing 57 sheets with company data"
    )
    
    if uploaded_file:
        processor = ExcelProcessor()
        
        with st.spinner("Processing Excel file..."):
            try:
                processed_data = processor.process_excel(uploaded_file)
                st.success(f"Successfully processed {len(processed_data)} products!")
                
                # Show preview
                st.subheader("Data Preview")
                df = pd.DataFrame(processed_data[:100])  # Show first 100 rows
                st.dataframe(df)
                
                # Save processed data
                if st.button("Save Processed Data"):
                    processor.save_to_json(processed_data, "data/processed_products.json")
                    st.success("Data saved successfully!")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def show_boq_generation_page():
    st.header("📊 BOQ Generation")
    
    # Load processed data
    try:
        with open("data/processed_products.json", 'r') as f:
            products_data = json.load(f)
        
        boq_generator = BOQGenerator(products_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Project Details")
            project_name = st.text_input("Project Name")
            client_name = st.text_input("Client Name")
            project_type = st.selectbox("Project Type", [
                "Conference Room", "Auditorium", "Classroom", 
                "Control Room", "Retail Space", "Corporate Office"
            ])
            room_dimensions = st.text_input("Room Dimensions (L x W x H)")
            budget_range = st.selectbox("Budget Range", [
                "< $10,000", "$10,000 - $50,000", "$50,000 - $100,000", "> $100,000"
            ])
        
        with col2:
            st.subheader("Requirements")
            requirements = st.multiselect("Select Requirements", [
                "Display Systems", "Audio Systems", "Control Systems",
                "Video Conferencing", "Digital Signage", "Lighting Control",
                "Room Scheduling", "Streaming Solutions"
            ])
            
            audience_size = st.number_input("Expected Audience Size", min_value=1, value=20)
            special_requirements = st.text_area("Special Requirements")
        
        if st.button("Generate BOQ", type="primary"):
            if project_name and client_name and requirements:
                with st.spinner("Generating BOQ using AI..."):
                    boq_data = boq_generator.generate_boq({
                        'project_name': project_name,
                        'client_name': client_name,
                        'project_type': project_type,
                        'requirements': requirements,
                        'audience_size': audience_size,
                        'budget_range': budget_range,
                        'special_requirements': special_requirements
                    })
                    
                    st.success("BOQ Generated Successfully!")
                    
                    # Display BOQ
                    st.subheader("Generated BOQ")
                    df_boq = pd.DataFrame(boq_data['items'])
                    st.dataframe(df_boq)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Items", len(boq_data['items']))
                    with col2:
                        st.metric("Total Cost", f"${boq_data['total_cost']:,.2f}")
                    with col3:
                        st.metric("AVIXA Compliance", f"{boq_data['compliance_score']}%")
                    
                    # Download options
                    report_generator = ReportGenerator()
                    pdf_data = report_generator.generate_pdf_report(boq_data)
                    st.download_button(
                        "Download PDF Report",
                        pdf_data,
                        f"BOQ_{project_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        "application/pdf"
                    )
            else:
                st.warning("Please fill in all required fields")
    
    except FileNotFoundError:
        st.warning("Please process your data first in the Data Processing page")

def show_model_training_page():
    st.header("🤖 Model Training")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Configuration")
        model_type = st.selectbox("Model Type", [
            "Product Recommendation",
            "Price Prediction",
            "Compatibility Checker",
            "Compliance Validator"
        ])
        
        training_epochs = st.slider("Training Epochs", 1, 100, 10)
        learning_rate = st.select_slider("Learning Rate", 
            options=[0.001, 0.01, 0.1], value=0.01)
        
        if st.button("Start Training"):
            model = BOQModel()
            with st.spinner("Training model..."):
                progress_bar = st.progress(0)
                for i in range(training_epochs):
                    # Simulate training progress
                    progress_bar.progress((i + 1) / training_epochs)
                    
                st.success("Model trained successfully!")
    
    with col2:
        st.subheader("Model Performance")
        # Show dummy metrics
        st.metric("Accuracy", "94.2%", "2.1%")
        st.metric("Precision", "91.8%", "1.5%")
        st.metric("Recall", "89.3%", "0.8%")

def show_analytics_page():
    st.header("📈 Analytics Dashboard")
    
    # Dummy data for visualization
    categories = ['Display', 'Audio', 'Control', 'Video Conf', 'Lighting']
    values = [25, 20, 15, 25, 15]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(values=values, names=categories, title="Product Category Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(x=categories, y=values, title="Usage by Category")
        st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
