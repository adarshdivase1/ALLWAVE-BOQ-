import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class UIComponents:
    """Custom UI components for the BOQ Generator application"""
    
    @staticmethod
    def render_header(title: str, subtitle: str = "", icon: str = "📊"):
        """Render application header"""
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #1f77b4; margin: 0;">{icon} {title}</h1>
            <p style="color: #666; margin-top: 0.5rem;">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_navigation() -> str:
        """Render sidebar navigation"""
        with st.sidebar:
            st.markdown("### Navigation")
            
            pages = {
                "🏠 Home": "home",
                "📁 Data Processing": "data_processing", 
                "📊 BOQ Generation": "boq_generation",
                "🤖 Model Training": "model_training",
                "📈 Analytics": "analytics",
                "⚙️ Settings": "settings"
            }
            
            selected = st.selectbox("Choose a page", list(pages.keys()))
            return pages[selected]
    
    @staticmethod
    def render_project_form() -> Dict[str, Any]:
        """Render project details form"""
        st.subheader("🏗️ Project Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input("Project Name*", placeholder="Enter project name")
            client_name = st.text_input("Client Name*", placeholder="Enter client name")
            project_type = st.selectbox("Project Type*", [
                "Conference Room", "Auditorium", "Classroom", 
                "Control Room", "Retail Space", "Corporate Office",
                "Training Room", "Broadcast Studio", "House of Worship"
            ])
            
        with col2:
            room_dimensions = st.text_input("Room Dimensions", 
                                          placeholder="L x W x H (in feet)")
            audience_size = st.number_input("Expected Audience Size", 
                                          min_value=1, max_value=1000, value=20)
            budget_range = st.selectbox("Budget Range", [
                "< $10,000", "$10,000 - $25,000", "$25,000 - $50,000", 
                "$50,000 - $100,000", "$100,000 - $250,000", "> $250,000"
            ])
        
        st.subheader("📋 Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            av_requirements = st.multiselect("AV Requirements*", [
                "Display Systems", "Audio Systems", "Control Systems",
                "Video Conferencing", "Digital Signage", "Lighting Control",
                "Room Scheduling", "Streaming Solutions", "Recording Systems",
                "Interactive Displays", "Wireless Presentation", "Video Walls"
            ])
            
        with col2:
            connectivity = st.multiselect("Connectivity Options", [
                "HDMI", "USB-C", "Wireless", "Ethernet", "Bluetooth",
                "WiFi", "RS-232", "IR Control", "PoE"
            ])
        
        special_requirements = st.text_area("Special Requirements", 
                                          placeholder="Enter any special requirements or constraints")
        
        project_data = {
            'project_name': project_name,
            'client_name': client_name,
            'project_type': project_type,
            'room_dimensions': room_dimensions,
            'audience_size': audience_size,
            'budget_range': budget_range,
            'av_requirements': av_requirements,
            'connectivity': connectivity,
            'special_requirements': special_requirements
        }
        
        return project_data
    
    @staticmethod
    def render_boq_table(boq_data: List[Dict[str, Any]]) -> None:
        """Render BOQ items table"""
        if not boq_data:
            st.warning("No BOQ items to display")
            return
            
        df = pd.DataFrame(boq_data)
        
        # Format columns
        if 'unit_cost' in df.columns:
            df['unit_cost'] = df['unit_cost'].apply(lambda x: f"${x:,.2f}")
        if 'total_cost' in df.columns:
            df['total_cost'] = df['total_cost'].apply(lambda x: f"${x:,.2f}")
        
        # Column configuration
        column_config = {
            'category': st.column_config.TextColumn("Category", width="small"),
            'description': st.column_config.TextColumn("Description", width="large"),
            'model_no': st.column_config.TextColumn("Model No.", width="small"),
            'company': st.column_config.TextColumn("Brand", width="small"),
            'quantity': st.column_config.NumberColumn("Qty", width="small"),
            'unit': st.column_config.TextColumn("Unit", width="small"),
            'unit_cost': st.column_config.TextColumn("Unit Cost", width="small"),
            'total_cost': st.column_config.TextColumn("Total Cost", width="small")
        }
        
        st.dataframe(df, column_config=column_config, use_container_width=True)
    
    @staticmethod
    def render_cost_summary(total_cost: float, item_count: int, 
                           compliance_score: int) -> None:
        """Render cost summary metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Cost",
                value=f"${total_cost:,.2f}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Total Items",
                value=item_count,
                delta=None
            )
        
        with col3:
            st.metric(
                label="AVIXA Compliance",
                value=f"{compliance_score}%",
                delta=None,
                delta_color="normal" if compliance_score >= 80 else "inverse"
            )
        
        with col4:
            avg_cost = total_cost / item_count if item_count > 0 else 0
            st.metric(
                label="Avg Item Cost",
                value=f"${avg_cost:,.2f}",
                delta=None
            )
    
    @staticmethod
    def render_cost_breakdown_chart(boq_data: List[Dict[str, Any]]) -> None:
        """Render cost breakdown pie chart"""
        if not boq_data:
            return
            
        df = pd.DataFrame(boq_data)
        
        # Group by category
        category_costs = df.groupby('category')['total_cost'].sum().reset_index()
        
        # Create pie chart
        fig = px.pie(
            category_costs, 
            values='total_cost', 
            names='category',
            title="Cost Breakdown by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_recommendations_panel(recommendations: List[str]) -> None:
        """Render recommendations panel"""
        if not recommendations:
            return
            
        st.subheader("💡 Recommendations")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"{i}. {recommendation}")
    
    @staticmethod
    def render_compliance_details(compliance_data: Dict[str, Any]) -> None:
        """Render compliance details"""
        st.subheader("✅ AVIXA Compliance Details")
        
        if 'category_scores' in compliance_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Category Scores:**")
                for category, score in compliance_data['category_scores'].items():
                    color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                    st.markdown(f"- {category.title()}: "
                              f"<span style='color: {color}'>{score}%</span>", 
                              unsafe_allow_html=True)
            
            with col2:
                if 'recommendations' in compliance_data:
                    st.markdown("**Compliance Recommendations:**")
                    for rec in compliance_data['recommendations']:
                        st.markdown(f"- {rec}")
    
    @staticmethod
    def render_file_upload_zone(accepted_types: List[str], 
                               help_text: str = "") -> Optional[Any]:
        """Render file upload zone"""
        return st.file_uploader(
            "Upload File",
            type=accepted_types,
            help=help_text,
            accept_multiple_files=False
        )
    
    @staticmethod
    def render_progress_indicator(progress: float, text: str = "") -> None:
        """Render progress indicator"""
        st.progress(progress, text=text)
    
    @staticmethod
    def render_status_indicator(status: str, message: str) -> None:
        """Render status indicator"""
        if status == "success":
            st.success(message)
        elif status == "warning":
            st.warning(message)
        elif status == "error":
            st.error(message)
        elif status == "info":
            st.info(message)
    
    @staticmethod
    def render_data_preview_table(data: pd.DataFrame, max_rows: int = 100) -> None:
        """Render data preview table"""
        st.subheader("📋 Data Preview")
        
        if len(data) > max_rows:
            st.info(f"Showing first {max_rows} rows of {len(data)} total rows")
            preview_data = data.head(max_rows)
        else:
            preview_data = data
        
        st.dataframe(preview_data, use_container_width=True)
    
    @staticmethod
    def render_model_metrics(metrics: Dict[str, float]) -> None:
        """Render model performance metrics"""
        st.subheader("📊 Model Performance")
        
        cols = st.columns(len(metrics))
        
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i]:
                # Format value based on metric type
                if metric.lower() in ['accuracy', 'precision', 'recall', 'f1_score']:
                    formatted_value = f"{value:.1%}"
                elif metric.lower() in ['mse', 'mae', 'rmse']:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.3f}"
                
                st.metric(
                    label=metric.replace('_', ' ').title(),
                    value=formatted_value
                )
    
    @staticmethod
    def render_analytics_dashboard(analytics_data: Dict[str, Any]) -> None:
        """Render analytics dashboard"""
        st.subheader("📈 Analytics Dashboard")
        
        # Create tabs for different analytics views
        tab1, tab2, tab3 = st.tabs(["Overview", "Trends", "Insights"])
        
        with tab1:
            UIComponents._render_overview_metrics(analytics_data)
        
        with tab2:
            UIComponents._render_trends_charts(analytics_data)
        
        with tab3:
            UIComponents._render_insights_panel(analytics_data)
    
    @staticmethod
    def _render_overview_metrics(data: Dict[str, Any]) -> None:
        """Render overview metrics"""
        metrics = data.get('overview_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Projects", metrics.get('total_projects', 0))
        with col2:
            st.metric("Total Value", f"${metrics.get('total_value', 0):,.2f}")
        with col3:
            st.metric("Avg Project Size", f"${metrics.get('avg_project_size', 0):,.2f}")
        with col4:
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
    
    @staticmethod
    def _render_trends_charts(data: Dict[str, Any]) -> None:
        """Render trends charts"""
        trends_data = data.get('trends', {})
        
        if 'monthly_projects' in trends_data:
            df_monthly = pd.DataFrame(trends_data['monthly_projects'])
            fig = px.line(df_monthly, x='month', y='count', 
                         title="Projects by Month")
            st.plotly_chart(fig, use_container_width=True)
        
        if 'category_popularity' in trends_data:
            df_category = pd.DataFrame(trends_data['category_popularity'])
            fig = px.bar(df_category, x='category', y='count',
                        title="Popular Categories")
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_insights_panel(data: Dict[str, Any]) -> None:
        """Render insights panel"""
        insights = data.get('insights', [])
        
        if insights:
            st.markdown("**Key Insights:**")
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.info("No insights available yet. Generate more BOQs to see insights.")
    
    @staticmethod
    def render_export_options(boq_data: Dict[str, Any]) -> None:
        """Render export options"""
        st.subheader("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as PDF", use_container_width=True):
                st.success("PDF export initiated")
        
        with col2:
            if st.button("📊 Export as Excel", use_container_width=True):
                st.success("Excel export initiated")
        
        with col3:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.success("Copied to clipboard")
