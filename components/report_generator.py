import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from typing import Dict, List, Any
from datetime import datetime
import json
import io
import base64

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        styles = {}
        
        styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f77b4'),
            alignment=1  # Center alignment
        )
        
        styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#333333'),
            borderWidth=1,
            borderColor=colors.HexColor('#1f77b4'),
            borderPadding=5
        )
        
        styles['CustomNormal'] = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        return styles
    
    def generate_pdf_report(self, boq_data: Dict[str, Any]) -> bytes:
        """Generate PDF report from BOQ data"""
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Add title
        story.append(Paragraph("BILL OF QUANTITIES", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Add project details
        story.extend(self._create_project_section(boq_data))
        story.append(Spacer(1, 20))
        
        # Add BOQ table
        story.extend(self._create_boq_table(boq_data['items']))
        story.append(Spacer(1, 20))
        
        # Add summary
        story.extend(self._create_summary_section(boq_data))
        story.append(Spacer(1, 20))
        
        # Add recommendations
        if boq_data.get('recommendations'):
            story.extend(self._create_recommendations_section(boq_data['recommendations']))
        
        # Add compliance details
        story.extend(self._create_compliance_section(boq_data))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    def _create_project_section(self, boq_data: Dict[str, Any]) -> List:
        """Create project details section"""
        elements = []
        
        elements.append(Paragraph("PROJECT DETAILS", self.custom_styles['CustomHeading']))
        
        project_details = [
            ['Project Name:', boq_data.get('project_name', 'N/A')],
            ['Client Name:', boq_data.get('client_name', 'N/A')],
            ['Project Type:', boq_data.get('project_type', 'N/A')],
            ['Generated Date:', datetime.fromisoformat(boq_data.get('generated_date', datetime.now().isoformat())).strftime('%B %d, %Y')],
            ['Total Items:', str(len(boq_data.get('items', [])))],
            ['Total Cost:', f"${boq_data.get('total_cost', 0):,.2f}"],
            ['AVIXA Compliance:', f"{boq_data.get('compliance_score', 0)}%"]
        ]
        
        table = Table(project_details, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        return elements
    
    def _create_boq_table(self, items: List[Dict[str, Any]]) -> List:
        """Create BOQ items table"""
        elements = []
        
        elements.append(Paragraph("BILL OF QUANTITIES ITEMS", self.custom_styles['CustomHeading']))
        
        # Table headers
        headers = ['S.No.', 'Category', 'Description', 'Model No.', 'Brand', 'Qty', 'Unit', 'Unit Cost', 'Total Cost']
        
        # Table data
        table_data = [headers]
        
        for i, item in enumerate(items, 1):
            row = [
                str(i),
                item.get('category', '').title(),
                item.get('description', '')[:40] + '...' if len(item.get('description', '')) > 40 else item.get('description', ''),
                item.get('model_no', ''),
                item.get('company', ''),
                str(item.get('quantity', 0)),
                item.get('unit', ''),
                f"${item.get('unit_cost', 0):,.2f}",
                f"${item.get('total_cost', 0):,.2f}"
            ]
            table_data.append(row)
        
        # Add total row
        total_cost = sum(item.get('total_cost', 0) for item in items)
        table_data.append(['', '', '', '', '', '', '', 'TOTAL:', f"${total_cost:,.2f}"])
        
        table = Table(table_data, colWidths=[0.5*inch, 0.8*inch, 2.5*inch, 1*inch, 0.8*inch, 0.5*inch, 0.5*inch, 0.8*inch, 0.8*inch])
        
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            
            # Data styling
            ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -2), 8),
            ('GRID', (0, 0), (-1, -2), 1, colors.black),
            
            # Total row styling
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, -1), (-1, -1), 9),
            ('LINEABOVE', (0, -1), (-1, -1), 2, colors.black),
            
            # Alignment
            ('ALIGN', (5, 1), (8, -1), 'RIGHT'),  # Right align numeric columns
            ('ALIGN', (1, 1), (4, -2), 'LEFT'),   # Left align text columns
        ]))
        
        elements.append(table)
        return elements
    
    def _create_summary_section(self, boq_data: Dict[str, Any]) -> List:
        """Create summary section"""
        elements = []
        
        elements.append(Paragraph("PROJECT SUMMARY", self.custom_styles['CustomHeading']))
        
        # Calculate category-wise breakdown
        items = boq_data.get('items', [])
        category_breakdown = {}
        
        for item in items:
            category = item.get('category', 'Other')
            if category not in category_breakdown:
                category_breakdown[category] = {'count': 0, 'cost': 0}
            
            category_breakdown[category]['count'] += item.get('quantity', 0)
            category_breakdown[category]['cost'] += item.get('total_cost', 0)
        
        summary_data = [['Category', 'Items', 'Total Cost', 'Percentage']]
        total_cost = boq_data.get('total_cost', 0)
        
        for category, data in category_breakdown.items():
            percentage = (data['cost'] / total_cost * 100) if total_cost > 0 else 0
            summary_data.append([
                category.title(),
                str(data['count']),
                f"${data['cost']:,.2f}",
                f"{percentage:.1f}%"
            ])
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1*inch, 1.5*inch, 1*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        
        elements.append(summary_table)
        return elements
    
    def _create_recommendations_section(self, recommendations: List[str]) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("RECOMMENDATIONS", self.custom_styles['CustomHeading']))
        
        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {rec}", self.custom_styles['CustomNormal']))
        
        return elements
    
    def _create_compliance_section(self, boq_data: Dict[str, Any]) -> List:
        """Create AVIXA compliance section"""
        elements = []
        
        elements.append(Paragraph("AVIXA COMPLIANCE", self.custom_styles['CustomHeading']))
        
        compliance_score = boq_data.get('compliance_score', 0)
        
        compliance_text = f"""
        This project achieves {compliance_score}% compliance with AVIXA (Audiovisual and Integrated Experience Association) 
        standards and best practices. The evaluation covers system design, component selection, 
        installation requirements, and operational guidelines.
        """
        
        elements.append(Paragraph(compliance_text, self.custom_styles['CustomNormal']))
        
        if compliance_score >= 90:
            status = "EXCELLENT - Exceeds industry standards"
            color = colors.green
        elif compliance_score >= 80:
            status = "GOOD - Meets industry standards"
            color = colors.blue
        elif compliance_score >= 70:
            status = "ACCEPTABLE - Minor improvements needed"
            color = colors.orange
        else:
            status = "NEEDS IMPROVEMENT - Significant gaps identified"
            color = colors.red
        
        status_style = ParagraphStyle(
            'Status',
            parent=self.custom_styles['CustomNormal'],
            textColor=color,
            fontSize=12,
            fontName='Helvetica-Bold'
        )
        
        elements.append(Paragraph(f"Compliance Status: {status}", status_style))
        
        return elements
    
    def generate_excel_report(self, boq_data: Dict[str, Any]) -> bytes:
        """Generate Excel report from BOQ data"""
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # BOQ Items sheet
            if boq_data.get('items'):
                df_items = pd.DataFrame(boq_data['items'])
                df_items.to_excel(writer, sheet_name='BOQ Items', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': ['Project Name', 'Client Name', 'Total Items', 'Total Cost', 'AVIXA Compliance'],
                'Value': [
                    boq_data.get('project_name', 'N/A'),
                    boq_data.get('client_name', 'N/A'),
                    len(boq_data.get('items', [])),
                    f"${boq_data.get('total_cost', 0):,.2f}",
                    f"{boq_data.get('compliance_score', 0)}%"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Recommendations sheet
            if boq_data.get('recommendations'):
                df_recs = pd.DataFrame({
                    'Recommendation': boq_data['recommendations']
                })
                df_recs.to_excel(writer, sheet_name='Recommendations', index=False)
        
        buffer.seek(0)
        return buffer.read()
