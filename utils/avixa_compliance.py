from typing import Dict, List, Any
from config import Config

class AVIXACompliance:
    def __init__(self):
        self.guidelines = Config.get_avixa_guidelines()
    
    def check_compliance(self, boq_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check BOQ compliance with AVIXA standards"""
        
        compliance_results = {
            'overall_score': 0,
            'category_scores': {},
            'recommendations': [],
            'violations': []
        }
        
        # Check each category
        categories = set(item['category'] for item in boq_items)
        
        for category in categories:
            category_items = [item for item in boq_items if item['category'] == category]
            score = self._check_category_compliance(category, category_items)
            compliance_results['category_scores'][category] = score
        
        # Calculate overall score
        if compliance_results['category_scores']:
            compliance_results['overall_score'] = sum(
                compliance_results['category_scores'].values()
            ) / len(compliance_results['category_scores'])
        
        return compliance_results
    
    def _check_category_compliance(self, category: str, items: List[Dict[str, Any]]) -> int:
        """Check compliance for specific category"""
        if category == 'display':
            return self._check_display_compliance(items)
        elif category == 'audio':
            return self._check_audio_compliance(items)
        elif category == 'control':
            return self._check_control_compliance(items)
        else:
            return 80  # Default score for other categories
    
    def _check_display_compliance(self, items: List[Dict[str, Any]]) -> int:
        """Check display system compliance"""
        score = 70  # Base score
        
        # Check if displays are present
        if items:
            score += 20
        
        # Additional checks can be added here
        
        return min(100, score)
    
    def _check_audio_compliance(self, items: List[Dict[str, Any]]) -> int:
        """Check audio system compliance"""
        score = 70  # Base score
        
        # Check coverage
        if len(items) >= 2:  # Multiple audio sources
            score += 15
        
        # Check for amplification
        has_amplifier = any('amplifier' in item['description'].lower() for item in items)
        if has_amplifier:
            score += 15
        
        return min(100, score)
    
    def _check_control_compliance(self, items: List[Dict[str, Any]]) -> int:
        """Check control system compliance"""
        score = 80  # Base score for having control system
        
        # Check for redundancy if required
        if self.guidelines['control_redundancy'] and len(items) > 1:
            score += 20
        
        return min(100, score)
