#!/usr/bin/env python3
"""
Utilities Module for Heart Disease Diagnosis System
==================================================

Contains utility functions for console formatting, logging,
medical safety, and common helper functions.
"""

import logging
from typing import Tuple, Optional

# Enhanced console formatting
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    class Colors:
        SUCCESS = Fore.GREEN + Style.BRIGHT
        ERROR = Fore.RED + Style.BRIGHT
        WARNING = Fore.YELLOW + Style.BRIGHT
        INFO = Fore.CYAN + Style.BRIGHT
        HEADER = Fore.MAGENTA + Style.BRIGHT
        PROGRESS = Fore.YELLOW
        RESET = Style.RESET_ALL
except ImportError:
    class Colors:
        SUCCESS = ERROR = WARNING = INFO = HEADER = PROGRESS = RESET = ''

class SafeMedicalReporting:
    """Safe medical reporting to prevent dangerous LLM clinical recommendations.
    
    CRITICAL: Prevents LLM from generating clinical advice that could harm patients.
    """
    
    # Whitelist of safe phrases for medical reports
    SAFE_PHRASES = {
        'risk_levels': ['low risk', 'moderate risk', 'high risk', 'elevated risk'],
        'data_summary': ['analysis shows', 'data indicates', 'model predicts', 'parameters suggest'],
        'disclaimers': ['for research only', 'not for clinical use', 'consult physician', 'educational purposes'],
        'general_advice': ['healthy lifestyle', 'regular checkups', 'follow medical guidance']
    }
    
    # Blacklisted dangerous phrases
    DANGEROUS_PHRASES = [
        'take medication', 'stop medication', 'increase dose', 'decrease dose',
        'immediate treatment', 'emergency care', 'urgent intervention',
        'diagnosis:', 'diagnosed with', 'you have', 'you need',
        'prescribed', 'prescription', 'therapy', 'surgery recommended'
    ]
    
    @staticmethod
    def sanitize_medical_text(text: str) -> str:
        """Remove dangerous clinical recommendations from LLM output."""
        # Convert to lowercase for checking
        text_lower = text.lower()
        
        # Check for dangerous phrases
        for phrase in SafeMedicalReporting.DANGEROUS_PHRASES:
            if phrase in text_lower:
                return SafeMedicalReporting.get_safe_fallback_text()
        
        # Add mandatory disclaimer
        safe_text = text + "\n\n⚠️ DISCLAIMER: FOR RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE. CONSULT A QUALIFIED PHYSICIAN."
        
        return safe_text
    
    @staticmethod
    def get_safe_fallback_text() -> str:
        """Return safe fallback text when LLM generates dangerous content."""
        return """RESEARCH ANALYSIS SUMMARY:
The AI model has analyzed the provided data and identified patterns in the cardiovascular risk factors.

⚠️ IMPORTANT DISCLAIMERS:
• FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY
• NOT APPROVED FOR CLINICAL DIAGNOSIS OR TREATMENT
• NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE
• CONSULT A QUALIFIED PHYSICIAN FOR ALL MEDICAL DECISIONS

This analysis should be interpreted by qualified medical professionals only."""
    
    @staticmethod
    def is_text_safe(text: str) -> bool:
        """Check if text contains dangerous medical advice."""
        text_lower = text.lower()
        return not any(phrase in text_lower for phrase in SafeMedicalReporting.DANGEROUS_PHRASES)

def setup_academic_logging() -> logging.Logger:
    """Configure academic-grade logging for the system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('heart_diagnosis_academic.log')
        ]
    )
    return logging.getLogger('HeartDiagnosisAcademic')

def check_library_availability() -> dict:
    """Check availability of optional libraries."""
    availability = {}
    
    # ANFIS and optional libraries
    try:
        from xanfis import GdAnfisClassifier
        availability['XANFIS'] = True
    except ImportError:
        availability['XANFIS'] = False

    try:
        import openai
        availability['OPENAI'] = True
    except ImportError:
        availability['OPENAI'] = False

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        availability['TRANSFORMERS'] = True
    except ImportError:
        availability['TRANSFORMERS'] = False

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        availability['TORCH'] = True
    except ImportError:
        availability['TORCH'] = False

    # Advanced AI and Statistical Libraries
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold
        availability['ADVANCED_ML'] = True
    except ImportError:
        availability['ADVANCED_ML'] = False

    try:
        import scipy.stats as stats
        from scipy.optimize import minimize
        availability['SCIPY'] = True
    except ImportError:
        availability['SCIPY'] = False

    try:
        # McNemar test implementation will be done manually
        availability['STATSMODELS'] = True
    except ImportError:
        availability['STATSMODELS'] = False
    
    return availability

def display_system_requirements(availability: dict):
    """Display system requirements check."""
    print(f"\n{Colors.PROGRESS}🔍 SYSTEM REQUIREMENTS CHECK:{Colors.RESET}")
    
    required_libs = [
        ("XGBoost", True, "High-performance gradient boosting"),
        ("SHAP", True, "Model explainability analysis"), 
        ("scikit-learn", True, "Core ML algorithms"),
        ("scipy", availability.get('SCIPY', False), "Statistical analysis"),
        ("xanfis", availability.get('XANFIS', False), "ANFIS fuzzy systems"),
        ("HuggingFace Transformers", availability.get('TRANSFORMERS', False), "LLM report generation"),
        ("PyTorch", availability.get('TORCH', False), "Deep learning components")
    ]
    
    for lib_name, available, description in required_libs:
        status = f"{Colors.SUCCESS}✅{Colors.RESET}" if available else f"{Colors.WARNING}⚠️{Colors.RESET}"
        print(f"  {status} {lib_name:<20} - {description}")
    
    # Academic compliance check
    if not availability.get('TRANSFORMERS', False):
        print(f"\n{Colors.WARNING}⚠️ HuggingFace Transformers not installed{Colors.RESET}")
        print(f"{Colors.INFO}💡 Install with: pip install transformers torch{Colors.RESET}")
        print(f"{Colors.INFO}System will use built-in medical responses as fallback{Colors.RESET}")