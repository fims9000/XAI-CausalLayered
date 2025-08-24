#!/usr/bin/env python3
"""
Heart Disease Diagnosis System with Interpretable AI Pipeline v2.0
================================================================

A modular, academic-grade heart disease diagnosis system with:
- XGBoost classification
- SHAP explainability 
- ANFIS fuzzy rule extraction
- PyTorch deep learning integration
- Comprehensive trustworthy AI validation

Author: Academic Research Team
Version: 3.0
License: Academic Research Use
"""

__version__ = "3.0"
__author__ = "Academic Research Team"
__email__ = "research@academic.edu"

# Import main components for easy access
from .utils import Colors, SafeMedicalReporting
from .config import ClinicalThresholds, FairnessMetrics
from .data_handler import DataHandler, load_uci_heart_cleveland
from .analysis import XGBoostModel, ANFISModel, SHAPAnalyzer, RuleAggregator
from .models import DNFSModel, XAILoss, ContinualLearner

__all__ = [
    'Colors', 'SafeMedicalReporting',
    'ClinicalThresholds', 'FairnessMetrics', 
    'DataHandler', 'load_uci_heart_cleveland',
    'XGBoostModel', 'ANFISModel', 'SHAPAnalyzer', 'RuleAggregator',
    'DNFSModel', 'XAILoss', 'ContinualLearner'
]