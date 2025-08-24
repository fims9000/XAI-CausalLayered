#!/usr/bin/env python3
"""
Configuration Module for Heart Disease Diagnosis System
======================================================

Contains all configuration settings, constants, and clinical thresholds
for the AI-powered heart disease diagnosis system.
"""

from dataclasses import dataclass
from typing import Dict, List

# Academic Configuration and Data Structures
@dataclass
class ClinicalThresholds:
    """CORRECTED clinical decision thresholds based on medical literature.
    
    References:
    - Goff et al., Circulation, 2014 (ASCVD Risk Calculator)
    - D'Agostino et al., Circulation, 2008 (Framingham Risk Score)
    """
    low_risk_threshold: float = 0.075   # <7.5% 10-year risk
    intermediate_risk_threshold: float = 0.075  # 7.5% boundary  
    high_risk_threshold: float = 0.20   # >20% 10-year risk

@dataclass
class FairnessMetrics:
    """Fairness metrics for bias detection in medical AI.
    
    References:
    - Barocas et al., Fairness and Machine Learning, 2019
    - Chen et al., Nature Medicine, 2021
    """
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    calibration_difference: float = 0.0

@dataclass
class UncertaintyMetrics:
    """Uncertainty quantification metrics.
    
    References:
    - Gal & Ghahramani, ICML, 2016 (Monte Carlo Dropout)
    - Kendall & Gal, NIPS, 2017 (Epistemic vs Aleatoric)
    """
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    predictive_entropy: float = 0.0
    mutual_information: float = 0.0

@dataclass
class XAIMetrics:
    """XAI 2.0 explainability metrics for training integration.
    
    References:
    - Dhurandhar et al., Explanations based on the Missing: Towards Contrastive Explanations, NIPS 2018
    - Guidotti et al., A Survey of Methods for Explaining Black Box Models, ACM Computing Surveys 2018
    """
    fidelity_loss: float = 0.0
    stability_loss: float = 0.0
    simplicity_loss: float = 0.0
    total_xai_loss: float = 0.0

@dataclass
class CalibrationMetrics:
    """Model calibration metrics for XAI 2.0.
    
    References:
    - Guo et al., On Calibration of Modern Neural Networks, ICML 2017
    - Niculescu-Mizil & Caruana, Predicting good probabilities with supervised learning, ICML 2005
    """
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    average_calibration_error: float = 0.0
    brier_score: float = 0.0
    brier_score_decomposition: Dict[str, float] = None
    
    def __post_init__(self):
        if self.brier_score_decomposition is None:
            self.brier_score_decomposition = {
                'reliability': 0.0,
                'resolution': 0.0,
                'uncertainty': 0.0
            }

@dataclass
class ContinualLearningMetrics:
    """Metrics for continual learning performance.
    
    References:
    - Kirkpatrick et al., Overcoming catastrophic forgetting, PNAS 2017
    - Chaudhry et al., Efficient Lifelong Learning with A-GEM, ICLR 2019
    """
    catastrophic_forgetting: float = 0.0
    backward_transfer: float = 0.0
    forward_transfer: float = 0.0
    concept_drift_detected: bool = False
    adaptation_score: float = 0.0

# Feature names for UCI Heart Disease Cleveland dataset
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Medical interpretations for real features
FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (0=female, 1=male)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mmHg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar >120 mg/dl',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (0=no, 1=yes)',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment',
    'ca': 'Number of major vessels colored by fluoroscopy',
    'thal': 'Thalassemia (0=normal, 1=fixed, 2=reversible defect)'
}

# Academic citations for publication
ACADEMIC_CITATIONS = [
    "Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. The American Journal of Cardiology.",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.",
    "Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS 2017.",
    "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why should I trust you? Explaining the predictions of any classifier. KDD 2016.",
    "Jang, J. S. (1993). ANFIS: Adaptive-network-based fuzzy inference system. IEEE Transactions on Systems.",
    "Lin, C. T., & Lee, C. G. (1996). Neural-network-based fuzzy logic control and decision system. IEEE.",
    "Alvarez-Melis, D., & Jaakkola, T. S. (2018). Towards robust interpretability with self-explaining neural networks. NIPS.",
    "Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.",
    "Chaudhry, A., et al. (2019). Efficient lifelong learning with A-GEM. ICLR 2019.",
    "Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. ICML 2016.",
    "Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning. MIT Press.",
    "Pencina, M. J., et al. (2008). Evaluating the added predictive ability of a new marker. Statistics in Medicine."
]