# 🎓 Academic Heart Disease Diagnosis System v3.0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)](README.md)

> **Academic-grade heart disease diagnosis system with trustworthy AI, explainability, and clinical validation**

## 🏗️ Modular Architecture

The system has been refactored from a single 4000-line file into a clean, modular structure:

```
📁 Pr1/
├── 📄 main.py              # Main pipeline orchestrator (clean & simple)
├── 📄 config.py            # Configuration, constants, clinical thresholds
├── 📄 utils.py             # Utilities, colors, logging, safety controls
├── 📄 data_handler.py      # UCI data loading, preprocessing, bias detection
├── 📄 analysis.py          # XGBoost, SHAP, ANFIS, uncertainty quantification
├── 📄 models.py            # PyTorch models (DNFS, XAI Loss, Continual Learning)
├── 📄 requirements.txt     # Full dependencies
├── 📄 requirements-py38.txt # Python 3.8 compatible dependencies
├── 📄 __init__.py          # Package initialization
├── 📄 README.md            # This documentation
└── 📄 main_original_backup.py # Backup of original 4000-line file
```

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Create virtual environment (Python 3.8-3.11 recommended)
python -m venv heart_diagnosis_env
heart_diagnosis_env\Scripts\activate

# Install dependencies
pip install -r requirements-py38.txt  # For Python 3.8
# OR
pip install -r requirements.txt       # For Python 3.9+
```

### 2. **Run the System**
```bash
python main.py
```

### 3. **Expected Output**
The system will execute a 10-step academic pipeline:
1. 📊 Data Loading & Preprocessing  
2. 🚀 XGBoost Training & Calibration
3. 🔍 SHAP Explainability Analysis
4. 🧠 ANFIS Fuzzy Rule Extraction
5. 🤝 Rule Aggregation & Consensus
6. 🎲 Advanced AI Analysis (Uncertainty, Fairness)
7. 🔥 Deep Learning Integration (PyTorch)
8. 📝 AI Explanation Generation
9. 🎓 Academic Validation
10. 📚 Publication Summary

## 📦 Module Overview

### `main.py` - Pipeline Orchestrator
- **Purpose**: Clean entry point with step-by-step execution
- **Lines**: ~350 (down from 4000!)
- **Features**: Academic pipeline with progress tracking

### `config.py` - Configuration Hub  
- **Purpose**: All constants, thresholds, academic citations
- **Contains**: Clinical thresholds, fairness metrics, feature definitions
- **Academic**: References to medical literature

### `utils.py` - Utilities & Safety
- **Purpose**: Common utilities, console colors, medical safety
- **Safety**: `SafeMedicalReporting` prevents dangerous AI advice
- **Logging**: Academic-grade logging system

### `data_handler.py` - Data Processing
- **Purpose**: UCI Heart Disease data loading and preprocessing
- **Features**: Bias detection, differential privacy, fairness analysis
- **Academic**: Epidemiological demographic simulation

### `analysis.py` - ML Analysis Suite
- **Purpose**: XGBoost, SHAP, ANFIS, uncertainty quantification
- **Models**: Temperature scaling, calibration, bootstrap uncertainty
- **Academic**: Counterfactual explanations, causal analysis

### `models.py` - Deep Learning Models
- **Purpose**: PyTorch-based advanced AI components
- **Contains**: DNFS, XAI Loss, Continual Learning, Concept Drift
- **Academic**: State-of-the-art neural-fuzzy systems

## 🎯 Key Improvements

### ✅ **Code Organization**
- **Before**: 4000 lines in one file
- **After**: 6 focused modules (~300-800 lines each)
- **Result**: Easy maintenance, testing, and Git collaboration

### ✅ **Import Structure** 
```python
# Clean, organized imports
from utils import Colors, SafeMedicalReporting
from config import ClinicalThresholds
from data_handler import DataHandler
from analysis import XGBoostModel, SHAPAnalyzer
from models import DNFSModel
```

### ✅ **Git-Friendly**
- Separate files for different features
- Easy to track changes per component
- Better collaboration workflow
- Meaningful commit history

### ✅ **Academic Standards**
- Clear module responsibilities
- Comprehensive documentation
- Academic citations preserved
- Publication-ready structure

## 🔬 Academic Features

### **Trustworthy AI Pipeline**
- 🎲 **Uncertainty Quantification**: Monte Carlo dropout, epistemic/aleatoric uncertainty
- ⚖️ **Fairness Analysis**: Demographic parity, equalized odds, calibration
- 🔄 **Counterfactual Explanations**: Actionable interventions for risk reduction
- 🔗 **Causal Inference**: Backdoor criterion, do-calculus interventions
- 🔒 **Differential Privacy**: (ε,δ)-guarantees for patient data protection

### **Explainable AI (XAI)**
- 🔍 **SHAP Values**: TreeExplainer for feature importance
- 🧠 **ANFIS Rules**: Fuzzy logic interpretability  
- 🤖 **Deep Neuro-Fuzzy**: XAI 2.0 with learnable rules
- 📊 **Consensus Analysis**: Multi-method agreement

### **Clinical Validation**
- 🏥 **UCI Heart Disease Dataset**: Real cardiovascular features
- 📈 **Temperature Scaling**: Model calibration (Guo et al., 2017)
- 🎯 **Bootstrap Confidence**: Statistical significance testing
- 📋 **Medical Safety**: LLM output sanitization

## 🧪 Testing & Validation

### **Run Individual Modules**
```python
# Test data loading
from data_handler import DataHandler
handler = DataHandler()
handler.load_and_preprocess()

# Test XGBoost model
from analysis import XGBoostModel
model = XGBoostModel()
model.train(X_train, y_train, X_test, y_test)

# Test PyTorch components (if available)
from models import DNFSModel
dnfs = DNFSModel(input_dim=13)
```

### **Academic Compliance**
- ✅ FDA SaMD Classification Ready
- ✅ GDPR Right-to-Explanation Compliant  
- ✅ Peer Review Documentation
- ✅ Reproducible Research Standards

## 📚 Publications & Citations

The system includes **13 academic citations** covering:
- Machine Learning: XGBoost (Chen & Guestrin, 2016)
- Explainability: SHAP (Lundberg & Lee, 2017), LIME (Ribeiro et al., 2016)
- Fuzzy Systems: ANFIS (Jang, 1993), Neural-Fuzzy (Lin & Lee, 1996)
- Uncertainty: Monte Carlo Dropout (Gal & Ghahramani, 2016)
- Fairness: Algorithmic Fairness (Barocas et al., 2019)
- Privacy: Differential Privacy (Dwork & Roth, 2014)
- Medical: UCI Heart Disease (Detrano et al., 1989)

## 🛠️ Development

### **Adding New Features**
1. Choose appropriate module (or create new one)
2. Follow existing patterns and documentation
3. Update `__init__.py` if adding public APIs
4. Add academic references if applicable

### **Module Dependencies**
```python
config.py          # ← No dependencies (base configuration)
utils.py           # ← Depends on: config.py  
data_handler.py    # ← Depends on: utils.py, config.py
analysis.py        # ← Depends on: utils.py, config.py, models.py
models.py          # ← Depends on: config.py (minimal PyTorch dependency)
main.py           # ← Orchestrates all modules
```

## ⚠️ Important Notes

### **Medical Safety**
- 🔒 All AI outputs pass through `SafeMedicalReporting`
- ⚠️ System includes mandatory medical disclaimers
- 🏥 NOT approved for clinical diagnosis
- 📋 FOR RESEARCH PURPOSES ONLY

### **Academic Use**
- 📚 Suitable for research publications
- 🎓 Follows academic best practices
- 📊 Includes statistical validation
- 🔬 Trustworthy AI compliance

### **System Requirements**
- 🐍 **Python**: 3.8+ (3.11 recommended)
- 💾 **Memory**: 4GB+ RAM
- ⏱️ **Runtime**: 30-60 seconds full analysis
- 📦 **Dependencies**: See requirements.txt

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow modular structure
4. Add academic references for new algorithms
5. Test with both Python 3.8 and 3.11
6. Submit pull request

## 📄 License

Academic Research Use - See LICENSE file for details.

## 🎯 Publication Targets

- **Springer Nature Medicine** 
- **Journal of the American Medical Association (JAMA)**
- **New England Journal of Medicine (NEJM)**
- **IEEE Transactions on Biomedical Engineering**
- **Artificial Intelligence in Medicine**

---

**🎓 Ready for academic publication with comprehensive trustworthy AI validation**