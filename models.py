#!/usr/bin/env python3
"""
AI Models Module for Heart Disease Diagnosis System
==================================================

Contains all machine learning models including:
- Deep Neuro-Fuzzy Systems (DNFS)
- XAI Loss functions
- Continual Learning components
- Concept Drift Detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats

# Simple statistical test implementations
def mcnemar_test_simple(predictions1: np.ndarray, predictions2: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Simple implementation of McNemar's test for paired predictions.
    
    Args:
        predictions1: First model predictions
        predictions2: Second model predictions  
        y_true: True labels
        
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    try:
        # Create contingency table
        # Table format: [[both_correct, model1_correct_model2_wrong],
        #                [model1_wrong_model2_correct, both_wrong]]
        
        correct1 = (predictions1 == y_true)
        correct2 = (predictions2 == y_true)
        
        both_correct = np.sum(correct1 & correct2)
        model1_only = np.sum(correct1 & ~correct2)
        model2_only = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # McNemar test statistic
        # χ² = (|b - c| - 1)² / (b + c) where b and c are off-diagonal elements
        b = model1_only
        c = model2_only
        
        if (b + c) == 0:
            return 0.0, 1.0  # No disagreement between models
        
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        
        # Calculate p-value using chi-square distribution with 1 df
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        return chi2_stat, p_value
        
    except Exception as e:
        return 0.0, 1.0

class ResidualBlock(nn.Module):
    """Residual block for DNFS backbone."""
    
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn(self.linear(x)))
        return out + residual

class DNFSModel(nn.Module):
    """Deep Neuro-Fuzzy System with learnable rules for XAI 2.0.
    
    Implements a hybrid architecture combining:
    - MLP backbone with BatchNorm, Dropout, residual connections
    - Fuzzy membership layer (Gaussian/triangular functions)
    - Learnable fuzzy rules integration
    - Ensemble capability with XGBoost
    
    References:
    - Lin & Lee, Neural-Network-Based Fuzzy Logic Control, 1996
    - Jang, ANFIS: Adaptive-Network-Based Fuzzy Inference System, 1993
    - Wang & Mendel, Generating fuzzy rules by learning from examples, 1992
    """
    
    def __init__(self, input_dim: int = 13, hidden_dims: List[int] = [256, 128, 64], 
                 n_fuzzy_rules: int = 7, fuzzy_type: str = 'gaussian', dropout_rate: float = 0.3):
        super(DNFSModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_fuzzy_rules = n_fuzzy_rules
        self.fuzzy_type = fuzzy_type
        
        # MLP Backbone with residual connections
        self.backbone = self._build_backbone(input_dim, hidden_dims, dropout_rate)
        
        # Fuzzy membership layers
        self.fuzzy_centers = nn.Parameter(torch.randn(n_fuzzy_rules, input_dim))
        self.fuzzy_sigmas = nn.Parameter(torch.ones(n_fuzzy_rules, input_dim) * 0.5)
        
        # Learnable fuzzy rules (TSK-type)
        # Fix: rule_weights should map from input_dim to n_fuzzy_rules, not hidden_dims[-1]
        self.rule_weights = nn.Parameter(torch.randn(n_fuzzy_rules, input_dim))
        self.rule_bias = nn.Parameter(torch.zeros(n_fuzzy_rules))
        
        # Final ensemble layer
        self.ensemble_layer = nn.Linear(hidden_dims[-1] + n_fuzzy_rules, 1)
        
        # XAI components
        self.feature_importance = nn.Parameter(torch.ones(input_dim))
        self.rule_explanations = None
        
        self.init_weights()
    
    def _build_backbone(self, input_dim: int, hidden_dims: List[int], dropout_rate: float) -> nn.Module:
        """Build MLP backbone with BatchNorm, Dropout, and residual connections."""
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Residual connection (if dimensions match)
            if prev_dim == hidden_dim and i > 0:
                layers.append(ResidualBlock(hidden_dim))
            
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def gaussian_membership(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian fuzzy membership functions."""
        # x: (batch_size, input_dim)
        # centers: (n_rules, input_dim)
        # sigmas: (n_rules, input_dim)
        
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        centers_expanded = self.fuzzy_centers.unsqueeze(0)  # (1, n_rules, input_dim)
        sigmas_expanded = self.fuzzy_sigmas.unsqueeze(0)
        
        # Gaussian membership
        diff = x_expanded - centers_expanded
        membership = torch.exp(-0.5 * torch.sum((diff / sigmas_expanded) ** 2, dim=2))
        
        return membership  # (batch_size, n_rules)
    
    def triangular_membership(self, x: torch.Tensor) -> torch.Tensor:
        """Compute triangular fuzzy membership functions."""
        # Simplified triangular implementation
        x_expanded = x.unsqueeze(1)
        centers_expanded = self.fuzzy_centers.unsqueeze(0)
        sigmas_expanded = self.fuzzy_sigmas.unsqueeze(0)
        
        # Triangular membership (simplified)
        diff = torch.abs(x_expanded - centers_expanded)
        membership = torch.clamp(1 - torch.sum(diff / sigmas_expanded, dim=2), 0, 1)
        
        return membership
    
    def compute_fuzzy_rules(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fuzzy rule activations and outputs."""
        # Membership functions
        if self.fuzzy_type == 'gaussian':
            membership = self.gaussian_membership(x)
        else:
            membership = self.triangular_membership(x)
        
        # Normalize membership (sum to 1)
        membership_norm = membership / (torch.sum(membership, dim=1, keepdim=True) + 1e-8)
        
        # TSK-type rule outputs
        x_weighted = torch.mm(x, self.rule_weights.T) + self.rule_bias
        rule_outputs = membership_norm * x_weighted
        
        return membership_norm, rule_outputs
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with explainability components."""
        # MLP backbone
        backbone_output = self.backbone(x)
        
        # Fuzzy rule processing
        membership, rule_outputs = self.compute_fuzzy_rules(x)
        fuzzy_aggregated = torch.sum(rule_outputs, dim=1, keepdim=True)
        
        # Ensemble combination
        combined_features = torch.cat([backbone_output, rule_outputs], dim=1)
        final_output = torch.sigmoid(self.ensemble_layer(combined_features))
        
        # Generate explanations
        rule_importance = torch.mean(membership, dim=0)  # Average rule activation
        feature_gradients = torch.autograd.grad(
            final_output.sum(), x, retain_graph=True, create_graph=True
        )[0] if x.requires_grad else torch.zeros_like(x)
        
        return {
            'prediction': final_output,
            'backbone_output': backbone_output,
            'fuzzy_membership': membership,
            'rule_outputs': rule_outputs,
            'rule_importance': rule_importance,
            'feature_gradients': feature_gradients,
            'explanation': self.generate_explanation(membership, rule_importance)
        }
    
    def generate_explanation(self, membership: torch.Tensor, rule_importance: torch.Tensor) -> Dict[str, Any]:
        """Generate human-readable explanations from fuzzy rules."""
        # Find most activated rules
        top_rules = torch.topk(rule_importance, k=min(3, self.n_fuzzy_rules))
        
        explanations = {
            'top_rules': top_rules.indices.tolist(),
            'rule_activations': top_rules.values.tolist(),
            'rule_descriptions': []
        }
        
        # Generate rule descriptions (simplified)
        for rule_idx in top_rules.indices:
            rule_center = self.fuzzy_centers[rule_idx]
            rule_desc = f"Rule {rule_idx}: IF features ≈ {rule_center.detach().cpu().numpy():.2f}"
            explanations['rule_descriptions'].append(rule_desc)
        
        return explanations
    
    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class XAILoss(nn.Module):
    """Explainability loss for training integration (L_XAI).
    
    L_XAI = α * fidelity_loss + β * stability_loss + γ * simplicity_loss
    
    References:
    - Alvarez-Melis & Jaakkola, Towards Robust Interpretability, NIPS 2018
    - Yeh et al., On the (In)fidelity and Sensitivity of Explanations, NIPS 2019
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4):
        super(XAILoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def fidelity_loss(self, model_output: torch.Tensor, explanation_output: torch.Tensor) -> torch.Tensor:
        """Measure agreement between model and explanations."""
        return F.mse_loss(model_output, explanation_output)
    
    def stability_loss(self, explanations1: torch.Tensor, explanations2: torch.Tensor) -> torch.Tensor:
        """Measure consistency across similar inputs."""
        return F.cosine_embedding_loss(explanations1, explanations2, 
                                     torch.ones(explanations1.size(0)).to(explanations1.device))
    
    def simplicity_loss(self, feature_importance: torch.Tensor) -> torch.Tensor:
        """Encourage sparsity in feature importance."""
        return torch.norm(feature_importance, p=1)
    
    def forward(self, model_outputs: Dict[str, torch.Tensor], 
                perturbed_outputs: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Compute total XAI loss."""
        # Fidelity: agreement between prediction and rule-based explanation
        prediction = model_outputs['prediction']
        rule_prediction = torch.sum(model_outputs['rule_outputs'], dim=1, keepdim=True)
        fidelity = self.fidelity_loss(prediction, torch.sigmoid(rule_prediction))
        
        # Stability: consistency of explanations (if perturbed inputs provided)
        stability = torch.tensor(0.0, device=prediction.device)
        if perturbed_outputs is not None:
            original_gradients = model_outputs['feature_gradients']
            perturbed_gradients = perturbed_outputs['feature_gradients']
            stability = self.stability_loss(original_gradients, perturbed_gradients)
        
        # Simplicity: sparsity of feature importance
        feature_grads = model_outputs['feature_gradients']
        simplicity = self.simplicity_loss(torch.mean(torch.abs(feature_grads), dim=0))
        
        total_loss = self.alpha * fidelity + self.beta * stability + self.gamma * simplicity
        
        return total_loss

class ConceptDriftDetector:
    """Statistical concept drift detection.
    
    References:
    - Gama et al., Learning with Drift Detection, SBIA 2004
    - Page, Continuous Inspection Schemes, Biometrika 1954
    """
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_window = []
        self.current_window = []
        
    def add_sample(self, prediction: float, actual: float) -> bool:
        """Add a new sample and detect drift."""
        error = abs(prediction - actual)
        
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(error)
            return False
        
        self.current_window.append(error)
        
        if len(self.current_window) >= self.window_size:
            # Perform statistical test (Kolmogorov-Smirnov)
            drift_detected = self._detect_drift()
            
            if drift_detected:
                # Update reference window
                self.reference_window = self.current_window.copy()
                self.current_window = []
                return True
            
            # Sliding window
            self.current_window.pop(0)
        
        return False
    
    def _detect_drift(self) -> bool:
        """Statistical test for concept drift."""
        if len(self.current_window) < self.window_size:
            return False
        
        # Kolmogorov-Smirnov test
        from scipy.stats import ks_2samp
        statistic, p_value = ks_2samp(self.reference_window, self.current_window)
        
        return p_value < self.drift_threshold

class ContinualLearner:
    """Continual learning with A-GEM and EWC for concept drift adaptation.
    
    References:
    - Kirkpatrick et al., Overcoming catastrophic forgetting, PNAS 2017
    - Chaudhry et al., Efficient Lifelong Learning with A-GEM, ICLR 2019
    - Lopez-Paz & Ranzato, Gradient Episodic Memory, NIPS 2017
    """
    
    def __init__(self, model: nn.Module, memory_size: int = 1000, ewc_lambda: float = 0.4):
        self.model = model
        self.memory_size = memory_size
        self.ewc_lambda = ewc_lambda
        
        # Episodic memory for A-GEM
        self.episodic_memory = {
            'inputs': [],
            'targets': [],
            'task_ids': []
        }
        
        # Fisher Information for EWC
        self.fisher_information = {}
        self.optimal_params = {}
        
        # Concept drift detection
        self.drift_detector = ConceptDriftDetector()
        
    def add_to_memory(self, X: torch.Tensor, y: torch.Tensor, task_id: int):
        """Add samples to episodic memory."""
        batch_size = X.size(0)
        
        if len(self.episodic_memory['inputs']) < self.memory_size:
            self.episodic_memory['inputs'].append(X)
            self.episodic_memory['targets'].append(y)
            self.episodic_memory['task_ids'].extend([task_id] * batch_size)
        else:
            # Random replacement
            indices = np.random.choice(len(self.episodic_memory['inputs']), 
                                     size=batch_size, replace=False)
            for i, idx in enumerate(indices):
                self.episodic_memory['inputs'][idx] = X[i:i+1]
                self.episodic_memory['targets'][idx] = y[i:i+1]
                self.episodic_memory['task_ids'][idx] = task_id
    
    def compute_fisher_information(self, dataloader: DataLoader):
        """Compute Fisher Information Matrix for EWC."""
        self.model.eval()
        fisher = {}
        
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        for X, y in dataloader:
            self.model.zero_grad()
            output = self.model(X)['prediction']
            loss = F.binary_cross_entropy(output, y.float().unsqueeze(1))
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize by dataset size
        dataset_size = len(dataloader.dataset)
        for name in fisher:
            fisher[name] /= dataset_size
        
        self.fisher_information = fisher
        self.optimal_params = {name: param.clone() for name, param in self.model.named_parameters()}
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        ewc_loss = torch.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                ewc_loss += (self.fisher_information[name] * 
                           (param - self.optimal_params[name]) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def agem_gradient_correction(self, current_grads: Dict, memory_grads: Dict) -> Dict:
        """Apply A-GEM gradient correction."""
        # Compute dot product of current and memory gradients
        dot_product = sum(torch.sum(current_grads[name] * memory_grads[name]) 
                         for name in current_grads)
        
        if dot_product < 0:  # Conflicting gradients
            # Project current gradients onto memory gradients
            memory_norm_sq = sum(torch.sum(memory_grads[name] ** 2) for name in memory_grads)
            projection_coeff = dot_product / (memory_norm_sq + 1e-8)
            
            corrected_grads = {}
            for name in current_grads:
                corrected_grads[name] = (current_grads[name] - 
                                       projection_coeff * memory_grads[name])
        else:
            corrected_grads = current_grads
        
        return corrected_grads