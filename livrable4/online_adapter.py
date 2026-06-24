"""
LIVRABLE 4 - Adaptation en ligne pour réduction Sim-to-Real
Adaptation dynamique des politiques pour le monde réel

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class AdaptationState:
    """État de l'adaptation en ligne"""
    factor_temperature: float = 1.0
    factor_pressure: float = 1.0
    factor_speed: float = 1.0
    bias_temperature: float = 0.0
    bias_pressure: float = 0.0
    bias_speed: float = 0.0
    confidence: float = 1.0
    adaptation_count: int = 0


class OnlineAdapter:
    """
    Adaptateur en ligne pour la réduction Sim-to-Real
    
    Mécanismes:
    - Ajustement dynamique des facteurs de calibration
    - Apprentissage des biais système
    - Détection et correction des dérives
    """
    
    def __init__(self, 
                 adaptation_rate: float = 0.01,
                 forget_factor: float = 0.95,
                 window_size: int = 100,
                 min_confidence: float = 0.5):
        
        self.adaptation_rate = adaptation_rate
        self.forget_factor = forget_factor
        self.window_size = window_size
        self.min_confidence = min_confidence
        
        # État d'adaptation
        self.state = AdaptationState()
        
        # Historique des erreurs
        self.error_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        self.actual_history = deque(maxlen=window_size)
        
        # Métriques
        self.metrics = {
            'mean_errors': [],
            'adaptation_events': [],
            'confidence_evolution': []
        }
        
        # Modèle d'erreur
        self.error_model = {
            'temperature': {'slope': 0.0, 'intercept': 0.0},
            'pressure': {'slope': 0.0, 'intercept': 0.0},
            'speed': {'slope': 0.0, 'intercept': 0.0}
        }
    
    def adapt(self, predicted_state: Dict, actual_state: Dict) -> AdaptationState:
        """
        Adapte les facteurs en fonction de l'erreur observée
        
        Args:
            predicted_state: État prédit par la simulation
            actual_state: État réel mesuré
            
        Returns:
            Nouvel état d'adaptation
        """
        # Calculer les erreurs
        errors = {}
        for key in ['temperature', 'pressure', 'speed']:
            if key in predicted_state and key in actual_state:
                error = actual_state[key] - predicted_state[key]
                errors[key] = error
                
                # Normaliser l'erreur
                if key == 'temperature':
                    normalized_error = error / 850
                elif key == 'pressure':
                    normalized_error = error / 10
                else:
                    normalized_error = error / 10
                
                self.error_history.append(normalized_error)
        
        # Mettre à jour le modèle d'erreur
        self._update_error_model(predicted_state, actual_state)
        
        # Mettre à jour les facteurs d'adaptation
        if len(self.error_history) > 10:
            mean_error = np.mean(self.error_history)
            std_error = np.std(self.error_history)
            
            # Ajuster les facteurs proportionnellement à l'erreur moyenne
            self.state.factor_temperature += self.adaptation_rate * mean_error
            self.state.factor_pressure += self.adaptation_rate * mean_error
            self.state.factor_speed += self.adaptation_rate * mean_error
            
            # Limiter les facteurs
            self.state.factor_temperature = np.clip(self.state.factor_temperature, 0.7, 1.3)
            self.state.factor_pressure = np.clip(self.state.factor_pressure, 0.7, 1.3)
            self.state.factor_speed = np.clip(self.state.factor_speed, 0.7, 1.3)
            
            # Mettre à jour la confiance
            self.state.confidence = 1.0 / (1.0 + std_error)
            self.state.confidence = max(self.min_confidence, min(1.0, self.state.confidence))
            
            # Enregistrer l'adaptation
            self.metrics['adaptation_events'].append({
                'timestamp': datetime.now().isoformat(),
                'mean_error': mean_error,
                'std_error': std_error,
                'new_factors': {
                    'temperature': self.state.factor_temperature,
                    'pressure': self.state.factor_pressure,
                    'speed': self.state.factor_speed
                }
            })
        
        self.state.adaptation_count += 1
        
        # Enregistrer métriques
        if len(self.error_history) > 0:
            self.metrics['mean_errors'].append(np.mean(self.error_history))
            self.metrics['confidence_evolution'].append(self.state.confidence)
        
        return self.state
    
    def _update_error_model(self, predicted: Dict, actual: Dict):
        """Met à jour le modèle d'erreur linéaire"""
        for key in ['temperature', 'pressure', 'speed']:
            if key in predicted and key in actual:
                self.prediction_history.append(predicted[key])
                self.actual_history.append(actual[key])
                
                if len(self.prediction_history) > 10:
                    # Régression linéaire simple
                    preds = np.array(list(self.prediction_history))
                    acts = np.array(list(self.actual_history))
                    
                    slope, intercept = np.polyfit(preds, acts, 1)
                    self.error_model[key] = {'slope': slope, 'intercept': intercept}
    
    def correct_prediction(self, predicted_state: Dict) -> Dict:
        """
        Corrige une prédiction en utilisant le modèle d'adaptation
        
        Args:
            predicted_state: État prédit par la simulation
            
        Returns:
            État corrigé
        """
        corrected = predicted_state.copy()
        
        for key in ['temperature', 'pressure', 'speed']:
            if key in corrected:
                # Appliquer le facteur d'adaptation
                factor = getattr(self.state, f'factor_{key}', 1.0)
                corrected[key] *= factor
                
                # Appliquer le biais
                bias = getattr(self.state, f'bias_{key}', 0.0)
                corrected[key] += bias
        
        return corrected
    
    # Détection d'anomalies basée sur l'historique des erreurs et des prédictions 
    def detect_anomaly(self, predicted: Dict, actual: Dict) -> Tuple[bool, List[Dict]]:
       
        anomalies = []
        
        for key in ['temperature', 'pressure', 'speed']:
            if key in predicted and key in actual:
                error = abs(actual[key] - predicted[key])
                
                # Seuil adaptatif basé sur l'historique
                if len(self.error_history) > 20:
                    threshold = 3 * np.std(self.error_history)
                    
                    if error > threshold:
                        anomalies.append({
                            'key': key,
                            'error': error,
                            'threshold': threshold,
                            'predicted': predicted[key],
                            'actual': actual[key]
                        })
        
        return len(anomalies) > 0, anomalies
    
    def reset(self):
        """Réinitialise l'adaptateur"""
        self.state = AdaptationState()
        self.error_history.clear()
        self.prediction_history.clear()
        self.actual_history.clear()
        self.metrics = {
            'mean_errors': [],
            'adaptation_events': [],
            'confidence_evolution': []
        }
    
    def get_state(self) -> Dict:
        """Retourne l'état courant de l'adaptateur"""
        return {
            'factors': {
                'temperature': self.state.factor_temperature,
                'pressure': self.state.factor_pressure,
                'speed': self.state.factor_speed
            },
            'biases': {
                'temperature': self.state.bias_temperature,
                'pressure': self.state.bias_pressure,
                'speed': self.state.bias_speed
            },
            'confidence': self.state.confidence,
            'adaptation_count': self.state.adaptation_count,
            'recent_mean_error': np.mean(self.error_history) if self.error_history else 0,
            'recent_std_error': np.std(self.error_history) if self.error_history else 0
        }
    
    def get_metrics(self) -> Dict:
        """Retourne les métriques d'adaptation"""
        return {
            'mean_errors': list(self.metrics['mean_errors']),
            'final_confidence': self.state.confidence,
            'total_adaptations': self.state.adaptation_count,
            'error_model': self.error_model
        }


class SimToRealGapAnalyzer:
    """
    Analyseur de l'écart Sim-to-Real
    """
    
    def __init__(self):
        self.gap_history = []
        self.component_gaps = {
            'temperature': [],
            'pressure': [],
            'speed': [],
            'latency': []
        }
    
    def compute_gap(self, sim_state: Dict, real_state: Dict) -> float:
        """
        Calcule l'écart entre simulation et réalité
        
        Returns:
            Écart normalisé (0-1)
        """
        gaps = []
        
        for key in ['temperature', 'pressure', 'speed']:
            if key in sim_state and key in real_state:
                # Écart relatif
                max_val = {'temperature': 850, 'pressure': 10, 'speed': 10}[key]
                gap = abs(real_state[key] - sim_state[key]) / max_val
                gaps.append(gap)
                
                self.component_gaps[key].append(gap)
        
        total_gap = np.mean(gaps) if gaps else 0
        
        self.gap_history.append(total_gap)
        
        return total_gap
    
    def get_gap_trend(self, window: int = 10) -> float:
        """
        Calcule la tendance de l'écart
        
        Returns:
            Tendance (positive = écart croissant)
        """
        if len(self.gap_history) < window:
            return 0
        
        recent = np.mean(self.gap_history[-window:])
        older = np.mean(self.gap_history[-2*window:-window]) if len(self.gap_history) >= 2*window else recent
        
        return recent - older
    
    def get_component_analysis(self) -> Dict:
        """
        Analyse détaillée par composant
        """
        analysis = {}
        
        for component, gaps in self.component_gaps.items():
            if gaps:
                analysis[component] = {
                    'mean_gap': np.mean(gaps),
                    'std_gap': np.std(gaps),
                    'max_gap': np.max(gaps),
                    'trend': self._compute_trend(gaps)
                }
            else:
                analysis[component] = {'mean_gap': 0, 'std_gap': 0, 'max_gap': 0, 'trend': 0}
        
        return analysis
    
    def _compute_trend(self, values: List[float]) -> float:
        """Calcule la tendance linéaire"""
        if len(values) < 10:
            return 0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x[-50:], values[-50:], 1)
        return slope
    
    def get_summary(self) -> Dict:
        """Retourne un résumé de l'analyse"""
        return {
            'total_mean_gap': np.mean(self.gap_history) if self.gap_history else 0,
            'total_std_gap': np.std(self.gap_history) if self.gap_history else 0,
            'component_gaps': self.get_component_analysis(),
            'trend': self.get_gap_trend(),
            'sample_count': len(self.gap_history)
        }


class CalibrationOptimizer:
    """
    Optimiseur de calibration pour réduire l'écart Sim-to-Real
    """
    
    def __init__(self, n_iterations: int = 100, learning_rate: float = 0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.calibration_params = {
            'temp_gain': 1.0,
            'temp_offset': 0.0,
            'pressure_gain': 1.0,
            'pressure_offset': 0.0,
            'speed_gain': 1.0,
            'speed_offset': 0.0
        }
        self.loss_history = []
    
    def optimize(self, sim_data: List[Dict], real_data: List[Dict]) -> Dict:
        """
        Optimise les paramètres de calibration
        
        Args:
            sim_data: Données simulées
            real_data: Données réelles
            
        Returns:
            Paramètres optimisés
        """
        # Conversion en arrays numpy
        sim_temp = np.array([d.get('temperature', 0) for d in sim_data])
        real_temp = np.array([d.get('temperature', 0) for d in real_data])
        sim_pressure = np.array([d.get('pressure', 0) for d in sim_data])
        real_pressure = np.array([d.get('pressure', 0) for d in real_data])
        
        # Optimisation simple par descente de gradient
        for iteration in range(self.n_iterations):
            # Prédictions calibrées
            cal_temp = self.calibration_params['temp_gain'] * sim_temp + self.calibration_params['temp_offset']
            cal_pressure = self.calibration_params['pressure_gain'] * sim_pressure + self.calibration_params['pressure_offset']
            
            # Calcul de la perte
            loss_temp = np.mean((cal_temp - real_temp) ** 2)
            loss_pressure = np.mean((cal_pressure - real_pressure) ** 2)
            total_loss = loss_temp + loss_pressure
            
            self.loss_history.append(total_loss)
            
            # Gradients (simplifiés)
            grad_temp_gain = 2 * np.mean((cal_temp - real_temp) * sim_temp)
            grad_temp_offset = 2 * np.mean(cal_temp - real_temp)
            grad_pressure_gain = 2 * np.mean((cal_pressure - real_pressure) * sim_pressure)
            grad_pressure_offset = 2 * np.mean(cal_pressure - real_pressure)
            
            # Mise à jour
            self.calibration_params['temp_gain'] -= self.learning_rate * grad_temp_gain
            self.calibration_params['temp_offset'] -= self.learning_rate * grad_temp_offset
            self.calibration_params['pressure_gain'] -= self.learning_rate * grad_pressure_gain
            self.calibration_params['pressure_offset'] -= self.learning_rate * grad_pressure_offset
            
            # Contraintes
            self.calibration_params['temp_gain'] = np.clip(self.calibration_params['temp_gain'], 0.8, 1.2)
            self.calibration_params['pressure_gain'] = np.clip(self.calibration_params['pressure_gain'], 0.8, 1.2)
        
        return self.calibration_params
    
    def calibrate(self, state: Dict) -> Dict:
        """Applique la calibration à un état"""
        calibrated = state.copy()
        
        if 'temperature' in calibrated:
            calibrated['temperature'] = (calibrated['temperature'] - self.calibration_params['temp_offset']) / self.calibration_params['temp_gain']
        
        if 'pressure' in calibrated:
            calibrated['pressure'] = (calibrated['pressure'] - self.calibration_params['pressure_offset']) / self.calibration_params['pressure_gain']
        
        return calibrated
    
    def get_params(self) -> Dict:
        """Retourne les paramètres de calibration"""
        return self.calibration_params.copy()