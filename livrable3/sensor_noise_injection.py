"""
Module d'injection de bruit pour tester la robustesse du shield
À intégrer dans l'entraînement principal
"""

import numpy as np
import random
from typing import Tuple, Optional

class SensorNoiseInjector:
    """
    Simule des défaillances de capteurs pour tester la robustesse
    """
    
    def __init__(self, fault_probability: float = 0.02):
        """
        Args:
            fault_probability: Probabilité d'injecter un défaut (2% par défaut)
        """
        self.fault_probability = fault_probability
        self.fault_types = [
            "nan",           # Valeur manquante
            "out_of_range",  # Valeur hors [0,1]
            "spike",         # Pic soudain
            "slow_drift",    # Dérive lente
            "stuck"          # Valeur bloquée
        ]
        self.fault_counters = {ft: 0 for ft in self.fault_types}
        self.stuck_values = {}
    
    def inject_noise(self, observation: np.ndarray, step: int, agent_id: int) -> Tuple[np.ndarray, dict]:
        """
        Injecte aléatoirement des défauts dans l'observation
        
        Returns:
            (observation_modifiée, rapport_de_faut)
        """
        if random.random() > self.fault_probability:
            return observation, {"fault_injected": False}
        
        fault_type = random.choice(self.fault_types)
        obs_copy = observation.copy()
        
        fault_report = {
            "fault_injected": True,
            "fault_type": fault_type,
            "step": step,
            "agent_id": agent_id,
            "timestamp": None  # Sera rempli plus tard
        }
        
        if fault_type == "nan":
            # Valeur manquante sur un capteur aléatoire
            sensor_idx = random.randint(0, 5)
            obs_copy[sensor_idx] = np.nan
            fault_report["sensor"] = sensor_idx
            fault_report["message"] = f"NaN sur capteur {sensor_idx}"
            
        elif fault_type == "out_of_range":
            # Valeur hors intervalle [0,1]
            sensor_idx = random.randint(0, 5)
            if random.random() < 0.5:
                obs_copy[sensor_idx] = random.uniform(1.5, 3.0)  # > 1
            else:
                obs_copy[sensor_idx] = random.uniform(-2.0, -0.5)  # < 0
            fault_report["sensor"] = sensor_idx
            fault_report["value"] = obs_copy[sensor_idx]
            fault_report["message"] = f"Valeur {obs_copy[sensor_idx]:.2f} hors [0,1]"
            
        elif fault_type == "spike":
            # Pic soudain de température
            # La température est l'index 0
            original_temp = obs_copy[0]
            obs_copy[0] = min(1.2, original_temp + random.uniform(0.3, 0.8))
            fault_report["sensor"] = 0
            fault_report["original"] = original_temp
            fault_report["new"] = obs_copy[0]
            fault_report["message"] = f"Pic de température: {original_temp:.2f} → {obs_copy[0]:.2f}"
            
        elif fault_type == "slow_drift":
            # Dérive lente de la température
            # On utilise un compteur pour accumuler la dérive
            drift_key = f"drift_{agent_id}"
            if drift_key not in self.fault_counters:
                self.fault_counters[drift_key] = 0
            self.fault_counters[drift_key] += 0.01
            
            obs_copy[0] = min(1.0, obs_copy[0] + self.fault_counters[drift_key])
            fault_report["sensor"] = 0
            fault_report["drift_accumulated"] = self.fault_counters[drift_key]
            fault_report["message"] = f"Dérive température: +{self.fault_counters[drift_key]:.2f}"
            
        elif fault_type == "stuck":
            # Valeur bloquée (répète la même valeur)
            stuck_key = f"stuck_{agent_id}"
            if stuck_key not in self.stuck_values:
                self.stuck_values[stuck_key] = observation.mean()
            obs_copy[:] = self.stuck_values[stuck_key]
            fault_report["message"] = f"Capteurs bloqués à {self.stuck_values[stuck_key]:.2f}"
        
        self.fault_counters[fault_type] += 1
        
        return obs_copy, fault_report
    
    def get_statistics(self) -> dict:
        """Retourne les statistiques des défauts injectés"""
        total = sum(self.fault_counters.values())
        return {
            "total_faults": total,
            "by_type": self.fault_counters,
            "injection_rate": self.fault_probability
        }
    
    def reset(self):
        """Réinitialise les compteurs"""
        self.fault_counters = {ft: 0 for ft in self.fault_types}
        self.stuck_values = {}


# Fonction pour tester la montée rapide de température
def create_rapid_temperature_rise(observation: np.ndarray, step: int, 
                                   start_step: int = 100, 
                                   rise_rate: float = 0.02) -> np.ndarray:
    """
    Simule une montée rapide de température (scénario de panne)
    
    Args:
        observation: Observation actuelle
        step: Step actuel
        start_step: Step où commence la montée
        rise_rate: Taux de montée par step (en valeur normalisée)
    """
    obs_copy = observation.copy()
    
    if step >= start_step:
        # Augmentation progressive mais rapide
        increase = min(0.8, (step - start_step) * rise_rate)
        obs_copy[0] = min(1.0, observation[0] + increase)
        
        # La pression suit aussi
        obs_copy[1] = min(1.0, observation[1] + increase * 0.5)
    
    return obs_copy


# Fonction pour créer des conflits entre règles
def create_rule_conflict_scenario(observation: np.ndarray, conflict_type: str = "temp_pressure") -> np.ndarray:
    """
    Crée un état qui déclenche plusieurs règles simultanément
    
    Args:
        conflict_type: 'temp_pressure' (température + pression élevées)
                      'temp_double' (deux seuils température)
    """
    obs_copy = observation.copy()
    
    if conflict_type == "temp_pressure":
        # Température élevée (>800°C) ET pression élevée (>9.0 bar)
        obs_copy[0] = 0.96  # ~816°C
        obs_copy[1] = 0.92  # ~9.2 bar
    elif conflict_type == "temp_double":
        # Température entre deux seuils (déclenche R5 et R3)
        obs_copy[0] = 0.94  # ~799°C (juste en dessous de 800)
    
    return obs_copy