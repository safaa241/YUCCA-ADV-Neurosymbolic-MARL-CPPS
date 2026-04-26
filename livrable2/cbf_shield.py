# Control Barrier Functions (fonctions barrière)
# ce qu'il produit: CBFs pour évaluer la sécurité des actions proposées par les agents,
# et un shield qui intercepte les actions dangereuses avant qu'elles ne soient exécutées. 

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CBFParameters:
    """Paramètres pour les Control Barrier Functions"""
    alpha: float = 1.0           # Taux de convergence
    beta: float = 0.5            # Facteur de robustesse
    gamma: float = 0.1           # Marge de sécurité
    safety_margin: float = 0.05  # Marge (5%)


class ControlBarrierFunction:
    """
    Control Barrier Function (CBF) pour garantir la sécurité
    
    La CBF h(x) définit l'ensemble sûr S = {x | h(x) ≥ 0}
    La condition CBF: dh/dt + α h(x) ≥ 0
    """
    
    def __init__(self, params: CBFParameters = None):
        self.params = params or CBFParameters()
        
    def compute_temperature_barrier(self, temp: float) -> float:
        """
        Fonction barrière pour la température
        
        Args:
            temp: Température en °C
            
        Returns:
            Valeur de la barrière (positive = sûr)
        """
        # h(x) = (T_max - T) / T_max
        T_max = 850.0
        h = (T_max - temp) / T_max
        
        # Marge de sécurité
        h = h - self.params.safety_margin
        
        return h
    
    def compute_pressure_barrier(self, pressure: float) -> float:
        """
        Fonction barrière pour la pression
        
        Args:
            pressure: Pression en bar
            
        Returns:
            Valeur de la barrière (positive = sûr)
        """
        P_max = 10.0
        h = (P_max - pressure) / P_max
        h = h - self.params.safety_margin
        return h
    
    def compute_joint_barrier(self, temp: float, pressure: float) -> float:
        """
        Fonction barrière conjointe température + pression
        
        Returns:
            Valeur minimale des barrières individuelles
        """
        h_temp = self.compute_temperature_barrier(temp)
        h_pressure = self.compute_pressure_barrier(pressure)
        return min(h_temp, h_pressure)
    
    def is_safe_state(self, temp: float, pressure: float) -> bool:
        """Vérifie si l'état est sûr"""
        h = self.compute_joint_barrier(temp, pressure)
        return h >= 0
    
    def compute_action_safety(self, temp: float, pressure: float, 
                             action: int, speed: float) -> Tuple[bool, float]:
        """
        Évalue la sécurité d'une action
        
        Args:
            temp: Température actuelle
            pressure: Pression actuelle
            action: Action proposée
            speed: Vitesse actuelle
            
        Returns:
            (action_sûre, dégradation)
        """
        h_current = self.compute_joint_barrier(temp, pressure)
        
        # Prédire le prochain état (simplifié)
        dt = 0.01
        
        if action == 2:  # increase_speed
            delta_speed = 0.5
            new_speed = min(10, speed + delta_speed)
            new_temp = temp + new_speed * dt * 100
            new_pressure = pressure + 0.3 * delta_speed if new_speed > 0 else pressure
        elif action == 0:  # reduce_speed
            delta_speed = -0.5
            new_speed = max(0, speed + delta_speed)
            new_temp = temp + new_speed * dt * 100
            new_pressure = pressure
        elif action == 4:  # emergency_stop
            new_speed = 0
            new_temp = max(20, temp - 1.0)
            new_pressure = max(0, pressure - 0.2)
        else:  # maintain or idle
            new_speed = speed
            new_temp = temp + new_speed * dt * 100
            new_pressure = pressure
        
        # Calculer la nouvelle barrière
        h_next = self.compute_joint_barrier(new_temp, new_pressure)
        
        # Vérifier la condition CBF
        dh_dt = (h_next - h_current) / dt
        cbf_condition = dh_dt + self.params.alpha * h_current
        
        is_safe = cbf_condition >= -self.params.beta
        
        return is_safe, cbf_condition
    
    def get_safe_action_set(self, temp: float, pressure: float, 
                           speed: float, available_actions: List[int]) -> List[int]:
        """Retourne l'ensemble des actions sûres"""
        safe_actions = []
        
        for action in available_actions:
            is_safe, _ = self.compute_action_safety(temp, pressure, action, speed)
            if is_safe:
                safe_actions.append(action)
        
        # Si aucune action n'est sûre, retourner l'action la moins dangereuse
        if not safe_actions:
            # STOP d'urgence par défaut
            return [4]
        
        return safe_actions
    
    def get_safest_action(self, temp: float, pressure: float,
                         speed: float, available_actions: List[int]) -> int:
        """Retourne l'action la plus sûre"""
        best_action = 1  # maintain par défaut
        best_score = -float('inf')
        
        for action in available_actions:
            is_safe, cbf_value = self.compute_action_safety(temp, pressure, action, speed)
            if is_safe and cbf_value > best_score:
                best_score = cbf_value
                best_action = action
        
        return best_action


class CBFShield:
    """
    Shield basé sur Control Barrier Functions
    
    Intercepte les actions dangereuses avant exécution
    """
    
    def __init__(self, cbf: ControlBarrierFunction):
        self.cbf = cbf
        self.blocked_actions = 0
        self.corrected_actions = 0
        self.safety_violations_prevented = 0
        
    def filter_action(self, observation: np.ndarray, 
                     proposed_action: int,
                     agent_id: int) -> Tuple[int, bool, str]:
        """
        Filtre une action avec CBF
        
        Args:
            observation: Observation normalisée
            proposed_action: Action proposée
            agent_id: Identifiant de l'agent
            
        Returns:
            (action_filtrée, a_été_modifiée, raison)
        """
        # Dénormalisation
        temp = observation[0] * 850
        pressure = observation[1] * 10
        speed = observation[2] * 10
        
        # Vérifier si l'état est déjà dangereux
        if not self.cbf.is_safe_state(temp, pressure):
            self.blocked_actions += 1
            return 4, True, "CRITICAL: État dangereux → STOP d'urgence"
        
        # Vérifier la sécurité de l'action proposée
        is_safe, cbf_value = self.cbf.compute_action_safety(temp, pressure, 
                                                            proposed_action, speed)
        
        if is_safe:
            return proposed_action, False, "Action sûre"
        
        # Action dangereuse - trouver une alternative sûre
        available_actions = [0, 1, 2, 3, 4]
        safest_action = self.cbf.get_safest_action(temp, pressure, speed, available_actions)
        
        self.corrected_actions += 1
        self.safety_violations_prevented += 1
        
        explanation = f"CBF: Action {proposed_action} dangereuse → remplacée par {safest_action}"
        
        return safest_action, True, explanation
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du shield CBF"""
        return {
            'blocked_actions': self.blocked_actions,
            'corrected_actions': self.corrected_actions,
            'safety_violations_prevented': self.safety_violations_prevented,
            'total_interventions': self.blocked_actions + self.corrected_actions
        }