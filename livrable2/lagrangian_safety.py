# Multiplicateurs de Lagrange + pénalités adaptatives
# ce qu'il produit : Calcul de λ et pénalités adaptatives pour guider l'apprentissage vers des politiques sûres, en pénalisant les actions qui violent les contraintes de sécurité.

import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class LagrangianSafetyLayer:
    """
    Couche de sécurité basée sur les multiplicateurs Lagrangiens
    
    Formulation CMDP:
        max_π E[Σ γ^t R(s_t, a_t)]
        s.t. E[Σ γ^t C(s_t, a_t)] ≤ d
    
    Lagrangien: L(π, λ) = E[R] - λ (E[C] - d)
    """
    
    def __init__(self, 
                 cost_limit: float = 10.0,
                 lambda_init: float = 0.0,
                 lambda_lr: float = 0.01,
                 lambda_max: float = 100.0):
        
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        
        # Multiplicateur Lagrangien
        self.lambda_value = lambda_init
        
        # Historique des coûts
        self.cost_history = deque(maxlen=100)
        self.constraint_violations = 0
         
    # Calcul Pénalité Lagrangienne = λ * (cost - d) 
    def compute_lagrangian_penalty(self, cost: float) -> float:
       
        return self.lambda_value * max(0, cost - self.cost_limit)
    
    # Mise à jour du multiplicateur Lagrangien λ ← max(0, λ + α_λ (avg_cost - d))
    def update_lagrangian(self, episode_costs: List[float]):
        if not episode_costs:
            return
        
        avg_cost = np.mean(episode_costs)
        
        # Vérification de violation
        if avg_cost > self.cost_limit:
            self.constraint_violations += 1
        
        # Mise à jour du multiplicateur
        delta = avg_cost - self.cost_limit
        self.lambda_value = max(0, self.lambda_value + self.lambda_lr * delta)
        self.lambda_value = min(self.lambda_max, self.lambda_value)
        
        # Historique
        self.cost_history.append(avg_cost)
    
    # Calcul de la récompense modifiée R_mod = R - λ * (C - d) 
    def compute_modified_reward(self, base_reward: float, cost: float) -> float:
        
        penalty = self.compute_lagrangian_penalty(cost)
        modified_reward = base_reward - penalty
        return modified_reward
    
    def get_constraint_satisfaction(self) -> Dict:
        """Retourne l'état de satisfaction des contraintes"""
        avg_cost = np.mean(self.cost_history) if self.cost_history else 0
        
        return {
            'cost_limit': self.cost_limit,
            'average_cost': avg_cost,
            'constraint_satisfied': avg_cost <= self.cost_limit,
            'lagrangian_lambda': self.lambda_value,
            'violation_rate': self.constraint_violations / max(1, len(self.cost_history))
        }


class AdaptivePenalty:
    """
    Pénalité adaptative basée sur l'historique des violations
    
    La pénalité augmente si les violations persistent
    """
    
    def __init__(self, 
                 initial_penalty: float = 1.0,
                 max_penalty: float = 100.0,
                 adaptation_rate: float = 0.1,
                 window_size: int = 20):
        
        self.penalty_factor = initial_penalty
        self.max_penalty = max_penalty
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        self.violation_window = deque(maxlen=window_size)
        
    def compute_penalty(self, cost: float) -> float:
        """
        Calcule la pénalité adaptative
        
        penalty = β * cost * (1 + violation_rate)
        """
        violation_rate = self.get_violation_rate()
        
        # Facteur adaptatif
        adaptive_factor = 1.0 + violation_rate * self.adaptation_rate
        
        # Pénalité
        penalty = self.penalty_factor * cost * adaptive_factor
        
        return min(penalty, self.max_penalty)
    
    def update(self, cost: float):
        """Met à jour l'historique et le facteur de pénalité"""
        is_violation = cost > 0
        self.violation_window.append(1 if is_violation else 0)
        
        # Ajustement du facteur de pénalité
        violation_rate = self.get_violation_rate()
        if violation_rate > 0.5:
            # Beaucoup de violations → augmenter pénalité
            self.penalty_factor = min(self.max_penalty, 
                                     self.penalty_factor * (1 + 0.1))
        elif violation_rate < 0.1 and len(self.violation_window) > self.window_size // 2:
            # Peu de violations → réduire pénalité
            self.penalty_factor = max(1.0, self.penalty_factor * (1 - 0.05))
    
    def get_violation_rate(self) -> float:
        """Retourne le taux de violation récent"""
        if not self.violation_window:
            return 0.0
        return np.mean(self.violation_window)
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de la pénalité adaptative"""
        return {
            'penalty_factor': self.penalty_factor,
            'violation_rate': self.get_violation_rate(),
            'window_size': len(self.violation_window)
        }


class ConstrainedOptimizer:
    """
    Optimiseur sous contraintes pour Safe RL
    
    Combine:
    - Lagrangien dual
    - Pénalités adaptatives
    - Projection sur l'ensemble admissible
    """
    
    def __init__(self, 
                 cost_limit: float = 10.0,
                 lambda_lr: float = 0.01,
                 penalty_initial: float = 1.0):
        
        self.lagrangian = LagrangianSafetyLayer(cost_limit, lambda_lr=lambda_lr)
        self.adaptive_penalty = AdaptivePenalty(initial_penalty=penalty_initial)
        
        self.episode_costs = []
        self.episode_rewards = []
        
    def process_transition(self, reward: float, cost: float) -> float:
        """
        Traite une transition avec contraintes
        
        Args:
            reward: Récompense de base
            cost: Coût de sécurité
            
        Returns:
            Récompense modifiée pour l'apprentissage
        """
        # Mise à jour pénalité adaptative
        self.adaptive_penalty.update(cost)
        
        # Pénalité Lagrangienne
        lagrangian_penalty = self.lagrangian.compute_lagrangian_penalty(cost)
        
        # Pénalité adaptative
        adaptive_penalty = self.adaptive_penalty.compute_penalty(cost)
        
        # Récompense modifiée
        modified_reward = reward - lagrangian_penalty - adaptive_penalty
        
        # Stockage
        self.episode_costs.append(cost)
        self.episode_rewards.append(reward)
        
        return modified_reward
    
    def end_episode(self):
        """Finalise un épisode et met à jour les paramètres"""
        if self.episode_costs:
            # Mise à jour du Lagrangien
            self.lagrangian.update_lagrangian(self.episode_costs)
        
        # Réinitialisation pour l'épisode suivant
        self.episode_costs = []
        self.episode_rewards = []
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de l'optimiseur contraint"""
        return {
            'lagrangian': self.lagrangian.get_constraint_satisfaction(),
            'adaptive_penalty': self.adaptive_penalty.get_stats(),
            'episode_cost': np.sum(self.episode_costs) if self.episode_costs else 0
        }