"""
LIVRABLE 2 - ENTRAÎNEMENT SAFE RL
Support pour MAPPO, QMIX, MADDPG avec méthodes Safe RL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Import des modules
from cpps_environment import CPPSProductionEnv


class SafeRLMethod(Enum):
    """Méthodes Safe RL disponibles"""
    LAGRANGIAN = "lagrangian"
    CBF = "cbf"
    ADAPTIVE = "adaptive"


class SafeRLAgent:
    """
    Agent générique avec Safe RL pour MAPPO/QMIX/MADDPG
    """
    
    def __init__(self,
                 agent_id: int,
                 method: SafeRLMethod,
                 obs_dim: int = 6,
                 action_dim: int = 5,
                 cost_limit: float = 10.0,
                 learning_rate: float = 3e-4):
        
        self.agent_id = agent_id
        self.method = method
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cost_limit = cost_limit
        
        # Pour méthode Lagrangienne
        self.lagrangian_lambda = 0.0
        self.lambda_lr = 0.01
        
        # Pour méthode Adaptative
        self.penalty_factor = 1.0
        self.violation_window = []
        self.window_size = 20
        
        # Historique
        self.cost_history = []
        
        # Réseaux simplifiés (à remplacer par MAPPO/QMIX/MADDPG réels)
        self._init_networks()
    
    def _init_networks(self):
        """Initialisation des réseaux (simplifiée pour démo)"""
        import torch
        import torch.nn as nn
        
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        """Sélectionne une action avec contraintes Safe RL"""
        
        # Action proposée par le réseau (simulée)
        raw_action = self._propose_action(observation)
        
        # Application CBF si nécessaire
        if self.method == SafeRLMethod.CBF:
            safe_action = self._apply_cbf(raw_action, observation)
            return safe_action
        
        return raw_action
    
    def _propose_action(self, observation: np.ndarray) -> int:
        """Action proposée par le réseau neuronal (simulation)"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        
        # Politique simple basée sur l'état
        if temp < 700 and pressure < 7:
            return 2  # increase_speed
        elif temp > 800:
            return 0  # reduce_speed
        else:
            return 1  # maintain
    
    def _apply_cbf(self, action: int, observation: np.ndarray) -> int:
        """Applique Control Barrier Function"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        speed = observation[2] * 10
        
        # Vérification des conditions de sécurité
        if action == 2:  # increase_speed
            if temp > 800 or pressure > 9.0 or speed >= 9.5:
                return 0  # reduce_speed
        elif action == 4:  # emergency_stop
            if temp < 800 and not (pressure > 9.0):
                # STOP non nécessaire, remplacer par maintain
                return 1
        
        return action
    
    def compute_cost(self, observation: np.ndarray, action: int) -> float:
        """Calcule le coût de sécurité pour une transition"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        
        cost = 0.0
        
        # Coût température
        if temp >= 850:
            cost += 100.0
        elif temp > 800:
            cost += 50.0
        elif temp > 750:
            cost += 20.0
        
        # Coût pression (spécifique agent peinture)
        if self.agent_id == 1:
            if pressure >= 10:
                cost += 100.0
            elif pressure > 9.0:
                cost += 50.0
            elif pressure > 8.5:
                cost += 20.0
        
        return cost
    
    def compute_modified_reward(self, reward: float, cost: float) -> float:
        """Calcule la récompense modifiée selon la méthode Safe RL"""
        
        if self.method == SafeRLMethod.LAGRANGIAN:
            # Pénalité Lagrangienne
            penalty = self.lagrangian_lambda * max(0, cost - self.cost_limit)
            return reward - penalty
        
        elif self.method == SafeRLMethod.ADAPTIVE:
            # Pénalité adaptative
            violation_rate = np.mean(self.violation_window) if self.violation_window else 0
            adaptive_factor = 1.0 + 5.0 * violation_rate
            penalty = self.penalty_factor * cost * adaptive_factor
            return reward - min(penalty, 100.0)
        
        else:
            # CBF - pas de modification de récompense
            return reward
    
    def update(self, cost: float):
        """Met à jour les paramètres Safe RL après chaque épisode"""
        self.cost_history.append(cost)
        
        # Mise à jour du multiplicateur Lagrangien
        if self.method == SafeRLMethod.LAGRANGIAN:
            avg_cost = np.mean(self.cost_history[-50:]) if self.cost_history else 0
            delta = avg_cost - self.cost_limit
            self.lagrangian_lambda = max(0, self.lagrangian_lambda + self.lambda_lr * delta)
            self.lagrangian_lambda = min(100.0, self.lagrangian_lambda)
        
        # Mise à jour de la pénalité adaptative
        elif self.method == SafeRLMethod.ADAPTIVE:
            is_violation = 1 if cost > self.cost_limit else 0
            self.violation_window.append(is_violation)
            if len(self.violation_window) > self.window_size:
                self.violation_window.pop(0)
            
            violation_rate = np.mean(self.violation_window)
            if violation_rate > 0.5:
                self.penalty_factor = min(10.0, self.penalty_factor * 1.1)
            elif violation_rate < 0.1 and len(self.violation_window) > 10:
                self.penalty_factor = max(1.0, self.penalty_factor * 0.95)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques de l'agent"""
        return {
            'method': self.method.value,
            'lagrangian_lambda': self.lagrangian_lambda if self.method == SafeRLMethod.LAGRANGIAN else None,
            'penalty_factor': self.penalty_factor if self.method == SafeRLMethod.ADAPTIVE else None,
            'avg_cost': np.mean(self.cost_history) if self.cost_history else 0,
            'cost_limit': self.cost_limit
        }


def train_safe_agent(algorithm_name: str,
                     method: SafeRLMethod,
                     num_episodes: int = 50,
                     episode_length: int = 500,
                     num_agents: int = 3,
                     cost_limit: float = 10.0) -> Dict:
    """
    Entraîne un agent MARL avec Safe RL
    
    Args:
        algorithm_name: 'mappo', 'qmix', 'maddpg'
        method: Méthode Safe RL
        num_episodes: Nombre d'épisodes
        episode_length: Longueur par épisode
        num_agents: Nombre d'agents
        cost_limit: Limite de coût
        
    Returns:
        Métriques d'entraînement
    """
    
    print(f"\n{'='*60}")
    print(f"Entraînement: {algorithm_name.upper()} + Safe RL ({method.value})")
    print(f"Cost limit: {cost_limit}")
    print(f"{'='*60}")
    
    env = CPPSProductionEnv(num_agents=num_agents, episode_length=episode_length)
    
    # Création des agents
    agents = {}
    for i in range(num_agents):
        agents[i] = SafeRLAgent(
            agent_id=i,
            method=method,
            cost_limit=cost_limit
        )
    
    # Métriques
    episode_rewards = []
    episode_violations = []
    episode_safety_rates = []
    episode_productions = []
    episode_costs = []
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        episode_violation = 0
        episode_cost = 0
        episode_costs_list = []
        
        for step in range(episode_length):
            actions = {}
            
            for agent_id in range(num_agents):
                # Sélection action avec Safe RL
                action = agents[agent_id].select_action(observations[agent_id])
                actions[agent_id] = action
                
                # Calcul du coût
                cost = agents[agent_id].compute_cost(observations[agent_id], action)
                episode_costs_list.append(cost)
                episode_cost += cost
            
            # Exécution
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            # Mise à jour des récompenses modifiées
            for agent_id in range(num_agents):
                reward = rewards[agent_id]
                modified_reward = agents[agent_id].compute_modified_reward(reward, episode_costs_list[-1] if episode_costs_list else 0)
                episode_reward += modified_reward
                
                if info[agent_id].get('violations'):
                    episode_violation += len(info[agent_id]['violations'])
            
            observations = next_obs
        
        # Mise à jour des agents
        for agent_id in range(num_agents):
            agents[agent_id].update(episode_cost)
        
        # Calcul du taux de sécurité
        total_steps = episode_length * num_agents
        safety_rate = 1 - (episode_violation / total_steps) if episode_violation > 0 else 1.0
        
        # Enregistrement
        episode_rewards.append(episode_reward)
        episode_violations.append(episode_violation)
        episode_safety_rates.append(safety_rate)
        episode_productions.append(env.total_production)
        episode_costs.append(episode_cost)
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{num_episodes}: Safety={safety_rate:.2%}, Reward={episode_reward:.0f}, Prod={env.total_production}")
    
    # Résultats finaux
    print(f"\nRésultats {algorithm_name.upper()} + {method.value}:")
    print(f"  Sécurité moyenne: {np.mean(episode_safety_rates):.2%}")
    print(f"  Production totale: {episode_productions[-1]}")
    print(f"  Violations totales: {sum(episode_violations)}")
    
    return {
        'algorithm': algorithm_name,
        'safe_method': method.value,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_safety': float(np.mean(episode_safety_rates)),
        'std_safety': float(np.std(episode_safety_rates)),
        'total_violations': int(sum(episode_violations)),
        'total_production': int(episode_productions[-1]) if episode_productions else 0,
        'mean_cost': float(np.mean(episode_costs)),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_safety_rates': [float(s) for s in episode_safety_rates],
        'episode_violations': [int(v) for v in episode_violations],
        'episode_productions': [int(p) for p in episode_productions]
    }


if __name__ == "__main__":
    # Test rapide
    results = train_safe_agent("mappo", SafeRLMethod.CBF, num_episodes=10)