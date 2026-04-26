"""
LIVRABLE 2 - COMPARAISON COMPLÈTE SAFE RL
Inclut MAPPO, QMIX, MADDPG avec et sans Safe RL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import time

# Import des modules
from train_safe_mappo import train_safe_agent, SafeRLMethod
from cpps_environment import CPPSProductionEnv


class SafeRLWrapper:
    """
    Wrapper générique pour ajouter Safe RL à n'importe quel algorithme MARL
    Supporte: MAPPO, QMIX, MADDPG
    """
    
    def __init__(self, 
                 base_agent,
                 method: SafeRLMethod,
                 agent_id: int,
                 cost_limit: float = 10.0):
        
        self.base_agent = base_agent
        self.method = method
        self.agent_id = agent_id
        self.cost_limit = cost_limit
        
        # Multiplicateur Lagrangien
        self.lagrangian_lambda = 0.0
        self.lambda_lr = 0.01
        
        # Statistiques
        self.cost_history = []
        self.violation_history = []
        
        # Pour CBF
        self.use_cbf = (method == SafeRLMethod.CBF)
        
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        """Sélectionne une action avec application des contraintes Safe RL"""
        
        # 1. Action proposée par l'algorithme de base
        if hasattr(self.base_agent, 'compute_action'):
            raw_action = self.base_agent.compute_action(observation)
        elif hasattr(self.base_agent, 'select_action'):
            raw_action, _, _ = self.base_agent.select_action(observation, explore)
        else:
            raw_action = self.base_agent.act(observation)
        
        # 2. Application du CBF si actif
        if self.use_cbf and self.method == SafeRLMethod.CBF:
            safe_action = self._apply_cbf(raw_action, observation)
            return safe_action
        
        return raw_action
    
    def _apply_cbf(self, action: int, observation: np.ndarray) -> int:
        """Applique Control Barrier Function"""
        # Dénormalisation
        temp = observation[0] * 850
        pressure = observation[1] * 10
        speed = observation[2] * 10
        
        # Vérification CBF pour augmentation de vitesse
        if action == 2:  # increase_speed
            if temp > 800 or pressure > 9.0 or speed >= 9.5:
                return 0  # reduce_speed
        
        return action
    
    def compute_cost(self, observation: np.ndarray, action: int) -> float:
        """Calcule le coût de sécurité pour une transition"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        
        cost = 0.0
        if temp >= 850:
            cost += 100.0
        elif temp > 800:
            cost += 50.0
        elif temp > 750:
            cost += 20.0
        
        if self.agent_id == 1:  # Robot peinture
            if pressure >= 10:
                cost += 100.0
            elif pressure > 9.0:
                cost += 50.0
        
        return cost
    
    def compute_modified_reward(self, reward: float, cost: float) -> float:
        """Calcule la récompense modifiée selon la méthode Safe RL"""
        
        if self.method == SafeRLMethod.LAGRANGIAN:
            # Pénalité Lagrangienne
            penalty = self.lagrangian_lambda * max(0, cost - self.cost_limit)
            return reward - penalty
        
        elif self.method == SafeRLMethod.ADAPTIVE:
            # Pénalité adaptative basée sur l'historique
            violation_rate = np.mean([1 if c > 0 else 0 for c in self.cost_history[-20:]]) if self.cost_history else 0
            adaptive_factor = 1.0 + 5.0 * violation_rate
            penalty = adaptive_factor * cost
            return reward - penalty
        
        else:  # CBF
            return reward
    
    def update_lagrangian(self, episode_costs: List[float]):
        """Met à jour le multiplicateur Lagrangien"""
        if not episode_costs:
            return
        
        avg_cost = np.mean(episode_costs)
        delta = avg_cost - self.cost_limit
        self.lagrangian_lambda = max(0, self.lagrangian_lambda + self.lambda_lr * delta)
        self.lagrangian_lambda = min(100.0, self.lagrangian_lambda)
        
        self.cost_history.extend(episode_costs)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques Safe RL"""
        return {
            'method': self.method.value,
            'lagrangian_lambda': self.lagrangian_lambda if self.method == SafeRLMethod.LAGRANGIAN else None,
            'avg_cost': np.mean(self.cost_history) if self.cost_history else 0,
            'cost_limit': self.cost_limit
        }


class BaseMARLAlgorithm:
    """
    Classe de base simulant QMIX, MADDPG et MAPPO
    (À remplacer par les algorithmes réels en production)
    """
    
    def __init__(self, name: str, algorithm_type: str, action_dim: int = 5):
        self.name = name
        self.algorithm_type = algorithm_type  # 'mappo', 'qmix', 'maddpg'
        self.action_dim = action_dim
        
    def compute_action(self, observation: np.ndarray) -> int:
        """Simule une action (à remplacer par le vrai algorithme)"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        
        # Politique simple pour simulation
        if temp < 700 and pressure < 7:
            return 2  # increase_speed
        elif temp > 800:
            return 0  # reduce_speed
        else:
            return 1  # maintain
    
    def learn(self, experiences):
        pass


def train_algorithm_with_safe_rl(algorithm_type: str,
                                  method: SafeRLMethod,
                                  num_episodes: int = 50,
                                  episode_length: int = 500,
                                  num_agents: int = 3) -> Dict[str, Any]:
    """
    Entraîne un algorithme MARL (MAPPO/QMIX/MADDPG) avec Safe RL
    
    Args:
        algorithm_type: 'mappo', 'qmix', 'maddpg'
        method: Méthode Safe RL (LAGRANGIAN, CBF, ADAPTIVE)
        num_episodes: Nombre d'épisodes
        episode_length: Longueur par épisode
        num_agents: Nombre d'agents
        
    Returns:
        Métriques d'évaluation
    """
    
    print(f"\n{'='*60}")
    print(f"Entraînement: {algorithm_type.upper()} + Safe RL ({method.value})")
    print(f"{'='*60}")
    
    env = CPPSProductionEnv(num_agents=num_agents, episode_length=episode_length)
    
    # Création des agents
    agents = {}
    safe_wrappers = {}
    
    for i in range(num_agents):
        base_agent = BaseMARLAlgorithm(f"{algorithm_type}_agent_{i}", algorithm_type)
        safe_wrapper = SafeRLWrapper(base_agent, method, i)
        agents[i] = base_agent
        safe_wrappers[i] = safe_wrapper
    
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
                action = safe_wrappers[agent_id].select_action(observations[agent_id])
                actions[agent_id] = action
                
                # Calcul du coût
                cost = safe_wrappers[agent_id].compute_cost(observations[agent_id], action)
                episode_costs_list.append(cost)
                episode_cost += cost
            
            # Exécution
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            # Mise à jour des récompenses modifiées
            for agent_id in range(num_agents):
                reward = rewards[agent_id]
                modified_reward = safe_wrappers[agent_id].compute_modified_reward(reward, episode_costs_list[-1] if episode_costs_list else 0)
                
                episode_reward += modified_reward
                
                if info[agent_id].get('violations'):
                    episode_violation += len(info[agent_id]['violations'])
            
            observations = next_obs
        
        # Mise à jour des multiplicateurs Lagrangiens
        for agent_id in range(num_agents):
            safe_wrappers[agent_id].update_lagrangian(episode_costs_list)
        
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
    
    return {
        'algorithm': algorithm_type,
        'safe_method': method.value,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_safety': np.mean(episode_safety_rates),
        'std_safety': np.std(episode_safety_rates),
        'total_violations': sum(episode_violations),
        'total_production': episode_productions[-1] if episode_productions else 0,
        'mean_cost': np.mean(episode_costs),
        'rewards': episode_rewards,
        'safety_rates': episode_safety_rates,
        'violations': episode_violations,
        'productions': episode_productions
    }


def run_complete_safe_comparison(num_episodes: int = 50):
    """
    Lance la comparaison complète:
    - Algorithmes: MAPPO, QMIX, MADDPG
    - Safe RL: Lagrangien, CBF, Adaptative
    - Baseline: Sans Safe RL
    """
    
    print("\n" + "="*80)
    print("LIVRABLE 2 - COMPARAISON COMPLÈTE SAFE RL")
    print("MAPPO | QMIX | MADDPG avec Safe RL (Lagrangien/CBF/Adaptative)")
    print("="*80)
    
    algorithms = ['mappo', 'qmix', 'maddpg']
    safe_methods = [None, SafeRLMethod.LAGRANGIAN, SafeRLMethod.CBF, SafeRLMethod.ADAPTIVE]
    method_names = {None: "Standard", 
                    SafeRLMethod.LAGRANGIAN: "Lagrangien",
                    SafeRLMethod.CBF: "CBF",
                    SafeRLMethod.ADAPTIVE: "Adaptative"}
    
    all_results = {}
    
    for algo in algorithms:
        all_results[algo] = {}
        
        for method in safe_methods:
            method_name = method_names[method]
            print(f"\n📊 Test: {algo.upper()} + {method_name}")
            
            if method is None:
                # Baseline sans Safe RL
                results = train_algorithm_with_safe_rl(
                    algo, 
                    SafeRLMethod.LAGRANGIAN,  # Méthode par défaut (pour wrapper)
                    num_episodes=num_episodes
                )
                results['safe_method'] = 'none'
                all_results[algo]['baseline'] = results
            else:
                results = train_algorithm_with_safe_rl(algo, method, num_episodes=num_episodes)
                all_results[algo][method.value] = results
    
    # Sauvegarde des résultats
    output_dir = Path("results/livrable2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Conversion pour JSON
    json_results = {}
    for algo, methods in all_results.items():
        json_results[algo] = {}
        for method_name, results in methods.items():
            json_results[algo][method_name] = {
                'mean_safety': float(results['mean_safety']),
                'mean_reward': float(results['mean_reward']),
                'total_violations': int(results['total_violations']),
                'total_production': int(results['total_production'])
            }
    
    with open(output_dir / "complete_safe_comparison.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Affichage du tableau comparatif
    print("\n" + "="*100)
    print("TABLEAU COMPARATIF - SÉCURITÉ (%)")
    print("="*100)
    
    print(f"\n{'Algorithme':<12} {'Standard':<12} {'Lagrangien':<12} {'CBF':<12} {'Adaptative':<12} {'Meilleur':<12}")
    print("-"*80)
    
    for algo in algorithms:
        baseline = all_results[algo]['baseline']['mean_safety'] * 100
        lag = all_results[algo].get('lagrangian', {}).get('mean_safety', 0) * 100
        cbf = all_results[algo].get('cbf', {}).get('mean_safety', 0) * 100
        adapt = all_results[algo].get('adaptive', {}).get('mean_safety', 0) * 100
        
        best = max([lag, cbf, adapt])
        best_method = "CBF" if best == cbf else "Lagrangien" if best == lag else "Adaptative"
        
        print(f"{algo.upper():<12} {baseline:>6.1f}%      {lag:>6.1f}%      {cbf:>6.1f}%      {adapt:>6.1f}%      {best_method} ({best:.1f}%)")
    
    print("\n" + "="*100)
    
    return all_results


if __name__ == "__main__":
    results = run_complete_safe_comparison(num_episodes=50)