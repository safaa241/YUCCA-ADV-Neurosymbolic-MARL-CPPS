"""
LIVRABLE 2 - Entraînement Safe RL pour CPPS

Entraîne les agents MAPPO avec contraintes de sécurité
Compare différentes méthodes Safe RL

Auteur: FEKNI Safaa
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des modules locaux (dans livrable2)
from safe_mappo_agent import SafeMAPPOAgent, SafeCPPSEnvironment
from cbf_shield import ControlBarrierFunction, CBFShield, CBFParameters
from lagrangian_safety import LagrangianSafetyLayer, ConstrainedOptimizer

# Import de l'environnement CPPS DEPUIS livrable1
from cpps_environment import CPPSProductionEnv

def setup_logger(name: str) -> logging.Logger:
    """Configure le logger"""
    os.makedirs("results/logs", exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/logs/{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def train_safe_mappo(method: str = "cbf", 
                     num_episodes: int = 50,
                     cost_limit: float = 10.0) -> Tuple[Dict, Dict]:
    """
    Entraîne MAPPO avec méthode Safe RL
    
    Args:
        method: Méthode Safe RL ('lagrangian', 'cbf', 'adaptive', 'all')
        num_episodes: Nombre d'épisodes
        cost_limit: Limite de coût admissible
        
    Returns:
        (métriques, statistiques_shield)
    """
    logger = setup_logger(f"SAFE_MAPPO_{method.upper()}")
    
    logger.info("="*60)
    logger.info(f"LIVRABLE 2 - Safe RL: {method.upper()}")
    logger.info(f"Cost limit: {cost_limit}")
    logger.info("="*60)
    
    # Configuration
    use_cbf = (method == "cbf")
    use_lagrangian = (method in ["lagrangian", "all"])
    adaptive_penalty = (method in ["adaptive", "all"])
    
    # Environnement
    base_env = CPPSProductionEnv(num_agents=3, episode_length=500)
    env = SafeCPPSEnvironment(base_env, cost_limit=cost_limit)
    
    # Agents Safe RL
    agents = {}
    obs_dim = base_env.observation_spaces[0].shape[0]
    action_dim = base_env.action_spaces[0].n
    
    for i in range(base_env.num_agents):
        agents[i] = SafeMAPPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            cost_limit=cost_limit,
            use_cbf=use_cbf,
            use_lagrangian=use_lagrangian,
            adaptive_penalty=adaptive_penalty
        )
    
    logger.info(f"{base_env.num_agents} agents Safe RL créés")
    
    # Shield CBF (si utilisé)
    cbf_params = CBFParameters(alpha=1.0, beta=0.5)
    cbf = ControlBarrierFunction(cbf_params)
    cbf_shield = CBFShield(cbf) if use_cbf else None
    
    # Métriques
    metrics = {
        'episode_rewards': [],
        'episode_costs': [],
        'episode_violations': [],
        'episode_safety_rates': [],
        'episode_productions': [],
        'lagrangian_lambdas': []
    }
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_violations = 0
        episode_production = 0
        
        for step in range(500):
            actions = {}
            
            for agent_id in range(base_env.num_agents):
                # Sélection action avec CBF si actif
                if use_cbf and cbf_shield:
                    action, log_prob, value = agents[agent_id].select_action(
                        observations[agent_id], explore=True
                    )
                    
                    filtered_action, modified, reason = cbf_shield.filter_action(
                        observations[agent_id], action, agent_id
                    )
                    
                    final_action = filtered_action
                else:
                    final_action, log_prob, value = agents[agent_id].select_action(
                        observations[agent_id], explore=True
                    )
                
                actions[agent_id] = final_action
                
                # Stockage temporaire
                agents[agent_id].store_transition(
                    observations[agent_id],
                    final_action,
                    0,
                    value,
                    log_prob,
                    False,
                    0
                )
            
            # Exécution
            observations, rewards, dones, truncated, info = env.step(actions)
            
            # Mise à jour des transitions
            for agent_id in range(base_env.num_agents):
                reward = rewards[agent_id]
                cost = info['costs'][agent_id] if 'costs' in info else 0
                
                # Récompense modifiée avec contraintes
                if use_lagrangian or adaptive_penalty:
                    optimizer = ConstrainedOptimizer(cost_limit=cost_limit)
                    modified_reward = optimizer.process_transition(reward, cost)
                else:
                    modified_reward = reward
                
                agents[agent_id].memory['rewards'][-1] = modified_reward
                agents[agent_id].memory['costs'][-1] = cost
                agents[agent_id].memory['dones'][-1] = dones[agent_id]
                
                episode_reward += reward
                episode_cost += cost
                episode_violations += len(info[agent_id].get('violations', []))
            
            episode_production = base_env.total_production
        
        # Mise à jour des agents
        total_actor_loss = 0
        total_critic_loss = 0
        total_cost_critic_loss = 0
        
        for agent_id in range(base_env.num_agents):
            next_obs = observations[agent_id]
            _, _, next_value = agents[agent_id].select_action(next_obs, explore=False)
            _, _, next_cost_value = agents[agent_id].select_action(next_obs, explore=False)
            
            returns, advantages = agents[agent_id].compute_returns_and_advantages(next_value.item())
            cost_returns = agents[agent_id].compute_cost_returns(next_cost_value.item())
            
            actor_loss, critic_loss, cost_critic_loss, lag_lambda = agents[agent_id].update(
                returns, advantages, cost_returns
            )
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_cost_critic_loss += cost_critic_loss
        
        # Calcul du taux de sécurité
        total_possible_violations = 500 * base_env.num_agents
        safety_rate = 1 - (episode_violations / total_possible_violations) if episode_violations > 0 else 1.0
        
        # Stockage métriques
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_costs'].append(episode_cost)
        metrics['episode_violations'].append(episode_violations)
        metrics['episode_safety_rates'].append(safety_rate)
        metrics['episode_productions'].append(episode_production)
        metrics['lagrangian_lambdas'].append(agents[0].lagrangian_lambda)
        
        # Logging
        if (episode + 1) % 5 == 0:
            logger.info(f"Episode {episode+1}/{num_episodes}")
            logger.info(f"  Reward: {episode_reward:.2f}")
            logger.info(f"  Cost: {episode_cost:.2f}")
            logger.info(f"  Safety: {safety_rate:.2%}")
            logger.info(f"  Production: {episode_production}")
            logger.info(f"  Violations: {episode_violations}")
            if use_lagrangian:
                logger.info(f"  λ: {agents[0].lagrangian_lambda:.4f}")
    
    # Résultats finaux
    logger.info("\n" + "="*60)
    logger.info(f"RÉSULTATS FINAUX - Safe RL ({method.upper()})")
    logger.info(f"Reward moyen: {np.mean(metrics['episode_rewards']):.2f} ± {np.std(metrics['episode_rewards']):.2f}")
    logger.info(f"Coût moyen: {np.mean(metrics['episode_costs']):.2f}")
    logger.info(f"Taux sécurité moyen: {np.mean(metrics['episode_safety_rates']):.2%}")
    logger.info(f"Production totale: {metrics['episode_productions'][-1] if metrics['episode_productions'] else 0}")
    logger.info(f"Total violations: {sum(metrics['episode_violations'])}")
    logger.info("="*60)
    
    # Statistiques du shield
    shield_stats = {}
    if cbf_shield:
        shield_stats = cbf_shield.get_stats()
        logger.info(f"\nShield CBF: {shield_stats['total_interventions']} interventions")
    
    return metrics, shield_stats


def run_all_methods_comparison(num_episodes: int = 50):
    """Compare toutes les méthodes Safe RL"""
    print("\n" + "="*70)
    print("LIVRABLE 2 - Comparaison des méthodes Safe RL")
    print("="*70)
    
    results = {}
    
    # Baseline
    try:
        with open("results/part1/metrics.json", "r") as f:
            baseline_data = json.load(f)
            baseline_metrics = baseline_data.get('metrics', {})
            results['baseline'] = {
                'mean_reward': np.mean(baseline_metrics.get('total_reward', [-37088])),
                'mean_safety': np.mean(baseline_metrics.get('safety_rate', [0.0085])),
                'total_violations': sum(baseline_metrics.get('total_violations', [74360])),
                'total_production': baseline_metrics.get('total_production', [0])[-1] if baseline_metrics.get('total_production') else 0
            }
    except:
        results['baseline'] = {
            'mean_reward': -37088,
            'mean_safety': 0.0085,
            'total_violations': 74360,
            'total_production': 0
        }
    
    # Safe RL méthodes
    methods = ['lagrangian', 'cbf', 'adaptive']
    
    for method in methods:
        print(f"\n📊 Entraînement Safe RL - {method.upper()}...")
        metrics, shield_stats = train_safe_mappo(method, num_episodes=num_episodes)
        
        results[method] = {
            'mean_reward': np.mean(metrics['episode_rewards']),
            'std_reward': np.std(metrics['episode_rewards']),
            'mean_safety': np.mean(metrics['episode_safety_rates']),
            'std_safety': np.std(metrics['episode_safety_rates']),
            'total_violations': sum(metrics['episode_violations']),
            'total_production': metrics['episode_productions'][-1] if metrics['episode_productions'] else 0,
            'mean_cost': np.mean(metrics['episode_costs']),
            'shield_stats': shield_stats
        }
    
    # Sauvegarde
    output_dir = Path("results/livrable2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "safe_rl_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_episodes": num_episodes,
            "results": results
        }, f, indent=2)
    
    # Affichage
    print("\n" + "="*70)
    print("📊 TABLEAU COMPARATIF - Safe RL")
    print("="*70)
    print(f"\n{'Méthode':<15} {'Sécurité':<12} {'Reward':<12} {'Violations':<12} {'Production':<12}")
    print("-"*70)
    
    for method, res in results.items():
        safety_pct = res['mean_safety'] * 100
        print(f"{method:<15} {safety_pct:>6.2f}%    {res['mean_reward']:>+10.0f}    {res['total_violations']:>10,}    {res['total_production']:>10,}")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_all_methods_comparison(num_episodes=50)
    
    output_dir = Path("results/livrable2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ LIVRABLE 2 COMPLÉTÉ !")
    print(f"📁 Résultats dans {output_dir}/")