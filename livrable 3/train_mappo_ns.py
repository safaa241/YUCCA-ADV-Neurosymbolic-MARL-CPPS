"""
LIVRABLE 3 - Entraînement MAPPO-NS (Neurosymbolique)
Entraîne le système multi-agents avec shield neurosymbolique

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
from typing import Dict, List, Tuple, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des modules
from mappo_ns_agent import MultiAgentMAPPO_NS
from cpps_environment import CPPSProductionEnv


def setup_logger(name: str) -> logging.Logger:
    """Configure le logger"""
    os.makedirs("results/logs", exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/logs/{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Entraînement du système MAPPO-NS avec ou sans shield et collecte des métriques détaillées pour analyse approfondie et comparaison avec la baseline MAPPO standard. 
def train_mappo_ns(num_episodes: int = 50,
                   episode_length: int = 500,
                   use_shield: bool = True,
                   log_interval: int = 5,
                   save_interval: int = 25) -> Tuple[MultiAgentMAPPO_NS, Dict]:
    
    algo_name = "MAPPO-NS" if use_shield else "MAPPO-Standard"
    logger = setup_logger(algo_name)
    
    logger.info("="*70)
    logger.info(f"LIVRABLE 3 - Entraînement {algo_name}")
    logger.info(f"Shield neurosymbolique: {'ACTIF' if use_shield else 'INACTIF'}")
    logger.info("="*70)
    
    # Environnement
    env = CPPSProductionEnv(num_agents=3, episode_length=episode_length)
    logger.info(f"Environnement créé: {env.num_agents} agents, {episode_length} steps/épisode")
    
    # Dimensions
    obs_dim = env.observation_spaces[0].shape[0]
    action_dim = env.action_spaces[0].n
    
    logger.info(f"Dimensions: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Système multi-agents
    system = MultiAgentMAPPO_NS(
        num_agents=env.num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_shield=use_shield,
        learning_rate=3e-4
    )
    
    logger.info(f"Système {system.name} initialisé")
    
    # Métriques
    metrics = {
        'episode_rewards': [],
        'episode_violations': [],
        'episode_safety_rates': [],
        'episode_productions': [],
        'episode_corrections': [],
        'actor_losses': [],
        'critic_losses': []
    }
    
    # Boucle d'entraînement
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = 0
        episode_violations = 0
        episode_corrections = 0
        
        # Stockage pour l'épisode
        episode_values = {i: [] for i in range(env.num_agents)}
        episode_log_probs = {i: [] for i in range(env.num_agents)}
        
        for step in range(episode_length):
            # Sélection des actions
            actions = {}
            values = {}
            log_probs = {}
            
            for agent_id, obs in observations.items():
                action, log_prob, value = system.agents[agent_id].select_action(obs, explore=True)
                actions[agent_id] = action
                values[agent_id] = value
                log_probs[agent_id] = log_prob
                
                # Compter les corrections si shield actif
                if use_shield and system.agents[agent_id].shield:
                    # Vérifier si l'action a été modifiée
                    # (simplifié - dans la réalité, on vérifierait le flag)
                    pass
            
            # Exécution dans l'environnement
            next_obs, rewards, dones, truncated, info = env.step(actions)
            
            # Stockage des transitions
            for agent_id in range(env.num_agents):
                system.agents[agent_id].store_transition(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    values[agent_id],
                    log_probs[agent_id],
                    dones[agent_id]
                )
                
                episode_reward += rewards[agent_id]
                episode_violations += len(info[agent_id].get('violations', []))
            
            observations = next_obs
        
        # Mise à jour des agents
        next_observations = observations
        actor_loss, critic_loss = system.update_all(next_observations)
        
        # Statistiques du shield
        if use_shield:
            shield_stats = system.get_shield_stats()
            episode_corrections = shield_stats.get('corrected_actions', 0)
        
        # Calcul du taux de sécurité
        total_steps = episode_length * env.num_agents
        safety_rate = 1 - (episode_violations / total_steps) if episode_violations > 0 else 1.0
        
        # Enregistrement des métriques
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_violations'].append(episode_violations)
        metrics['episode_safety_rates'].append(safety_rate)
        metrics['episode_productions'].append(env.total_production)
        metrics['episode_corrections'].append(episode_corrections)
        metrics['actor_losses'].append(actor_loss)
        metrics['critic_losses'].append(critic_loss)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            logger.info(f"Episode {episode+1}/{num_episodes}")
            logger.info(f"  Reward: {episode_reward:.2f}")
            logger.info(f"  Safety: {safety_rate:.2%}")
            logger.info(f"  Violations: {episode_violations}")
            logger.info(f"  Production: {env.total_production}")
            if use_shield:
                logger.info(f"  Shield corrections: {episode_corrections}")
            logger.info(f"  Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")
            logger.info("-" * 50)
        
        # Sauvegarde périodique
        if (episode + 1) % save_interval == 0:
            save_path = Path(f"results/checkpoints/{algo_name}_ep{episode+1}.pt")
            system.agents[0].save(str(save_path))
    
    # Résultats finaux
    logger.info("\n" + "="*70)
    logger.info(f"RÉSULTATS FINAUX - {algo_name}")
    logger.info("="*70)
    logger.info(f"Reward moyen: {np.mean(metrics['episode_rewards']):.2f} ± {np.std(metrics['episode_rewards']):.2f}")
    logger.info(f"Taux sécurité moyen: {np.mean(metrics['episode_safety_rates']):.2%}")
    logger.info(f"Total violations: {sum(metrics['episode_violations'])}")
    logger.info(f"Production totale: {metrics['episode_productions'][-1] if metrics['episode_productions'] else 0}")
    
    if use_shield:
        final_shield_stats = system.get_shield_stats()
        logger.info(f"Shield - Actions corrigées: {final_shield_stats.get('corrected_actions', 0)}")
        logger.info(f"Shield - Actions bloquées: {final_shield_stats.get('blocked_actions', 0)}")
        logger.info(f"Shield - Taux d'intervention: {final_shield_stats.get('intervention_rate', 0):.2%}")
    
    logger.info("="*70)
    
    # Sauvegarde des métriques
    output_dir = Path("results/livrable3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": algo_name,
        "use_shield": use_shield,
        "num_episodes": num_episodes,
        "episode_length": episode_length,
        "metrics": {
            "episode_rewards": [float(r) for r in metrics['episode_rewards']],
            "episode_violations": [int(v) for v in metrics['episode_violations']],
            "episode_safety_rates": [float(s) for s in metrics['episode_safety_rates']],
            "episode_productions": [int(p) for p in metrics['episode_productions']],
            "episode_corrections": [int(c) for c in metrics['episode_corrections']],
            "actor_losses": [float(l) for l in metrics['actor_losses']],
            "critic_losses": [float(l) for l in metrics['critic_losses']]
        },
        "summary": {
            "mean_reward": float(np.mean(metrics['episode_rewards'])),
            "std_reward": float(np.std(metrics['episode_rewards'])),
            "mean_safety": float(np.mean(metrics['episode_safety_rates'])),
            "std_safety": float(np.std(metrics['episode_safety_rates'])),
            "total_violations": int(sum(metrics['episode_violations'])),
            "total_production": int(metrics['episode_productions'][-1]) if metrics['episode_productions'] else 0
        }
    }
    
    if use_shield:
        results["shield_stats"] = system.get_shield_stats()
        
        # Sauvegarder les explications
        explanations = system.get_all_explanations()
        with open(output_dir / "explanations.json", "w", encoding='utf-8') as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / f"{algo_name.lower().replace('-', '_')}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Résultats sauvegardés dans {output_dir}/")
    
    return system, metrics


def compare_with_baseline(num_episodes: int = 50):
    """
    Compare MAPPO-NS avec MAPPO standard
    
    Returns:
        Dictionnaire de comparaison
    """
    print("\n" + "="*70)
    print("COMPARAISON: MAPPO Standard vs MAPPO-NS (Neurosymbolique)")
    print("="*70)
    
    # Entraînement MAPPO standard (sans shield)
    print("\n📊 Entraînement MAPPO Standard (sans shield)...")
    system_std, metrics_std = train_mappo_ns(num_episodes=num_episodes, use_shield=False)
    
    # Entraînement MAPPO-NS (avec shield)
    print("\n🛡️ Entraînement MAPPO-NS (avec shield neurosymbolique)...")
    system_ns, metrics_ns = train_mappo_ns(num_episodes=num_episodes, use_shield=True)
    
    # Comparaison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "num_episodes": num_episodes,
        "standard": {
            "mean_safety": np.mean(metrics_std['episode_safety_rates']),
            "mean_reward": np.mean(metrics_std['episode_rewards']),
            "total_violations": sum(metrics_std['episode_violations']),
            "total_production": metrics_std['episode_productions'][-1] if metrics_std['episode_productions'] else 0
        },
        "neurosymbolic": {
            "mean_safety": np.mean(metrics_ns['episode_safety_rates']),
            "mean_reward": np.mean(metrics_ns['episode_rewards']),
            "total_violations": sum(metrics_ns['episode_violations']),
            "total_production": metrics_ns['episode_productions'][-1] if metrics_ns['episode_productions'] else 0
        }
    }
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("📊 RÉSULTATS DE LA COMPARAISON")
    print("="*70)
    print(f"\n{'Métrique':<25} {'MAPPO Standard':<20} {'MAPPO-NS':<20}")
    print("-"*70)
    print(f"{'Taux sécurité':<25} {comparison['standard']['mean_safety']*100:>6.2f}%        {comparison['neurosymbolic']['mean_safety']*100:>6.2f}%")
    print(f"{'Reward moyen':<25} {comparison['standard']['mean_reward']:>+12.0f}    {comparison['neurosymbolic']['mean_reward']:>+12.0f}")
    print(f"{'Violations totales':<25} {comparison['standard']['total_violations']:>12,}    {comparison['neurosymbolic']['total_violations']:>12,}")
    print(f"{'Production totale':<25} {comparison['standard']['total_production']:>12}    {comparison['neurosymbolic']['total_production']:>12}")
    print("="*70)
    
    # Sauvegarde
    output_dir = Path("results/livrable3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Comparaison sauvegardée dans {output_dir}/comparison_results.json")
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LIVRABLE 3 - Entraînement MAPPO-NS')
    parser.add_argument('--compare', action='store_true', 
                       help='Comparer MAPPO Standard vs MAPPO-NS')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Nombre d\'épisodes d\'entraînement')
    parser.add_argument('--no-shield', action='store_true',
                       help='Désactiver le shield (MAPPO standard)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_baseline(num_episodes=args.episodes)
    else:
        train_mappo_ns(num_episodes=args.episodes, use_shield=not args.no_shield)