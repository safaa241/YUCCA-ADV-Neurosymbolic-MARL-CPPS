# Lance l’entraînement MAPPO standard
# Ce qu’il produit: Agents entraînés, métriques d’entraînement (récompenses, production, violations, etc.) et logs détaillés de l’entraînement.

import sys
import os

# Ajouter le chemin parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import numpy as np
import yaml
from pathlib import Path

# Import des modules locaux
from cpps_environment import CPPSProductionEnv
from mappo_agent import MAPPOAgent

# Configuration du logger simple (sans le module utils)
import logging
from datetime import datetime

def setup_logger(name):
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


def train_mappo(num_episodes=50):
    """
    Entraîne les agents MAPPO
    """
    logger = setup_logger("PART1_MAPPO")
    logger.info("Démarrage de l'entraînement MAPPO classique (PARTIE 1)")
    
    # Environnement
    env = CPPSProductionEnv(num_agents=3, episode_length=500)
    logger.info(f"Environnement créé avec {env.num_agents} agents")
    
    # Agents
    agents = {}
    obs_dim = env.observation_spaces[0].shape[0]
    action_dim = env.action_spaces[0].n
    
    for i in range(env.num_agents):
        agents[i] = MAPPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            learning_rate=0.0003,
            gamma=0.99
        )
        logger.info(f"Agent {i} créé")
    
    # Stockage des métriques
    all_rewards = []
    all_productions = []
    all_violations = []
    all_safety_rates = []
    
    # Boucle d'entraînement
    logger.info(f"Entraînement sur {num_episodes} épisodes...")
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_total_reward = 0
        episode_violations = 0
        
        # Boucle d'épisode
        for step in range(500):
            actions = {}
            
            # Sélectionner les actions
            for agent_id in range(env.num_agents):
                action, log_prob, value = agents[agent_id].select_action(observations[agent_id])
                actions[agent_id] = action
                agents[agent_id].store_transition(
                    observations[agent_id],
                    action,
                    0,
                    value,
                    log_prob,
                    False
                )
            
            # Étape d'environnement
            observations, rewards, dones, truncated, info = env.step(actions)
            
            # Mettre à jour les transitions
            for agent_id in range(env.num_agents):
                agents[agent_id].memory['rewards'][-1] = rewards[agent_id]
                agents[agent_id].memory['dones'][-1] = dones[agent_id]
                episode_total_reward += rewards[agent_id]
                episode_violations += len(info[agent_id]['violations'])
        
        # Update des agents
        actor_losses = []
        critic_losses = []
        
        for agent_id in range(env.num_agents):
            next_obs = observations[agent_id]
            _, _, next_value = agents[agent_id].select_action(next_obs)
            returns, advantages = agents[agent_id].compute_returns_and_advantages(next_value.item())
            actor_loss, critic_loss = agents[agent_id].update(returns, advantages)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        # Enregistrer les métriques
        all_rewards.append(episode_total_reward)
        all_productions.append(env.total_production)
        all_violations.append(episode_violations)
        
        safety_rate = 1 - (episode_violations / (500 * 3)) if episode_violations > 0 else 1.0
        all_safety_rates.append(safety_rate)
        
        # Logging
        if (episode + 1) % 5 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"  Reward: {episode_total_reward:.2f}")
            logger.info(f"  Production: {env.total_production}")
            logger.info(f"  Violations: {episode_violations}")
            logger.info(f"  Safety Rate: {safety_rate:.2%}")
            logger.info(f"  Actor Loss: {np.mean(actor_losses):.4f}")
            logger.info(f"  Critic Loss: {np.mean(critic_losses):.4f}")
    
    # Résultats finaux
    logger.info("\n=== RÉSULTATS FINAUX ===")
    logger.info(f"Reward Moyen: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    logger.info(f"Production Totale: {all_productions[-1]}")
    logger.info(f"Taux de Sécurité Moyen: {np.mean(all_safety_rates):.2%}")
    logger.info(f"Total Violations: {sum(all_violations)}")
    
    # Sauvegarder les résultats
    results_dir = Path("results/part1")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    results = {
        "episode_count": num_episodes,
        "metrics": {
            "total_reward": all_rewards,
            "total_production": all_productions,
            "total_violations": all_violations,
            "safety_rate": all_safety_rates
        }
    }
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Résultats sauvegardés dans {results_dir / 'metrics.json'}")
    
    return agents, all_rewards, all_violations


if __name__ == "__main__":
    agents, rewards, violations = train_mappo(num_episodes=50)