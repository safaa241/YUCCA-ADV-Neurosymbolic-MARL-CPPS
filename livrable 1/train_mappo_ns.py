# Entraînement MAPPO-NS (Neurosymbolique) avec un shield efficace
# Ce qu’il produit: Agents entraînés avec un shield neurosymbolique, métriques d’entraînement (récompenses, production, violations, etc.) et logs détaillés de l’entraînement.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

from cpps_environment import CPPSProductionEnv
from mappo_agent import MAPPOAgent
from symbolic_shield import SymbolicShield, ActionMasking

# Base de connaissances simple pour le shield
class KnowledgeBase:
    """Base de connaissances symboliques"""
    
    def __init__(self):
        # ========== INITIALISATION DES COMPTEURS ==========
        self.debug_counter = 0 # Compteur pour les logs de debug
        self.debug_counter = 0
        self.total_checks = 0
        self.rules_triggered = 0
        
        self.rules = [
            {
                "name": "temperature_critical",
                "condition": lambda s: s.get('temperature', 0) >= 850,
                "action": 4,
                "message": "⚠️ TEMPÉRATURE CRITIQUE > 850°C → ARRÊT D'URGENCE"
            },
            {
                "name": "temperature_high",
                "condition": lambda s: s.get('temperature', 0) > 800,
                "forbidden_actions": [2],
                "safe_action": 0,
                "message": "⚠️ Température élevée > 800°C → réduction vitesse"
            },
            {
                "name": "pressure_high",
                "condition": lambda s: s.get('pressure', 0) > 9.0,
                "forbidden_actions": [2],
                "safe_action": 0,
                "message": "⚠️ Pression élevée > 9.0 bar → réduction vitesse"
            },
            {
                "name": "maintenance_required",
                "condition": lambda s: s.get('maintenance_needed', False),
                "action": 4,
                "message": "🔧 Maintenance requise → arrêt obligatoire"
            }
        ]
    
    def get_recommended_action(self, state):
        """Vérifie les règles et retourne une action forcée si nécessaire"""
        for rule in self.rules:
            if rule["condition"](state):
                if "action" in rule:
                    return rule["action"], 1.0, rule["message"]
        return None, 0.0, None
    
    def get_safe_action(self, state, proposed_action):
        """Retourne une action sûre - VERSION FINALE"""
    
        # DEBUG
        if self.debug_counter % 500 == 0:
            print(f"[SHIELD] T={state.get('temperature', 0):.1f}°C, action_proposée={proposed_action}")
        self.debug_counter += 1
    
    # ========== RÈGLES DE SÉCURITÉ ==========
    
    # Règle 1: Température critique → STOP (MAIS seulement si vraiment critique)
        if state.get('temperature', 0) >= 850:
            return 4, "CRITICAL: Température > 850°C → STOP"
    
    # Règle 2: Maintenance → STOP
        if state.get('maintenance_needed', False):
            return 4, "Maintenance requise → STOP"
    
    # Règle 3: Température élevée (>800°C) → empêcher augmentation
        if state.get('temperature', 0) > 800:
            if proposed_action == 2:  # increase_speed
                return 1, "Température élevée → maintien (augmentation interdite)"
            return proposed_action, None
    
    # Règle 4: Pression élevée (>9.0 bar) → empêcher augmentation
        if state.get('pressure', 0) > 9.0:
            if proposed_action == 2:
                return 1, "Pression élevée → maintien (augmentation interdite)"
            return proposed_action, None
    
    # ========== ACTION PAR DÉFAUT ==========
    # Ne jamais retourner 4 (STOP) si ce n'est pas une urgence
        if proposed_action == 4 and state.get('temperature', 0) < 800:
            # Remplacer STOP par MAINTAIN si pas critique
            return 1, "STOP non nécessaire → maintien"
    
        return proposed_action, None

    def get_stats(self):
        """Retourne les statistiques du shield"""
        return {
            'total_checks': self.total_checks,
            'rules_triggered': self.rules_triggered,
            'trigger_rate': self.rules_triggered / max(1, self.total_checks)
        }
    
    def is_action_allowed(self, state, action):
        """Vérifie si une action est autorisée"""
        # Température critique
        if state.get('temperature', 0) >= 850 and action != 4:
            return False, "Température critique - STOP requis"
        
        # Maintenance
        if state.get('maintenance_needed', False) and action != 4:
            return False, "Maintenance requise - STOP requis"
        
        # Température élevée
        if state.get('temperature', 0) > 800 and action == 2:
            return False, "Température élevée - augmentation interdite"
        
        # Pression élevée
        if state.get('pressure', 0) > 9.0 and action == 2:
            return False, "Pression élevée - augmentation interdite"
        
        return True, None
    
# ========== CORRECTION 3: Fonction d'entraînement avec shield efficace ==========
def train_mappo_ns(num_episodes=50, use_shield=True):
    """Entraîne MAPPO avec Shield Neurosymbolique"""
    
    # Logger
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"results/logs/PART3_MAPPO_NS_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("PARTIE 3: MAPPO-NS (Neurosymbolique)")
    logger.info(f"Shield activé: {use_shield}")
    logger.info("="*60)
    
    # Environnement
    env = CPPSProductionEnv(num_agents=3, episode_length=500)
    logger.info(f"Environnement créé avec {env.num_agents} agents")
    
    # Base de connaissances et Shield
    kb = KnowledgeBase()
    shield = SymbolicShield(kb)
    action_masker = ActionMasking()
    
    # Agents MAPPO standards
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
    logger.info(f"{env.num_agents} agents MAPPO créés")
    
    # Statistiques du shield
    blocked_actions = 0
    corrected_actions = 0
    
    # Métriques
    all_rewards = []
    all_violations = []
    all_safety_rates = []
    all_productions = []
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_total_reward = 0
        episode_violations = 0
        episode_blocked = 0
        episode_corrected = 0
        episode_production = 0
        
        for step in range(500):
            actions = {}
            
            for agent_id in range(env.num_agents):
                # 1. Agent propose une action
                raw_action, log_prob, value = agents[agent_id].select_action(observations[agent_id])
                
                # 2. Convertir l'observation en dictionnaire
                state_dict = {
                    'temperature': observations[agent_id][0] * 850,
                    'pressure': observations[agent_id][1] * 10,
                    'speed': observations[agent_id][2] * 10,
                    'production_count': observations[agent_id][3],
                    'maintenance_needed': observations[agent_id][4] > 0.5,
                    'time_step': observations[agent_id][5]
                }
                
                # 3. ========== SHIELD NEUROSYMBOLIQUE CORRIGÉ ==========
                if use_shield:
                    # Appliquer le shield pour obtenir une action sûre
                    final_action, reason = kb.get_safe_action(state_dict, raw_action)
                    
                    if reason is not None:
                        # L'action a été modifiée
                        episode_corrected += 1
                        corrected_actions += 1
                        if reason.startswith("CRITICAL"):
                            episode_blocked += 1
                            blocked_actions += 1
                        # Log debug optionnel (décommenter pour voir les corrections)
                        # if episode_corrected % 100 == 0:
                        #     logger.debug(f"Agent {agent_id}: {reason}")
                    else:
                        final_action = raw_action
                else:
                    final_action = raw_action
                
                actions[agent_id] = final_action
                
                # Stocker la transition
                agents[agent_id].store_transition(
                    observations[agent_id],
                    final_action,
                    0,  # reward temporaire
                    value,
                    log_prob,
                    False
                )
            
            # Exécuter les actions filtrées
            observations, rewards, dones, truncated, info = env.step(actions)
            
            # Mettre à jour les récompenses dans la mémoire
            for agent_id in range(env.num_agents):
                agents[agent_id].memory['rewards'][-1] = rewards[agent_id]
                agents[agent_id].memory['dones'][-1] = dones[agent_id]
                episode_total_reward += rewards[agent_id]
                episode_violations += len(info[agent_id]['violations'])
            
            episode_production = env.total_production
        
        # Mise à jour des agents
        for agent_id in range(env.num_agents):
            next_obs = observations[agent_id]
            _, _, next_value = agents[agent_id].select_action(next_obs)
            returns, advantages = agents[agent_id].compute_returns_and_advantages(next_value.item())
            agents[agent_id].update(returns, advantages)
        
        # Métriques
        all_rewards.append(episode_total_reward)
        all_violations.append(episode_violations)
        all_productions.append(episode_production)
        
        # Calcul du taux de sécurité
        total_possible_violations = 500 * 3
        safety_rate = 1 - (episode_violations / total_possible_violations) if episode_violations > 0 else 1.0
        all_safety_rates.append(safety_rate)
        
        # Logging
        if (episode + 1) % 5 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"  Reward: {episode_total_reward:.2f}")
            logger.info(f"  Production: {episode_production}")
            logger.info(f"  Violations: {episode_violations}")
            logger.info(f"  Safety Rate: {safety_rate:.2%}")
            logger.info(f"  Actions corrigées: {episode_corrected}")
            logger.info(f"  Actions bloquées: {episode_blocked}")
    
    # Résultats finaux
    logger.info("\n" + "="*60)
    logger.info("RÉSULTATS FINAUX - MAPPO-NS")
    logger.info(f"Reward Moyen: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    logger.info(f"Production Totale: {all_productions[-1] if all_productions else 0}")
    logger.info(f"Taux de Sécurité Moyen: {np.mean(all_safety_rates):.2%}")
    logger.info(f"Total Violations: {sum(all_violations)}")
    logger.info(f"Actions corrigées (total): {corrected_actions}")
    logger.info(f"Actions bloquées (total): {blocked_actions}")
    logger.info("="*60)
    
    # Sauvegarde
    results_dir = Path("results/part3")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "episode_count": num_episodes,
        "use_shield": use_shield,
        "metrics": {
            "total_reward": all_rewards,
            "total_production": all_productions,
            "total_violations": all_violations,
            "safety_rate": all_safety_rates
        },
        "shield_stats": {
            "blocked_actions": blocked_actions,
            "corrected_actions": corrected_actions
        }
    }
    
    with open(results_dir / "metrics_ns.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Résultats sauvegardés dans {results_dir / 'metrics_ns.json'}")
    
    return agents, all_rewards, all_violations, all_safety_rates


if __name__ == "__main__":
    # ========== CORRECTION 4: Entraînement complet ==========
    print("\n" + "🔬"*30)
    print("COMPARAISON MAPPO vs MAPPO-NS")
    print("🔬"*30 + "\n")
    
    # Entraînement avec shield (MAPPO-NS)
    print("🛡️ Entraînement MAPPO-NS (avec shield neurosymbolique)...")
    agents_ns, rewards_ns, violations_ns, safety_ns = train_mappo_ns(num_episodes=50, use_shield=True)
    
    print("\n" + "="*60)
    print("📊 RÉSULTATS MAPPO-NS")
    print("="*60)
    print(f"Reward Moyen: {np.mean(rewards_ns):.2f}")
    print(f"Taux de Sécurité: {np.mean(safety_ns):.2%}")
    print(f"Total Violations: {sum(violations_ns):,}")
    print("="*60)