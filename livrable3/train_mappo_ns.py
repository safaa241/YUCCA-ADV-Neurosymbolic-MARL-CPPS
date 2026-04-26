# Entraîne MAPPO avec ou sans shield neurosymbolique
# Version avec seuils dynamiques, tests robustesse, et injection de bruit

from asyncio.log import logger
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import logging
import random
from typing import Dict, List, Tuple, Optional

# Dans train_mappo_ns.py, à chaque changement de produit
from knowledge_base import KnowledgeBase
PRODUCTION_CONTEXTS = KnowledgeBase.PRODUCTION_CONTEXTS

# Import des modules
from mappo_ns_agent import MultiAgentMAPPO_NS
from cpps_environment import CPPSProductionEnv, PRODUCT_TYPES, PRODUCT_THRESHOLDS


def convert_to_serializable(obj):
    """Convertit les objets numpy en types Python natifs pour JSON"""
    import numpy as np
    
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
    
# ============================================================================
# MODULE D'INJECTION DE BRUIT (Intégré directement)
# ============================================================================

# ========== 5. INJECTEUR DE BRUIT / DÉFAILLANCES CAPTEURS ==========

class SensorNoiseInjector:
    """Simule des défaillances de capteurs pour tester la robustesse"""
    
    def __init__(self, fault_probability: float = 0.02):
        self.fault_probability = fault_probability
        self.fault_types = ["nan", "out_of_range", "spike", "stuck"]
        self.fault_counters = {ft: 0 for ft in self.fault_types}
        self.stuck_values = {}
    
    def inject_noise(self, observation: np.ndarray, step: int, agent_id: int):
        if random.random() > self.fault_probability:
            return observation, None
        
        fault_type = random.choice(self.fault_types)
        obs_copy = observation.copy()
        
        if fault_type == "nan":
            sensor_idx = random.randint(0, 5)
            obs_copy[sensor_idx] = np.nan
            msg = f"NaN sur capteur {sensor_idx}"
            
        elif fault_type == "out_of_range":
            sensor_idx = random.randint(0, 5)
            obs_copy[sensor_idx] = random.uniform(1.5, 3.0)
            msg = f"Valeur {obs_copy[sensor_idx]:.2f} hors [0,1]"
            
        elif fault_type == "spike":
            obs_copy[0] = min(1.2, obs_copy[0] + random.uniform(0.3, 0.8))
            msg = f"Pic de température: +{random.uniform(0.3,0.8):.2f}"
            
        elif fault_type == "stuck":
            stuck_key = f"stuck_{agent_id}"
            if stuck_key not in self.stuck_values:
                self.stuck_values[stuck_key] = observation.mean()
            obs_copy[:] = self.stuck_values[stuck_key]
            msg = f"Capteurs bloqués à {self.stuck_values[stuck_key]:.2f}"
        
        self.fault_counters[fault_type] += 1
        return obs_copy, msg
    
    def get_statistics(self):
        total = sum(self.fault_counters.values())
        return {"total_faults": total, "by_type": self.fault_counters}


# ========== 6. SCÉNARIOS DE TEST ==========

def create_rapid_temperature_rise(observation: np.ndarray, step: int, 
                                   start_step: int = 100, 
                                   rise_rate: float = 0.02) -> np.ndarray:
    """Simule une montée rapide de température (panne)"""
    obs_copy = observation.copy()
    if step >= start_step:
        increase = min(0.8, (step - start_step) * rise_rate)
        obs_copy[0] = min(1.0, observation[0] + increase)
        obs_copy[1] = min(1.0, observation[1] + increase * 0.5)
    return obs_copy


def create_rule_conflict_scenario(observation: np.ndarray, conflict_type: str = "temp_pressure") -> np.ndarray:
    """Crée un état qui déclenche plusieurs règles simultanément"""
    obs_copy = observation.copy()
    if conflict_type == "temp_pressure":
        obs_copy[0] = 0.96  # ~816°C → R3
        obs_copy[1] = 0.92  # ~9.2 bar → R4
    return obs_copy


# ========== 7. GESTION DES CONTEXTES DYNAMIQUES ==========

PRODUCTION_CYCLE = ['steel', 'aluminium', 'steel', 'titanium', 'plastic', 'steel']
CYCLE_LENGTH = 10  # Changement tous les 10 épisodes


def update_production_context(env, episode: int):
    """Change le contexte de production tous les N épisodes"""
    cycle_position = (episode // CYCLE_LENGTH) % len(PRODUCTION_CYCLE)
    new_context = PRODUCTION_CYCLE[cycle_position]
    
    current = getattr(update_production_context, 'last_context', None)
    
    if current != new_context:
        update_production_context.last_context = new_context
        env.set_product_type(new_context)
        return True, new_context
    
    return False, current

# ============================================================================
# GESTION DES SEUILS DYNAMIQUES
# ============================================================================

PRODUCTION_CYCLE = ['steel', 'aluminium', 'steel', 'titanium', 'plastic', 'steel']
CYCLE_LENGTH = 10  # Changement tous les 10 épisodes


def update_production_context(env, episode: int, logger: logging.Logger = None) -> Tuple[bool, str]:
    """
    Change le contexte de production tous les N épisodes
    """
    cycle_position = (episode // CYCLE_LENGTH) % len(PRODUCTION_CYCLE)
    new_context = PRODUCTION_CYCLE[cycle_position]
    
    current = getattr(update_production_context, 'last_context', None)
    
    if current != new_context:
        update_production_context.last_context = new_context
        env.set_product_type(new_context)
        
        if logger:
            thresholds = PRODUCT_THRESHOLDS[new_context]
            logger.info(f"🔄 [CONTEXTE] Changement: {current} → {new_context}")
            logger.info(f"   Température max: {thresholds['temperature_max']}°C")
            logger.info(f"   Pression max: {thresholds['pressure_max']} bar")
        
        return True, new_context
    
    return False, current


# ============================================================================
# LOGGER
# ============================================================================

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


# ============================================================================
# ENTRAÎNEMENT PRINCIPAL
# ============================================================================

def train_mappo_ns(num_episodes: int = 50,
                   episode_length: int = 500,
                   use_shield: bool = True,
                   log_interval: int = 5,
                   save_interval: int = 25,
                   test_robustness: bool = True,
                   dynamic_context: bool = True) -> Tuple[MultiAgentMAPPO_NS, Dict]:
    """
    Entraîne MAPPO-NS avec options avancées
    
    Args:
        num_episodes: Nombre d'épisodes
        episode_length: Longueur par épisode
        use_shield: Activer/désactiver le shield
        log_interval: Intervalle d'affichage
        save_interval: Intervalle de sauvegarde
        test_robustness: Activer les tests d'injection de bruit
        dynamic_context: Activer les changements dynamiques de contexte
        
    Returns:
        (système entraîné, métriques)
    """
    
    algo_name = "MAPPO-NS" if use_shield else "MAPPO-Standard"
    logger = setup_logger(algo_name)
    
    logger.info("="*70)
    logger.info(f"LIVRABLE 3 - Entraînement {algo_name}")
    logger.info(f"Shield neurosymbolique: {'ACTIF' if use_shield else 'INACTIF'}")
    logger.info(f"Tests robustesse: {'ACTIFS' if test_robustness else 'INACTIFS'}")
    logger.info(f"Contexte dynamique: {'ACTIF' if dynamic_context else 'INACTIF'}")
    logger.info("="*70)
    
    # Environnement
    env = CPPSProductionEnv(num_agents=3, episode_length=episode_length, product_type='steel')
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
    
    # Initialiser l'injecteur de bruit pour les tests robustesse
    noise_injector = SensorNoiseInjector(fault_probability=0.02) if test_robustness else None
    
    # Métriques
    metrics = {
        'episode_rewards': [],
        'episode_violations': [],
        'episode_safety_rates': [],
        'episode_productions': [],
        'episode_corrections': [],
        'actor_losses': [],
        'critic_losses': [],
        'context_changes': [],
        'robustness_events': []
    }
    
    robustness_stats = {
        "sensor_faults_detected": 0,
        "rapid_rise_events": 0,
        "conflict_resolutions": 0,
        "recovery_actions": 0
    }
    
    # Boucle d'entraînement
    for episode in range(num_episodes):
        # Changement dynamique de contexte
        if dynamic_context:
            context_changed, new_context = update_production_context(env, episode, logger)
            if context_changed:
                metrics['context_changes'].append({
                    "episode": episode,
                    "new_context": new_context,
                    "thresholds": env.get_current_thresholds()
                })
                # Mettre à jour les seuils du shield pour tous les agents
                if use_shield:
                    for agent_id, agent in system.agents.items():
                        if hasattr(agent, 'kb') and agent.kb:
                            agent.kb.update_thresholds_for_product(new_context)
        
        observations, _ = env.reset()
        episode_reward = 0
        episode_violations = 0
        episode_corrections = 0
        episode_robustness_events = []
        
        for step in range(episode_length):
            # ========== TESTS DE ROBUSTESSE INTÉGRÉS ==========
            
            # 1. Test de montée rapide de température (tous les 1000 steps)
            if test_robustness and step % 1000 == 0 and step > 0:
                for agent_id in range(env.num_agents):
                    observations[agent_id] = create_rapid_temperature_rise(
                        observations[agent_id], step, start_step=step, rise_rate=0.05
                    )
                robustness_stats["rapid_rise_events"] += 1
                episode_robustness_events.append({"type": "rapid_rise", "step": step})
                if step % 2000 == 0:
                    logger.warning(f"⚠️ [ROBUSTESSE] Montée rapide de température simulée au step {step}")
            
            # 2. Test de conflit de règles (tous les 2000 steps)
            if test_robustness and step % 2000 == 0 and step > 0:
                for agent_id in range(env.num_agents):
                    observations[agent_id] = create_rule_conflict_scenario(
                        observations[agent_id], conflict_type="temp_pressure"
                    )
                robustness_stats["conflict_resolutions"] += 1
                episode_robustness_events.append({"type": "rule_conflict", "step": step})
                if step % 2000 == 0:
                    logger.warning(f"⚠️ [ROBUSTESSE] Conflit de règles simulé au step {step}")
            
            # ========== SÉLECTION DES ACTIONS ==========
            actions = {}
            values = {}
            log_probs = {}
            
            for agent_id, obs in observations.items():
                # Injection aléatoire de bruit capteur
                if test_robustness and noise_injector:
                    noisy_obs, fault_report = noise_injector.inject_noise(obs, step, agent_id)
                    if fault_report:
                        robustness_stats["sensor_faults_detected"] += 1
                        episode_robustness_events.append({
                            "type": "sensor_fault",
                            "step": step,
                            "agent": agent_id,
                            "message": fault_report
                        })
                        if step % 500 == 0:
                            logger.warning(f"🔴 [ROBUSTESSE] {fault_report}")
                        obs = noisy_obs
                
                action, log_prob, value = system.agents[agent_id].select_action(obs, explore=True)
                actions[agent_id] = action
                values[agent_id] = value
                log_probs[agent_id] = log_prob
            
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
        if episode_robustness_events:
            metrics['robustness_events'].extend(episode_robustness_events)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            logger.info(f"Episode {episode+1}/{num_episodes}")
            logger.info(f"  Reward: {episode_reward:.2f}")
            logger.info(f"  Safety: {safety_rate:.2%}")
            logger.info(f"  Violations: {episode_violations}")
            logger.info(f"  Production: {env.total_production}")
            logger.info(f"  Produit: {env.get_product_type()}")
            if use_shield:
                logger.info(f"  Shield corrections: {episode_corrections}")
            logger.info(f"  Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")
            logger.info("-" * 50)
        
        # Sauvegarde périodique
        if (episode + 1) % save_interval == 0:
            save_path = Path(f"results/checkpoints/{algo_name}_ep{episode+1}.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                system.agents[0].save(str(save_path))
            except AttributeError as e:
                logger.warning(f"Impossible de sauvegarder le modèle: {e}")
    
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
    
    # Rapport de robustesse
    if test_robustness:
        logger.info("\n" + "="*70)
        logger.info("🔬 RAPPORT DE ROBUSTESSE")
        logger.info("="*70)
        logger.info(f"  Défauts capteurs injectés: {robustness_stats['sensor_faults_detected']}")
        logger.info(f"  Montées rapides simulées: {robustness_stats['rapid_rise_events']}")
        logger.info(f"  Conflits de règles simulés: {robustness_stats['conflict_resolutions']}")
        if noise_injector:
            logger.info(f"  Statistiques injecteur: {noise_injector.get_statistics()}")
        
        # Rapport de tolérance aux pannes du shield
        if use_shield and system.agents[0].shield:
            ft_report = system.agents[0].shield.get_fault_tolerance_report()
            logger.info(f"  Taux de tolérance aux pannes: {ft_report['fault_tolerance_rate']:.2%}")
            logger.info(f"  Recommandation: {ft_report['recommendation']}")
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
        "dynamic_context": dynamic_context,
        "test_robustness": test_robustness,
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
        },
        "robustness_stats": robustness_stats if test_robustness else None,
        "context_changes": metrics['context_changes'] if dynamic_context else None
        }
    # Convertir les données avant sauvegarde
    results_serializable = convert_to_serializable(results)

    with open(output_dir / f"{algo_name.lower().replace('-', '_')}_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    if use_shield:
        results["shield_stats"] = system.get_shield_stats()
    
    # Sauvegarder les explications avec conversion
    explanations = system.get_all_explanations()
    
    # Fonction de conversion pour JSON
    def to_serializable(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        return obj
    
    try:
        explanations_serializable = to_serializable(explanations)
        with open(output_dir / "explanations.json", "w", encoding='utf-8') as f:
            json.dump(explanations_serializable, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Impossible de sauvegarder les explications: {e}")

# Convertir results avant sauvegarde
    results_serializable = to_serializable(results)

    with open(output_dir / f"{algo_name.lower().replace('-', '_')}_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Fonction pour convertir les booléens et autres types non sérialisables
    def convert_explanation(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_explanation(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_explanation(item) for item in obj]
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        else:
            return obj
    
    explanations_converted = convert_explanation(explanations)
    
    with open(output_dir / "explanations.json", "w", encoding='utf-8') as f:
        json.dump(explanations_converted, f, indent=2, ensure_ascii=False)

    with open(output_dir / f"{algo_name.lower().replace('-', '_')}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Résultats sauvegardés dans {output_dir}/")
    
    return system, metrics


def compare_with_baseline(num_episodes: int = 50, test_robustness: bool = True):
    """
    Compare MAPPO-NS avec MAPPO standard
    """
    print("\n" + "="*70)
    print("COMPARAISON: MAPPO Standard vs MAPPO-NS (Neurosymbolique)")
    print("="*70)
    
    # Entraînement MAPPO standard (sans shield)
    print("\n📊 Entraînement MAPPO Standard (sans shield)...")
    system_std, metrics_std = train_mappo_ns(
        num_episodes=num_episodes, 
        use_shield=False,
        test_robustness=test_robustness
    )
    
    # Entraînement MAPPO-NS (avec shield)
    print("\n🛡️ Entraînement MAPPO-NS (avec shield neurosymbolique)...")
    system_ns, metrics_ns = train_mappo_ns(
        num_episodes=num_episodes, 
        use_shield=True,
        test_robustness=test_robustness
    )
    
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
    parser.add_argument('--no-robustness', action='store_true',
                       help='Désactiver les tests de robustesse')
    parser.add_argument('--no-context', action='store_true',
                       help='Désactiver les changements de contexte dynamique')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_baseline(
            num_episodes=args.episodes,
            test_robustness=not args.no_robustness
        )
    else:
        train_mappo_ns(
            num_episodes=args.episodes, 
            use_shield=not args.no_shield,
            test_robustness=not args.no_robustness,
            dynamic_context=not args.no_context
        )

def test_robustness_scenarios():
    """
    Test des 3 scénarios de robustesse
    À exécuter séparément pour générer les preuves
    """
    print("\n" + "🔬"*35)
    print("TEST DES SCÉNARIOS DE ROBUSTESSE")
    print("🔬"*35)
    
    from knowledge_base import KnowledgeBase
    from neurosymbolic_shield import SymbolicShield
    
    kb = KnowledgeBase()
    shield = SymbolicShield(kb)
    
    # Scénario 1: Capteur défaillant (NaN)
    print("\n📊 Scénario 1: Capteur température défaillant (NaN)")
    obs_nan = np.array([np.nan, 0.5, 0.2, 0.1, 0, 0.3])
    obs_corrected, corrected, msg = shield.validate_observation(obs_nan)
    print(f"   Avant: {obs_nan}")
    print(f"   Après: {obs_corrected}")
    print(f"   Correction: {'✅' if corrected else '❌'}")
    print(f"   Message: {msg}")
    
    # Scénario 2: Montée rapide
    print("\n📊 Scénario 2: Montée rapide de température")
    normal = np.array([0.35, 0.5, 0.2, 0.1, 0, 0.3])
    for rise in [0, 0.3, 0.6, 0.9]:
        obs = normal.copy()
        obs[0] = min(1.0, obs[0] + rise)
        temp = obs[0] * 850
        action, modified, exp, _ = shield.filter_action(2, obs)
        print(f"   T={temp:.0f}°C: action 2 → {shield.action_names.get(action)} ({'modifiée' if modified else 'conservée'})")
        print(f"      → {exp[:60]}...")
    
    # Scénario 3: Conflit de règles
    print("\n📊 Scénario 3: Conflit de règles")
    conflict_state = {'temperature': 820, 'pressure': 9.5, 'speed': 5.0}
    conflicts = kb.detect_conflicts(conflict_state)
    print(f"   État: T=820°C, P=9.5 bar")
    print(f"   Conflits détectés: {len(conflicts)}")
    for c in conflicts:
        print(f"   - {c['type']}: {c.get('rules', [])} → {c.get('resolution', 'N/A')}")
    
    print("\n✅ Tests de robustesse terminés")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--test-robustness":
    test_robustness_scenarios()
    