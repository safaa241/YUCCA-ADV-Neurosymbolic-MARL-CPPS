# Compare MAPPO vs heuristique
# Ce qu’il produit: Résultats de comparaison (récompenses, sécurité, violations) entre l’agent MAPPO entraîné et une heuristique de référence, avec une analyse des différences.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import des modules
from cpps_environment import CPPSProductionEnv
from mappo_agent import MAPPOAgent
from livrable1.baseline_heuristic import MultiAgentHeuristic, run_heuristic_episode


def load_mappo_results() -> Dict[str, Any]:
    """
    Charge les résultats MAPPO existants
    """
    results_path = Path("results/part1/metrics.json")
    
    if not results_path.exists():
        print("⚠️ Fichier MAPPO non trouvé. Lancez d'abord train_mappo.py")
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    return {
        'rewards': metrics.get('total_reward', []),
        'violations': metrics.get('total_violations', []),
        'safety_rates': metrics.get('safety_rate', []),
        'productions': metrics.get('total_production', []),
        'num_episodes': data.get('episode_count', len(metrics.get('total_reward', [])))
    }


def run_heuristic_evaluation(num_episodes: int = 50) -> Dict[str, Any]:
    """
    Exécute l'évaluation de l'heuristique
    """
    from cpps_environment import CPPSProductionEnv
    
    env = CPPSProductionEnv(num_agents=3, episode_length=500)
    heuristic = MultiAgentHeuristic(num_agents=3)
    
    rewards = []
    violations = []
    productions = []
    safety_rates = []
    
    print("\n" + "="*60)
    print("🤖 EXÉCUTION DE L'HEURISTIQUE")
    print("="*60)
    
    for episode in range(num_episodes):
        reward, violation_count, production, safety = run_heuristic_episode(env, heuristic)
        
        rewards.append(reward)
        violations.append(violation_count)
        productions.append(production)
        safety_rates.append(safety)
        
        if (episode + 1) % 10 == 0:
            print(f"  Épisode {episode+1}/{num_episodes}: "
                  f"Safety={safety:.2%}, Reward={reward:.0f}")
    
    return {
        'rewards': rewards,
        'violations': violations,
        'safety_rates': safety_rates,
        'productions': productions,
        'num_episodes': num_episodes,
        'agent_stats': heuristic.get_agent_statistics()
    }


def save_comparison_results(mappo_results: Dict, heuristic_results: Dict, output_dir: Path):
    """
    Sauvegarde les résultats de comparaison
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Préparer les données de comparaison
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "num_episodes": mappo_results.get('num_episodes', heuristic_results['num_episodes']),
        "algorithms": {
            "MAPPO": {
                "mean_reward": float(np.mean(mappo_results['rewards'])) if mappo_results['rewards'] else 0,
                "std_reward": float(np.std(mappo_results['rewards'])) if mappo_results['rewards'] else 0,
                "mean_safety": float(np.mean(mappo_results['safety_rates'])) if mappo_results['safety_rates'] else 0,
                "std_safety": float(np.std(mappo_results['safety_rates'])) if mappo_results['safety_rates'] else 0,
                "total_violations": int(sum(mappo_results['violations'])) if mappo_results['violations'] else 0,
                "total_production": int(sum(mappo_results['productions'])) if mappo_results['productions'] else 0,
                "rewards": [float(r) for r in mappo_results['rewards']],
                "violations": [int(v) for v in mappo_results['violations']],
                "safety_rates": [float(s) for s in mappo_results['safety_rates']]
            },
            "Heuristic": {
                "mean_reward": float(np.mean(heuristic_results['rewards'])),
                "std_reward": float(np.std(heuristic_results['rewards'])),
                "mean_safety": float(np.mean(heuristic_results['safety_rates'])),
                "std_safety": float(np.std(heuristic_results['safety_rates'])),
                "total_violations": int(sum(heuristic_results['violations'])),
                "total_production": int(sum(heuristic_results['productions'])),
                "rewards": [float(r) for r in heuristic_results['rewards']],
                "violations": [int(v) for v in heuristic_results['violations']],
                "safety_rates": [float(s) for s in heuristic_results['safety_rates']],
                "agent_stats": heuristic_results.get('agent_stats', {})
            }
        }
    }
    
    # Sauvegarde JSON
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✅ Résultats sauvegardés dans {output_dir}/comparison_results.json")
    
    return comparison_data


def print_comparison_summary(comparison_data: Dict):
    """
    Affiche un résumé de la comparaison
    """
    print("\n" + "="*70)
    print("📊 COMPARAISON MAPPO vs HEURISTIQUE")
    print("="*70)
    
    mappo = comparison_data['algorithms']['MAPPO']
    heuristic = comparison_data['algorithms']['Heuristic']
    
    print(f"\n{'Métrique':<30} {'MAPPO':<20} {'Heuristique':<20}")
    print("-"*70)
    print(f"{'Reward moyen':<30} {mappo['mean_reward']:>+15.2f}    {heuristic['mean_reward']:>+15.2f}")
    print(f"{'Taux de sécurité':<30} {mappo['mean_safety']:>14.2%}    {heuristic['mean_safety']:>14.2%}")
    print(f"{'Violations totales':<30} {mappo['total_violations']:>15,}    {heuristic['total_violations']:>15,}")
    print(f"{'Production totale':<30} {mappo['total_production']:>15,}    {heuristic['total_production']:>15,}")
    
    print("\n" + "-"*70)
    print("📈 ANALYSE")
    print("-"*70)
    
    safety_gap = (heuristic['mean_safety'] - mappo['mean_safety']) * 100
    violation_reduction = (1 - heuristic['total_violations'] / max(1, mappo['total_violations'])) * 100
    
    print(f"\n  • L'heuristique est {safety_gap:+.1f}% plus sûre que MAPPO")
    print(f"  • L'heuristique réduit les violations de {violation_reduction:.1f}%")
    
    if mappo['mean_safety'] < 0.10:
        print("\n  ⚠️ MAPPO a un taux de sécurité TRÈS FAIBLE (<10%)")
        print("  → Le MARL classique ne garantit PAS la sécurité industrielle")
    
    if heuristic['mean_safety'] > 0.90:
        print("\n  ✅ L'heuristique atteint un bon niveau de sécurité (>90%)")
        print("  → Une approche par règles peut être sûre mais peu performante")
    
    print("\n" + "="*70)


def main():
    """
    Fonction principale de comparaison
    """
    print("\n" + "🏭"*35)
    print("LIVRABLE 1 - Comparaison MAPPO vs Heuristique")
    print("🏭"*35)
    
    # Créer le dossier de sortie
    output_dir = Path("results/livrable1")
    
    # Charger les résultats MAPPO existants
    print("\n📂 Chargement des résultats MAPPO...")
    mappo_results = load_mappo_results()
    
    if mappo_results is None:
        print("\n❌ Impossible de charger MAPPO. Vérifiez que results/part1/metrics.json existe.")
        print("   Lancez d'abord: python train_mappo.py")
        return
    
    print(f"   MAPPO: {mappo_results['num_episodes']} épisodes chargés")
    print(f"   Reward moyen: {np.mean(mappo_results['rewards']):.2f}")
    print(f"   Taux de sécurité moyen: {np.mean(mappo_results['safety_rates']):.2%}")
    
    # Exécuter l'heuristique
    print("\n📂 Exécution de l'heuristique...")
    heuristic_results = run_heuristic_evaluation(num_episodes=mappo_results['num_episodes'])
    
    # Sauvegarder les résultats
    comparison_data = save_comparison_results(mappo_results, heuristic_results, output_dir)
    
    # Afficher le résumé
    print_comparison_summary(comparison_data)
    
    print("\n✅ Comparaison terminée !")
    print(f"📁 Résultats dans {output_dir}/comparison_results.json")
    print("\n💡 Prochaine étape: python livrable1/generate_report.py")


if __name__ == "__main__":
    main()