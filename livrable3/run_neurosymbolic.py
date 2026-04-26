# Lance l’expérience complète (entraînement + comparaison) pour le Livrable 3, avec des métriques détaillées et un rapport complet
# ce qu'il produit : un script qui exécute l'entraînement de MAPPO-NS, compare les résultats avec MAPPO standard, génère un rapport détaillé avec des graphiques et des explications, et sauvegarde tous les résultats dans un format structuré pour une analyse approfondie. Le script inclut également une option de "mode rapide" pour charger les résultats existants sans réentraînement.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import time

from train_mappo_ns import train_mappo_ns, compare_with_baseline



def run_complete_experiment(num_episodes: int = 50):
    """
    Lance l'expérience complète du Livrable 3
    
    Étapes:
    1. Entraînement MAPPO-NS
    2. Comparaison avec MAPPO standard
    3. Génération du rapport
    """
    
    print("\n" + "🧠"*35)
    print("LIVRABLE 3 - MARL NEUROSYMBOLIQUE POUR CPPS")
    print("🧠"*35)
    
    print("\n📋 Configuration:")
    print(f"  - Épisodes: {num_episodes}")
    print(f"  - Steps par épisode: 500")
    print(f"  - Nombre d'agents: 3")
    print(f"  - Actions: 5 (reduce, maintain, increase, idle, stop)")
    
    # 1. Entraînement MAPPO-NS
    print("\n" + "="*70)
    print("🛡️ ÉTAPE 1: Entraînement MAPPO-NS (Neurosymbolique)")
    print("="*70)
    
    start_time = time.time()
    system_ns, metrics_ns = train_mappo_ns(num_episodes=num_episodes, use_shield=True)
    elapsed_ns = time.time() - start_time
    
    # 2. Entraînement MAPPO standard pour comparaison
    print("\n" + "="*70)
    print("📊 ÉTAPE 2: Entraînement MAPPO Standard (Baseline)")
    print("="*70)
    
    start_time = time.time()
    system_std, metrics_std = train_mappo_ns(num_episodes=num_episodes, use_shield=False)
    elapsed_std = time.time() - start_time
    
    # 3. Calcul des métriques de comparaison
    print("\n" + "="*70)
    print("📈 ÉTAPE 3: Calcul des métriques de comparaison")
    print("="*70)
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "num_episodes": num_episodes,
        "training_times": {
            "mappo_std": elapsed_std,
            "mappo_ns": elapsed_ns
        },
        "standard": {
            "mean_safety": float(np.mean(metrics_std['episode_safety_rates'])),
            "std_safety": float(np.std(metrics_std['episode_safety_rates'])),
            "mean_reward": float(np.mean(metrics_std['episode_rewards'])),
            "std_reward": float(np.std(metrics_std['episode_rewards'])),
            "total_violations": int(sum(metrics_std['episode_violations'])),
            "total_production": int(metrics_std['episode_productions'][-1]) if metrics_std['episode_productions'] else 0,
            "safety_rates": [float(s) for s in metrics_std['episode_safety_rates']],
            "rewards": [float(r) for r in metrics_std['episode_rewards']],
            "violations": [int(v) for v in metrics_std['episode_violations']],
            "productions": [int(p) for p in metrics_std['episode_productions']]
        },
        "neurosymbolic": {
            "mean_safety": float(np.mean(metrics_ns['episode_safety_rates'])),
            "std_safety": float(np.std(metrics_ns['episode_safety_rates'])),
            "mean_reward": float(np.mean(metrics_ns['episode_rewards'])),
            "std_reward": float(np.std(metrics_ns['episode_rewards'])),
            "total_violations": int(sum(metrics_ns['episode_violations'])),
            "total_production": int(metrics_ns['episode_productions'][-1]) if metrics_ns['episode_productions'] else 0,
            "safety_rates": [float(s) for s in metrics_ns['episode_safety_rates']],
            "rewards": [float(r) for r in metrics_ns['episode_rewards']],
            "violations": [int(v) for v in metrics_ns['episode_violations']],
            "productions": [int(p) for p in metrics_ns['episode_productions']],
            "shield_stats": system_ns.get_shield_stats()
        }
    }
    
    # Calcul des améliorations
    safety_improvement = (comparison['neurosymbolic']['mean_safety'] - comparison['standard']['mean_safety']) * 100
    violation_reduction = (1 - comparison['neurosymbolic']['total_violations'] / max(1, comparison['standard']['total_violations'])) * 100
    reward_improvement = comparison['neurosymbolic']['mean_reward'] - comparison['standard']['mean_reward']
    production_gain = comparison['neurosymbolic']['total_production'] - comparison['standard']['total_production']
    
    comparison['improvements'] = {
        "safety_gain_points": safety_improvement,
        "violation_reduction_percent": violation_reduction,
        "reward_improvement": reward_improvement,
        "production_gain": production_gain
    }
    
    # 4. Sauvegarde des résultats
    print("\n" + "="*70)
    print("💾 ÉTAPE 4: Sauvegarde des résultats")
    print("="*70)
    
    output_dir = Path("results/livrable3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "complete_experiment_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✅ Résultats sauvegardés dans {output_dir}/complete_experiment_results.json")
    
    # 5. Génération du rapport
    print("\n" + "="*70)
    print("📝 ÉTAPE 5: Génération du rapport")
    print("="*70)
    
    generate_full_report(comparison, output_dir)
    
    # 6. Affichage du résumé
    print("\n" + "="*70)
    print("🏆 RÉSUMÉ FINAL - LIVRABLE 3")
    print("="*70)
    
    print(f"\n{'Métrique':<30} {'MAPPO Standard':<20} {'MAPPO-NS':<20}")
    print("-"*70)
    print(f"{'Taux sécurité':<30} {comparison['standard']['mean_safety']*100:>6.2f}%        {comparison['neurosymbolic']['mean_safety']*100:>6.2f}%")
    print(f"{'Reward moyen':<30} {comparison['standard']['mean_reward']:>+12.0f}    {comparison['neurosymbolic']['mean_reward']:>+12.0f}")
    print(f"{'Violations totales':<30} {comparison['standard']['total_violations']:>12,}    {comparison['neurosymbolic']['total_violations']:>12,}")
    print(f"{'Production totale':<30} {comparison['standard']['total_production']:>12}    {comparison['neurosymbolic']['total_production']:>12}")
    
    print("\n" + "-"*70)
    print("📈 AMÉLIORATIONS APPORTÉES PAR MAPPO-NS:")
    print("-"*70)
    print(f"  🔒 Sécurité: +{safety_improvement:.1f} points")
    print(f"  ⚠️ Violations: -{violation_reduction:.1f}%")
    print(f"  📈 Reward: +{reward_improvement:.0f}")
    print(f"  🏭 Production: +{production_gain} pièces")
    
    if comparison['neurosymbolic']['mean_safety'] == 1.0:
        print("\n" + "🎉"*20)
        print("✅ OBJECTIF ATTEINT: 100% DE SÉCURITÉ GARANTIE !")
        print("🎉"*20)
    
    print("\n" + "="*70)
    print("📁 Fichiers générés:")
    print(f"   - {output_dir}/complete_experiment_results.json")
    print(f"   - {output_dir}/livrable3_report.pdf")
    print(f"   - {output_dir}/livrable3_report.md")
    print(f"   - {output_dir}/explanations.json")
    print(f"   - {output_dir}/comparison_graphs.png")
    print("="*70)
    
    return comparison


def quick_run():
    """Version rapide - charge les résultats existants"""
    print("\n⚡ Mode rapide - Chargement des résultats existants")
    
    results_path = Path("results/livrable3/complete_experiment_results.json")
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            comparison = json.load(f)
        
        print("\n📊 Résultats chargés:")
        print(f"  MAPPO-NS Sécurité: {comparison['neurosymbolic']['mean_safety']*100:.2f}%")
        print(f"  MAPPO Standard Sécurité: {comparison['standard']['mean_safety']*100:.2f}%")
        
        output_dir = Path("results/livrable3")
        generate_full_report(comparison, output_dir)
        
        return comparison
    else:
        print("⚠️ Aucun résultat existant. Lancement de l'expérience complète...")
        return run_complete_experiment()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LIVRABLE 3 - MARL Neurosymbolique')
    parser.add_argument('--quick', action='store_true',
                       help='Mode rapide: charge les résultats existants')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Nombre d\'épisodes d\'entraînement')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_run()
    else:
        run_complete_experiment(num_episodes=args.episodes)