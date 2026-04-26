# Nouveau : compare_all_algorithms.py

import numpy as np
from typing import Dict, List
from enum import Enum

class SafeRLMethod(Enum):
    NONE = "none"
    LAGRANGIAN = "lagrangian"
    CBF = "cbf"
    ADAPTIVE = "adaptive"
    SYMBOLIC = "symbolic"  # Pour MAPPO-NS

def compare_all_algorithms(num_episodes: int = 50):
    """
    Compare tous les algorithmes avec toutes les méthodes Safe RL
    Algorithmes: MAPPO, QMIX, MADDPG
    Méthodes: Standard, Lagrangien, CBF, Adaptative, Symbolique
    """
    
    algorithms = ['mappo', 'qmix', 'maddpg']
    methods = [SafeRLMethod.NONE, SafeRLMethod.LAGRANGIAN, 
               SafeRLMethod.CBF, SafeRLMethod.ADAPTIVE, SafeRLMethod.SYMBOLIC]
    
    results = {}
    
    for algo in algorithms:
        results[algo] = {}
        for method in methods:
            print(f"\n📊 Entraînement: {algo.upper()} + {method.value}")
            metrics = train_algorithm_with_safe_rl(algo, method, num_episodes)
            results[algo][method.value] = metrics
    
    # Affichage des résultats sous forme de tableau
    print("\n" + "="*100)
    print("TABLEAU COMPARATIF COMPLET - SÉCURITÉ (%)")
    print("="*100)
    
    print(f"\n{'Algorithme':<12} {'Standard':<12} {'Lagrangien':<12} {'CBF':<12} {'Adaptative':<12} {'Symbolique':<12}")
    print("-"*85)
    
    for algo in algorithms:
        row = f"{algo.upper():<12}"
        for method in methods:
            safety = results[algo][method.value]['mean_safety'] * 100
            marker = " ✓" if safety >= 99 else ""
            row += f" {safety:>6.1f}%{marker:<5}"
        print(row)
    
    return results