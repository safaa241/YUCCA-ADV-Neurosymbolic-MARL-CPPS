# LIVRABLE 2 - Safe Reinforcement Learning pour CPPS

## Description

Ce livrable implémente et compare trois méthodes de **Safe Reinforcement Learning** (Safe RL) pour les Systèmes Cyber-Physiques de Production (CPPS). L'objectif est d'intégrer des contraintes de sécurité numériques dans l'algorithme MAPPO afin de réduire les violations par rapport au MARL standard (Livrable 1).

##  Objectifs

- Implémenter trois approches Safe RL sur l'environnement CPPS
- Comparer leurs performances en termes de sécurité et de production
- Analyser les compromis entre performance et sûreté
- Quantifier l'amélioration par rapport au MAPPO standard

##  Méthodes Implémentées

| Méthode | Fichier | Description |
|---------|---------|-------------|
| **Lagrangien** | `lagrangian_safety.py` | Multiplicateurs de Lagrange pour contrainte de coût |
| **CBF** | `cbf_shield.py` | Control Barrier Functions - garantie locale |
| **Adaptative** | `safe_mappo_agent.py` | Pénalités dynamiques basées sur l'historique |

##  Structure du Code
livrable2/

├── cbf_shield.py # Control Barrier Functions

├── dashboard_livrable2.py # Dashboard Streamlit pour visualisation

├── lagrangian_safety.py # Multiplicateurs Lagrangiens

├── run_safe_comparison.py # Script principal de comparaison

├── safe_mappo_agent.py # Agent MAPPO avec contraintes

└── train_safe_mappo.py # Entraînement des méthodes Safe RL


##  Installation

```
# Activer l'environnement conda
conda activate yucca_adv

# Installer les dépendances (si ce n'est pas déjà fait)
pip install -r ../requirements.txt
```

## Exécution
1. Comparaison complète (entraîne tous les modèles)
```
python run_safe_comparison.py --full
```

2. Mode rapide (charge les résultats existants)
```
python run_safe_comparison.py --quick
```

3. Exécuter une méthode spécifique
```
python train_safe_mappo.py --method cbf --episodes 50
```

4. Lancer le dashboard de visualisation
```
streamlit run dashboard_livrable2.py
```

## Résultats Sauvegardés

results/livrable2/

├── full_comparison_results.json   # Résultats complets
├── safe_rl_comparison.png         # Graphique de comparaison
├── safe_rl_learning_curves.png    # Courbes d'apprentissage
├── safe_rl_improvement.png        # Graphique d'amélioration
└── safe_rl_report.md              # Rapport Markdown
## Analyse des Résultats
### Constats clés
Amélioration significative : Safe RL améliore la sécurité de <1% à ~79%

CBF est la meilleure méthode : 78.5% de sécurité, 112 pièces produites

Plateau à ~79% : Aucune méthode Safe RL n'atteint 100% de sécurité

Limite fondamentale : Les contraintes numériques ne peuvent pas garantir formellement la sécurité

## Transition vers le Livrable 3
Les résultats montrent que Safe RL améliore mais ne garantit pas la sécurité. Cela justifie la nécessité d'une approche neurosymbolique (Livrable 3) où un shield symbolique garantit 100% de sécurité.

