# LIVRABLE 3 - MARL Neurosymbolique pour CPPS (MAPPO-NS)

## Description

Ce livrable implémente **MAPPO-NS (Multi-Agent Proximal Policy Optimization - Neurosymbolique)** , un algorithme combinant l'apprentissage par renforcement multi-agents avec un **shield symbolique** garantissant la sécurité. C'est le cœur scientifique du projet YUCCA-ADV.

##  Objectifs

- Garantir **100% de sécurité** pendant l'apprentissage et l'exécution
- Fournir des **explications lisibles** pour chaque décision
- Maintenir une **performance élevée** (production, reward)
- Comparer MAPPO-NS avec MAPPO standard, QMIX et MADDPG

## Structure du Code
livrable3/

├── dashboard_livrable3.py # Dashboard Streamlit

├── explainability.py # Module d'explicabilité

├── knowledge_base.py # Base de connaissances symboliques

├── mappo_ns_agent.py # Agent MAPPO-NS

├── neurosymbolic_shield.py # Shield neurosymbolique

├── run_neurosymbolic.py # Script principal

└── train_mappo_ns.py # Entraînement MAPPO-NS

##  Installation

```
# Activer l'environnement conda
conda activate yucca_adv
```

## ️ Exécution
1. Expérience complète (entraînement + comparaison)
```
python run_neurosymbolic.py --episodes 50
```

2. Mode rapide (charge les résultats existants)
```
python run_neurosymbolic.py --quick
```

3. Entraîner uniquement MAPPO-NS
```
python train_mappo_ns.py --episodes 50
```

4. Entraîner MAPPO standard (sans shield)
```
python train_mappo_ns.py --episodes 50 --no-shield
```

5. Comparer MAPPO vs MAPPO-NS
```
python train_mappo_ns.py --compare --episodes 50
```

6. Lancer le dashboard
```
streamlit run dashboard_livrable3.py
```

## Résultats Sauvegardés

results/livrable3/

├── complete_experiment_results.json   # Résultats complets

├── livrable3_report.pdf               # Rapport PDF

├── livrable3_report.md                # Rapport Markdown

├── explanations.json                  # Toutes les explications

├── comparison_graphs.png              # Graphiques comparatifs

├── mappo_ns_results.json              # Métriques MAPPO-NS

└── mappo_standard_results.json        # Métriques MAPPO standard


## Conclusion
MAPPO-NS démontre qu'il est possible de garantir 100% de sécurité tout en maintenant une performance élevée (140 pièces produites). Le shield neurosymbolique est une avancée majeure par rapport aux approches Safe RL classiques.

