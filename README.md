---

# YUCCA-ADV - Systèmes Multi-Agents Neurosymboliques pour l'Industrie 4.0

---

##  Description du Projet

Ce projet implémente une architecture **multi-agents neurosymbolique** pour l'optimisation adaptative des Systèmes Cyber-Physiques de Production (CPPS) dans le contexte de l'Industrie 4.0.

L'approche combine :
- **Apprentissage par renforcement multi-agents (MARL)** : MAPPO pour la performance
- **Safe Reinforcement Learning** : CBF, Lagrangien, Pénalités Adaptatives
- **IA Neurosymbolique** : Shield symbolique pour garantie de sécurité à 100%
- **Explicabilité** : Génération d'explications lisibles pour chaque décision

---

## Résultats Clés

| Algorithme | Taux de sécurité | Violations | Production | Explicabilité |
|------------|------------------|------------|------------|---------------|
| MAPPO Standard | 0.85% | 74 360 | 3 | non |
| Safe RL (CBF) | 78.5% | 14 780 | 112 | non |
| QMIX | 0.60% | 74 800 | 0 | non |
| MADDPG | 0.50% | 74 900 | 0 | non |
| **MAPPO-NS (Notre Algo)** | **100%**  | **0**  | **140**  | **Oui** |

### Amélioration MAPPO-NS vs MAPPO Standard :
- Sécurité : **+99.15 points**
- Violations : **-100%**
- Production : **+137 pièces**
- Reward : **+46 315 points**

---

##  Structure du Projet

```
MARL-SafeRL-Neurosymbolic/
│
├── livrable1/                    # MARL Neurosymbolique (Base)
│   ├── cpps_environment.py       # Jumeau numérique CPPS
│   ├── mappo_agent.py            # Agent MAPPO standard
│   ├── symbolic_shield.py        # Shield symbolique
│   ├── train_mappo.py            # Entraînement MAPPO standard
│   ├── train_mappo_ns.py         # Entraînement MAPPO-NS
│   ├── algo_NS_MARL.py           # Version complète MAPPO-NS
│   ├── compare_all_marllib.py    # Comparaison 4 algorithmes
│   ├── generate_report.py        # Génération rapport
│   ├── dashboard3.py             # Dashboard Streamlit
│   └── results/                  # Résultats et logs
│
├── livrable2/                    # Safe Reinforcement Learning
│   ├── cbf_shield.py             # Control Barrier Functions
│   ├── lagrangian_safety.py      # Multiplicateurs Lagrangiens
│   ├── safe_mappo_agent.py       # Agent Safe RL
│   ├── train_safe_mappo.py       # Entraînement Safe RL
│   ├── run_safe_comparison.py    # Comparaison des méthodes
│   ├── dashboard_livrable2.py    # Dashboard Streamlit
│   └── results/                  # Résultats et logs
│
├── livrable3/                    # MARL Neurosymbolique Avancé
│   ├── knowledge_base.py         # Base connaissances (7 règles)
│   ├── neurosymbolic_shield.py   # Shield neurosymbolique
│   ├── explainability.py         # Module d'explicabilité
│   ├── mappo_ns_agent.py         # Agent MAPPO-NS avancé
│   ├── train_mappo_ns.py         # Entraînement MAPPO-NS
│   ├── run_neurosymbolic.py      # Script principal
│   ├── test_robustness_complete.py # Tests robustesse
│   ├── dashboard_livrable3.py    # Dashboard Streamlit
│   └── results/                  # Résultats et explications
│
└── README.md                     # Ce fichier
```

---

## Shield Neurosymbolique - Règles de Sécurité

| ID | Règle | Priorité | Condition | Action |
|----|-------|----------|-----------|--------|
| R1 | Température Critique | 100 | T ≥ 850°C | STOP forcé (4) |
| R2 | Maintenance Requise | 90 | maintenance = True | STOP forcé (4) |
| R3 | Température Élevée | 80 | T > 800°C | Interdit increase (2) |
| R4 | Pression Élevée | 75 | P > 9.0 bar | Interdit increase (2) |
| R5 | Température Haute | 60 | 750 < T ≤ 800°C | maintain (1) |
| R6 | Pression Haute | 55 | 8.5 < P ≤ 9.0 bar | maintain (1) |
| R7 | Conditions Optimales | 10 | T < 700°C, P < 8 bar | Toutes actions |

---

## Installation

### 1. Cloner le dépôt
```
git clone https://github.com/safaa241/MARL-SafeRL-Neurosymbolic.git
cd MARL-SafeRL-Neurosymbolic
```

### 2. Créer l'environnement virtuel
```
# Avec conda
conda create -n yucca_adv python=3.10
conda activate yucca_adv

# Ou avec venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```
pip install torch numpy matplotlib plotly streamlit gymnasium pandas
```

---

## Exécution

### Livrable 1 - MARL Neurosymbolique (Base)

```
cd livrable1

# Entraîner MAPPO standard (baseline)
python train_mappo.py

# Entraîner MAPPO-NS (avec shield)
python train_mappo_ns.py

# Comparer MAPPO, MAPPO-NS, QMIX, MADDPG
python compare_all_marllib.py

# Générer rapport et graphiques
python generate_report.py

# Lancer le dashboard
streamlit run dashboard3.py
```

### Livrable 2 - Safe Reinforcement Learning

```
cd livrable2

# Comparaison complète des méthodes Safe RL
python run_safe_comparison.py --full

# Mode rapide (charge résultats existants)
python run_safe_comparison.py --quick

# Méthode spécifique (CBF, lagrangian, adaptive)
python train_safe_mappo.py --method cbf --episodes 50

# Lancer le dashboard
streamlit run dashboard_livrable2.py
```

### Livrable 3 - MARL Neurosymbolique Avancé

```
cd livrable3

# Tests de robustesse (3 scénarios)
python test_robustness_complete.py

# Expérience complète (MAPPO-NS vs MAPPO standard)
python run_neurosymbolic.py --episodes 50

# Mode rapide
python run_neurosymbolic.py --quick

# Entraîner uniquement MAPPO-NS
python train_mappo_ns.py --episodes 50

# Comparer MAPPO vs MAPPO-NS
python train_mappo_ns.py --compare --episodes 50

# Lancer le dashboard
streamlit run dashboard_livrable3.py
```

---

## Dashboard Streamlit

| Vue | Livrable | Description |
|-----|----------|-------------|
| Comparaison Globale | L1, L2, L3 | MAPPO vs Safe RL vs MAPPO-NS |
| Safe RL (CBF) | L2 | Détails Control Barrier Functions |
| Lagrangien | L2 | Multiplicateurs Lagrangiens |
| Pénalités Adaptatives | L2 | Adaptation dynamique |
| MAPPO-NS | L3 | Shield neurosymbolique |
| Explications | L3 | Traçabilité des décisions |
| Tests Robustesse | L3 | Validation 3 scénarios critiques |

---

## Méthodes Implémentées

### Livrable 2 - Safe RL

| Méthode | Principe | Fichier |
|---------|----------|---------|
| Lagrangien | Multiplicateurs λ pour contrainte de coût | `lagrangian_safety.py` |
| CBF | Fonctions barrière de contrôle (h(x) ≥ 0) | `cbf_shield.py` |
| Adaptative | Pénalité dynamique β = f(violation_rate) | `safe_mappo_agent.py` |

### Livrable 3 - MARL Neurosymbolique

| Composant | Rôle | Fichier |
|-----------|------|---------|
| Base connaissances | 7 règles sécurité prioritaires | `knowledge_base.py` |
| Shield symbolique | Filtrage actions temps réel | `neurosymbolic_shield.py` |
| Explicabilité | Génération explications lisibles | `explainability.py` |
| MAPPO-NS | Agent complet avec shield intégré | `mappo_ns_agent.py` |

---

##  Tests de Robustesse (Livrable 3)

Trois scénarios critiques testés :

1. **Valeurs aberrantes de capteurs** : NaN, valeurs hors échelle, températures impossibles
2. **Montée rapide de température** : T > 800°C → interdiction increase, T > 850°C → STOP forcé
3. **Conflit entre règles** : Détection et résolution par priorité

```
python test_robustness_complete.py
```

---

## Résultats Générés

| Fichier | Description |
|---------|-------------|
| `results/livrable1/comparison_results.json` | Comparaison 4 algorithmes |
| `results/livrable2/complete_safe_comparison.json` | Résultats Safe RL |
| `results/livrable3/complete_experiment_results.json` | Expérience complète MAPPO-NS |
| `results/livrable3/explanations.json` | Explications du shield |
| `results/livrable3/comparison_results.json` | MAPPO vs MAPPO-NS |

---

## Dépendances

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.14.0
streamlit>=1.25.0
gymnasium>=0.29.0
pandas>=2.0.0
```

---

## Conclusion

| Objectif | Livrable 2 (Safe RL) | Livrable 3 (MAPPO-NS) |
|----------|----------------------|----------------------|
| Sécurité | 78.5% (amélioration) | **100% (garantie)** |
| Production | 112 pièces | **140 pièces** |
| Explicabilité | Non | **Oui** |
| Garantie formelle | Non | **Oui** |

**Conclusion** : L'approche neurosymbolique (MAPPO-NS) garantit **100% de sécurité** avec explicabilité, surpassant les méthodes Safe RL standard.

---
