# YUCCA-ADV - Livrables 2 & 3

<div align="center">

**Safe Reinforcement Learning & MARL Neurosymbolique pour CPPS Industriels**

</div>

---

## Présentation

Ce repository contient les **Livrables 2 et 3** du projet YUCCA-ADV.

| Livrable | Titre | Description |
|----------|-------|-------------|
| **Livrable 2** | Safe Reinforcement Learning | Intégration de contraintes de sécurité numériques (Lagrangien, CBF, Pénalités adaptatives) |
| **Livrable 3** | MARL Neurosymbolique (MAPPO-NS) | Shield symbolique pour garantie de sécurité à 100% + explicabilité |

---

##  Résultats Clés

| Algorithme | Taux de Sécurité | Violations | Production | Explicabilité |
|------------|-----------------|------------|------------|---------------|
| MAPPO Standard | 0.85% | 74 360 | 3 | Non |
| Safe RL (CBF) | 78.5% | 14 780 | 112 | Non |
| **MAPPO-NS** | **100%** | **0** | **140** | **Oui** |

**Amélioration MAPPO-NS vs MAPPO Standard :**
- Sécurité : **+99.15 points**
- Violations : **-100%**
- Production : **+137 pièces**

---

##  Shield Neurosymbolique - Règles de Sécurité

| ID | Règle | Priorité | Condition | Action |
|----|-------|----------|-----------|--------|
| R1 | Température Critique | 100 | T ≥ 850°C | STOP forcé |
| R2 | Maintenance Requise | 90 | maintenance = True | STOP forcé |
| R3 | Température Élevée | 80 | T > 800°C | Interdit augmentation |
| R4 | Pression Élevée | 75 | P > 9.0 bar | Interdit augmentation |
| R5 | Température Haute | 60 | 750 < T ≤ 800°C | Maintien recommandé |
| R6 | Pression Haute | 55 | 8.5 < P ≤ 9.0 bar | Maintien recommandé |
| R7 | Conditions Optimales | 10 | T < 700°C, P < 8 bar | Toutes actions |

---

##  Installation

```
# Cloner le repository
git clone https://github.com/safaa241/MARL-SafeRL-Neurosymbolic.git
cd MARL-SafeRL-Neurosymbolic
```
```
# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```
```
# Installer les dépendances
pip install torch numpy matplotlib plotly streamlit gymnasium pandas
```

---

## Exécution

### Livrable 2 - Safe RL

```
cd livrable\ 2

# Comparaison complète des méthodes Safe RL
python run_safe_comparison.py --full

# Mode rapide (charge résultats existants)
python run_safe_comparison.py --quick

# Méthode spécifique (CBF, lagrangian, adaptive)
python train_safe_mappo.py --method cbf --episodes 50

# Lancer le dashboard
streamlit run dashboard_livrable2.py
```

### Livrable 3 - MAPPO-NS (Neurosymbolique)

```
cd livrable\ 3

# Expérience complète (MAPPO-NS vs MAPPO standard)
python run_neurosymbolic.py --episodes 50

# Mode rapide
python run_neurosymbolic.py --quick

# Entraîner uniquement MAPPO-NS
python train_mappo_ns.py --episodes 50 --shield

# Comparer MAPPO vs MAPPO-NS
python train_mappo_ns.py --compare --episodes 50

# Lancer le dashboard
streamlit run dashboard_livrable3.py
```

---

## Structure du Repository

```
MARL-SafeRL-Neurosymbolic/
│
├── livrable 2/                    # Safe Reinforcement Learning
│   ├── cbf_shield.py              # Control Barrier Functions
│   ├── lagrangian_safety.py       # Multiplicateurs Lagrangiens
│   ├── safe_mappo_agent.py        # Agent Safe RL
│   ├── train_safe_mappo.py        # Entraînement Safe RL
│   ├── run_safe_comparison.py     # Comparaison des méthodes
│   ├── dashboard_livrable2.py     # Dashboard Streamlit
│   └── results/                   # Résultats et logs
│
├── livrable 3/                    # MARL Neurosymbolique
│   ├── knowledge_base.py          # Base de connaissances (7 règles)
│   ├── neurosymbolic_shield.py    # Shield neurosymbolique
│   ├── explainability.py          # Module d'explicabilité
│   ├── mappo_ns_agent.py          # Agent MAPPO-NS
│   ├── train_mappo_ns.py          # Entraînement MAPPO-NS
│   ├── run_neurosymbolic.py       # Script principal
│   ├── dashboard_livrable3.py     # Dashboard Streamlit
│   └── results/                   # Résultats et explications
│
└── README.md                      # Ce fichier
```

---

## Dashboard Streamlit

| Vue | Livrable | Description |
|-----|----------|-------------|
| Comparaison Globale | L2 & L3 | Comparaison MAPPO vs Safe RL vs MAPPO-NS |
| Safe RL (CBF) | L2 | Détails Control Barrier Functions |
| Lagrangien | L2 | Multiplicateurs Lagrangiens |
| Pénalités Adaptatives | L2 | Adaptation dynamique |
| MAPPO-NS | L3 | Shield neurosymbolique |
| Explications | L3 | Traçabilité des décisions |

---

##  Méthodes Implémentées

### Livrable 2 - Safe RL

| Méthode | Principe | Fichier |
|---------|----------|---------|
| Lagrangien | Multiplicateurs λ pour contrainte de coût | `lagrangian_safety.py` |
| CBF | Control Barrier Functions (h(x) ≥ 0) | `cbf_shield.py` |
| Adaptative | Pénalité dynamique β = f(violation_rate) | `safe_mappo_agent.py` |

### Livrable 3 - MARL Neurosymbolique

| Composant | Rôle | Fichier |
|-----------|------|---------|
| Knowledge Base | 7 règles de sécurité priorisées | `knowledge_base.py` |
| Symbolic Shield | Filtrage des actions en temps réel | `neurosymbolic_shield.py` |
| Explainability | Génération d'explications lisibles | `explainability.py` |
| MAPPO-NS | Agent complet avec shield intégré | `mappo_ns_agent.py` |

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

##  Conclusion

| Objectif | Livrable 2 (Safe RL) | Livrable 3 (MAPPO-NS) |
|----------|---------------------|----------------------|
| Sécurité | 78.5% (amélioration) | **100% (garantie)** |
| Production | 112 pièces | **140 pièces** |
| Explicabilité | Non | **Oui** |
| Garantie formelle | Non | **Oui** |

**Conclusion :** L'approche neurosymbolique (MAPPO-NS) garantit 100% de sécurité avec explicabilité, surpassant les méthodes Safe RL standard.

