# YUCCA-ADV - Systèmes Multi-Agents Neurosymboliques pour l'Industrie 4.0

---

## Description du Projet

YUCCA-ADV est une plateforme de recherche dédiée à l'optimisation des Systèmes Cyber-Physiques de Production (CPPS) dans le contexte de l'Industrie 4.0.

Le projet combine :

* **Multi-Agent Reinforcement Learning (MARL)** : MAPPO, QMIX, MADDPG
* **Safe Reinforcement Learning (Safe RL)** : Control Barrier Functions (CBF), Multiplicateurs Lagrangiens, Pénalités Adaptatives
* **IA Neurosymbolique** : Shield symbolique garantissant la sécurité des décisions
* **Explicabilité** : Génération d'explications compréhensibles pour chaque action
* **Hardware-in-the-Loop (HIL)** : Validation progressive du transfert Simulation → Réalité
* **Réduction du Sim-to-Real Gap** : Domain Randomization et adaptation en ligne

L'objectif principal est de garantir une production optimale tout en assurant une sécurité formelle des systèmes industriels autonomes.

---

# Résultats Clés

| Algorithme                    | Taux de sécurité | Violations | Production | Explicabilité |
| ----------------------------- | ---------------- | ---------- | ---------- | ------------- |
| MAPPO Standard                | 0.85%            | 74 360     | 3          | Non           |
| Safe RL (CBF)                 | 78.5%            | 14 780     | 112        | Non           |
| QMIX                          | 0.60%            | 74 800     | 0          | Non           |
| MADDPG                        | 0.50%            | 74 900     | 0          | Non           |
| **MAPPO-NS (Notre approche)** | **100%**         | **0**      | **140**    | **Oui**       |

### Gains de MAPPO-NS par rapport à MAPPO Standard

* Sécurité : **+99.15 points**
* Violations : **−100 %**
* Production : **+137 pièces**
* Reward : **+46 315 points**

---

# Architecture Générale

Le projet est organisé en quatre livrables successifs :

```
MARL-SafeRL-Neurosymbolic/
│
├── livrable1/     # MARL Neurosymbolique (Base)
├── livrable2/     # Safe Reinforcement Learning
├── livrable3/     # MARL Neurosymbolique Avancé
├── livrable4/     # Validation HIL & Sim-to-Real
│
└── README.md
```

---

# Structure Complète du Projet

```text
MARL-SafeRL-Neurosymbolic/
│
├── livrable1/
│   ├── cpps_environment.py
│   ├── mappo_agent.py
│   ├── symbolic_shield.py
│   ├── train_mappo.py
│   ├── train_mappo_ns.py
│   ├── algo_NS_MARL.py
│   ├── compare_all_marllib.py
│   ├── generate_report.py
│   ├── dashboard3.py
│   └── results/
│
├── livrable2/
│   ├── cbf_shield.py
│   ├── lagrangian_safety.py
│   ├── safe_mappo_agent.py
│   ├── train_safe_mappo.py
│   ├── run_safe_comparison.py
│   ├── dashboard_livrable2.py
│   └── results/
│
├── livrable3/
│   ├── knowledge_base.py
│   ├── neurosymbolic_shield.py
│   ├── explainability.py
│   ├── mappo_ns_agent.py
│   ├── train_mappo_ns.py
│   ├── run_neurosymbolic.py
│   ├── test_robustness_complete.py
│   ├── dashboard_livrable3.py
│   └── results/
│
├── livrable4/
│   ├── dashboard_livrable4.py
│   ├── domain_randomization.py
│   ├── hil_environment.py
│   ├── online_adapter.py
│   ├── train_hil.py
│   └── results/
│
└── README.md
```

---

# Livrable 1 — MARL Neurosymbolique (Base)

## Objectifs

* Construire un jumeau numérique CPPS.
* Implémenter MAPPO.
* Introduire un premier shield symbolique.
* Comparer plusieurs algorithmes MARL.

## Exécution

```bash
cd livrable1
```

### MAPPO Standard

```bash
python train_mappo.py
```

### MAPPO-NS

```bash
python train_mappo_ns.py
```

### Comparaison globale

```bash
python compare_all_marllib.py
```

### Génération des rapports

```bash
python generate_report.py
```

### Dashboard

```bash
streamlit run dashboard3.py
```

---

# Livrable 2 — Safe Reinforcement Learning

## Méthodes Implémentées

| Méthode    | Principe                                        |
| ---------- | ----------------------------------------------- |
| Lagrangien | Multiplicateurs λ pour contraintes de coût      |
| CBF        | Control Barrier Functions                       |
| Adaptative | Pénalités dynamiques selon le taux de violation |

## Exécution

```bash
cd livrable2
```

### Comparaison complète

```bash
python run_safe_comparison.py --full
```

### Mode rapide

```bash
python run_safe_comparison.py --quick
```

### Entraînement spécifique

```bash
python train_safe_mappo.py --method cbf --episodes 50
```

### Dashboard

```bash
streamlit run dashboard_livrable2.py
```

---

# Livrable 3 — MARL Neurosymbolique Avancé

## Architecture Neurosymbolique

| Composant              | Rôle                         |
| ---------------------- | ---------------------------- |
| Base de connaissances  | Règles industrielles         |
| Shield neurosymbolique | Filtrage temps réel          |
| Explicabilité          | Justification des décisions  |
| MAPPO-NS               | Apprentissage + raisonnement |

---

## Règles de Sécurité du Shield

| ID | Règle                | Priorité | Condition          | Action             |
| -- | -------------------- | -------- | ------------------ | ------------------ |
| R1 | Température critique | 100      | T ≥ 850°C          | STOP               |
| R2 | Maintenance requise  | 90       | Maintenance=True   | STOP               |
| R3 | Température élevée   | 80       | T > 800°C          | Interdire Increase |
| R4 | Pression élevée      | 75       | P > 9.0 bar        | Interdire Increase |
| R5 | Température haute    | 60       | 750 < T ≤ 800°C    | Maintain           |
| R6 | Pression haute       | 55       | 8.5 < P ≤ 9.0 bar  | Maintain           |
| R7 | Conditions optimales | 10       | T < 700°C et P < 8 | Toutes actions     |

---

## Tests de Robustesse

Trois scénarios critiques :

### 1. Valeurs aberrantes

* NaN
* Capteurs défaillants
* Températures impossibles

### 2. Montée rapide de température

* T > 800°C → blocage des augmentations
* T > 850°C → arrêt immédiat

### 3. Conflits de règles

* Détection automatique
* Résolution par priorité

---

## Exécution

```bash
cd livrable3
```

### Expérience complète

```bash
python run_neurosymbolic.py --episodes 50
```

### Mode rapide

```bash
python run_neurosymbolic.py --quick
```

### MAPPO-NS uniquement

```bash
python train_mappo_ns.py --episodes 50
```

### Comparaison MAPPO vs MAPPO-NS

```bash
python train_mappo_ns.py --compare --episodes 50
```

### Tests de robustesse

```bash
python test_robustness_complete.py
```

### Dashboard

```bash
streamlit run dashboard_livrable3.py
```

---

# Livrable 4 — Validation Hardware-in-the-Loop (HIL) & Réduction Sim-to-Real

## Description

Ce livrable valide le transfert du modèle MAPPO-NS depuis la simulation vers un système physique.

L'approche suit une méthodologie progressive en quatre phases :

### Phase 1 — Simulation Pure

Entraînement dans le jumeau numérique.

### Phase 2 — Domain Randomization

Variation aléatoire :

* Température
* Pression
* Bruit capteur
* Paramètres physiques

afin d'améliorer la robustesse.

### Phase 3 — Hardware-in-the-Loop

Interaction avec une plateforme matérielle :

* Arduino
* Raspberry Pi
* Automate industriel

tout en conservant le shield neurosymbolique.

### Phase 4 — Déploiement Réel

Validation sur système réel avec :

* Adaptation en ligne
* Monitoring continu
* Réduction du Sim-to-Real Gap

---

## Objectifs

* Quantifier l'écart Simulation → Réalité
* Valider MAPPO-NS en Hardware-in-the-Loop
* Implémenter l'adaptation en ligne
* Maintenir une sécurité de 100 %
* Réduire le Sim-to-Real Gap

---

## Exécution

```bash
cd livrable4
```

### Validation complète

```bash
python train_hil.py
```

### Phase 1

```bash
python -c "from train_hil import HILTrainer; t=HILTrainer(); t.train_phase1_simulation(50)"
```

### Phase 2

```bash
python -c "from train_hil import HILTrainer; t=HILTrainer(); t.train_phase2_domain_randomization(50)"
```

### Phase 3

```bash
python -c "from train_hil import HILTrainer; t=HILTrainer(); t.train_phase3_hil(30)"
```

### Phase 4

```bash
python -c "from train_hil import HILTrainer; t=HILTrainer(); t.train_phase4_real(20)"
```

### Dashboard

```bash
streamlit run dashboard_livrable4.py
```

---

# Dashboard Streamlit

| Dashboard     | Fonction                        |
| ------------- | ------------------------------- |
| Dashboard L1  | Comparaison MARL                |
| Dashboard L2  | Safe RL                         |
| Dashboard L3  | MAPPO-NS                        |
| Dashboard L4  | Sim-to-Real & HIL               |
| Explicabilité | Justification des décisions     |
| Robustesse    | Analyse des scénarios critiques |

---

# Résultats Générés

## Livrable 1

```text
results/livrable1/comparison_results.json
```

## Livrable 2

```text
results/livrable2/complete_safe_comparison.json
```

## Livrable 3

```text
results/livrable3/complete_experiment_results.json
results/livrable3/explanations.json
results/livrable3/comparison_results.json
```

## Livrable 4

```text
results/livrable4/
│
├── hil_validation_results.json
├── adaptation_history.json
├── gap_history.json
│
└── models/
    ├── phase1_simulation_agent_0.pt
    ├── phase2_domain_rand_agent_0.pt
    ├── phase3_hil_agent_0.pt
    └── phase4_real_agent_0.pt
```

---

# Installation

## Clonage

```bash
git clone https://github.com/safaa241/MARL-SafeRL-Neurosymbolic.git
cd MARL-SafeRL-Neurosymbolic
```

## Environnement Virtuel

### Conda

```bash
conda create -n yucca_adv python=3.10
conda activate yucca_adv
```

### Venv

```bash
python -m venv .venv
```

Windows :

```bash
.venv\Scripts\activate
```

Linux / Mac :

```bash
source .venv/bin/activate
```

---

## Installation des Dépendances

```bash
pip install torch numpy matplotlib plotly streamlit gymnasium pandas
```

### Dépendances HIL Optionnelles

```bash
pip install pyserial
pip install RPi.GPIO
pip install paho-mqtt
```

---

# Dépendances

```text
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.14.0
streamlit>=1.25.0
gymnasium>=0.29.0
pandas>=2.0.0

# HIL
pyserial>=3.5
paho-mqtt>=1.6
RPi.GPIO>=0.7
```

---

# Contributions Scientifiques

1. Intégration de MARL et Safe RL dans un CPPS industriel.
2. Conception d'un shield neurosymbolique garantissant une sécurité formelle.
3. Génération d'explications interprétables pour chaque décision.
4. Validation expérimentale sur plusieurs algorithmes MARL.
5. Validation Hardware-in-the-Loop pour le transfert vers le monde réel.
6. Réduction du Sim-to-Real Gap par Domain Randomization et adaptation en ligne.

---

# Conclusion

| Critère           | Safe RL    | MAPPO-NS | MAPPO-NS + HIL |
| ----------------- | ---------- | -------- | -------------- |
| Sécurité          | 78.5%      | 100%     | 100%           |
| Violations        | 14 780     | 0        | 0              |
| Production        | 112        | 140      | 140            |
| Explicabilité     | Non        | Oui      | Oui            |
| Garantie formelle | Non        | Oui      | Oui            |
| Sim-to-Real       | Non traité | Partiel  | Oui            |

## Résultat Final

L'architecture **MAPPO-NS** combine les avantages du Multi-Agent Reinforcement Learning, du Safe Reinforcement Learning et de l'IA Neurosymbolique pour fournir :

* **100 % de sécurité**
* **0 violation**
* **140 pièces produites**
* **Explicabilité complète**
* **Validation Hardware-in-the-Loop**
* **Réduction du Sim-to-Real Gap**

Cette approche constitue une solution robuste et fiable pour les futurs systèmes industriels autonomes de l'Industrie 4.0.
