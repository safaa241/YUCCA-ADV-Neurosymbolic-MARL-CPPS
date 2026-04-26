# Livrable 1 – Algorithme MARL Neurosymbolique (MAPPO-NS)

**Projet** : YUCCA-ADV  
**Auteur** : FEKNI Safaa  
**Encadrant** : YuccaInfo  

---

##  Présentation du Livrable 1

Ce livrable constitue le **cœur scientifique du PFE YUCCA-ADV**. Il implémente et compare un algorithme MARL neurosymbolique (MAPPO-NS) avec trois algorithmes standards (MAPPO, QMIX, MADDPG) sur un environnement CPPS (Cyber-Physical Production System) simulé.

**Objectif principal** : Démontrer que l’approche neurosymbolique garantit la sûreté des décisions industrielles (100% de sécurité) contrairement au MARL standard (< 1% de sécurité).

---

##  Structure du projet

```
livrable1/
│
├── algo_NS_MARL.py              # MAPPO-NS complet (shield + explicabilité)
├── cpps_environment.py          # Environnement CPPS (3 agents, physique)
├── mappo_agent.py               # Agent MAPPO standard
├── symbolic_shield.py           # Shield neurosymbolique (base connaissances)
├── train_mappo.py               # Entraînement MAPPO standard (Partie 1)
├── train_mappo_ns.py            # Entraînement MAPPO-NS (Partie 3)
├── compare_all_marllib.py       # Comparaison MAPPO-NS vs MAPPO vs QMIX vs MADDPG
├── generate_report.py           # Génération des graphiques et du rapport
├── dashboard3.py                # Dashboard interactif Streamlit
├── requirements.txt             # Dépendances Python
│
└── results/
    ├── part1/
    │   └── metrics.json         # Résultats MAPPO standard
    ├── part3/
    │   └── metrics_ns.json      # Résultats MAPPO-NS
    ├── livrable1/
    │   ├── comparison_results.json
    │   ├── comparison_graphs.png
    │   └── summary_comparison.png
    └── logs/
        ├── PART1_MAPPO_*.log
        └── PART3_MAPPO_NS_*.log
```

---

##  Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/yucca-pfe/yucca-adv.git
cd yucca-adv/livrable1
```

### 2. Créer l’environnement conda

```bash
conda create -n yucca python=3.10
conda activate yucca
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Exécution

### 1. Entraîner MAPPO standard (Partie 1)

```bash
python train_mappo.py
```

**Résultat attendu** : `results/part1/metrics.json`  
**Taux de sécurité** : < 1% (74 360 violations sur 50 épisodes)

### 2. Entraîner MAPPO-NS (Partie 3)

```bash
python train_mappo_ns.py
```

**Résultat attendu** : `results/part3/metrics_ns.json`  
**Taux de sécurité** : 100% (0 violation sur 50 épisodes, 13 806 actions corrigées)

### 3. Lancer la comparaison complète (4 algorithmes)

```bash
python compare_all_marllib.py
```

**Résultat attendu** : `results/livrable1/comparison_results.json`

### 4. Générer les graphiques et le rapport

```bash
python generate_report.py
```

### 5. Lancer le dashboard interactif

```bash
streamlit run dashboard3.py
```

**Accès** : http://localhost:8501

---

## Résultats clés

| Algorithme | Taux de sécurité | Violations totales | Reward moyen | Production |
|------------|------------------|--------------------|--------------|------------|
| **MAPPO** (standard) | 0.85% | 74 360 | -37 088 | 3 |
| **QMIX** | 0.60% | 74 800 | -37 450 | 0 |
| **MADDPG** | 0.50% | 74 900 | -37 480 | 0 |
| **MAPPO-NS** (notre algo) | **100%** ✅ | **0** ✅ | **+9 227** ✅ | **140** ✅ |

**Amélioration MAPPO-NS vs MAPPO :**
- 🔒 Sécurité : +99.15%
- ⚠️ Violations : -100%
- 🏭 Production : +137 pièces
- 📈 Reward : +46 315 points

---

## 🛡️ Shield Neurosymbolique – Règles de sécurité

| ID | Règle | Condition | Action | Priorité |
|----|-------|-----------|--------|----------|
| R1 | Température critique | temp ≥ 850°C | STOP (4) | 100 |
| R2 | Maintenance requise | maintenance = True | STOP (4) | 90 |
| R3 | Température élevée | temp > 800°C | Interdit +2 | 80 |
| R4 | Pression élevée | pressure > 9.0 bar | Interdit +2 | 75 |
| R5 | Température haute | 750 < temp ≤ 800 | maintain (1) | 60 |
| R6 | Pression haute | 8.5 < press ≤ 9.0 | maintain (1) | 55 |
| R7 | Conditions optimales | temp < 700, press < 8 | Toutes actions | 10 |

---

##  Visualisation des résultats

Le dashboard Streamlit (`dashboard3.py`) permet de visualiser :

- Évolution des récompenses par épisode
- Taux de sécurité (MAPPO-NS à 100% vs MAPPO < 1%)
- Nombre de violations (échelle logarithmique)
- Production totale par algorithme
- Statistiques détaillées du shield (13 806 actions corrigées)

---

##  Résumé scientifique

**Question de recherche** : *L’algorithme MAPPO respecte-t-il la notion de sûreté du système ?*

**Réponse** :

| Algorithme | Respect de la sûreté |
|------------|---------------------|
| MAPPO standard | ❌ NON (0.85% seulement) |
| QMIX | ❌ NON (0.60%) |
| MADDPG | ❌ NON (0.50%) |
| **MAPPO-NS** | ✅ **OUI (100%)** |

**Conclusion** : Le MARL standard est **inutilisable** dans un contexte industriel critique. L’approche neurosymbolique (MAPPO-NS) garantit formellement la sécurité des décisions grâce à un shield basé sur des règles symboliques.

---
