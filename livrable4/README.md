# LIVRABLE 4 - Validation Hardware-in-the-Loop (HIL) & Réduction Sim-to-Real

##  Description

Ce livrable valide la capacité de l'architecture MAPPO-NS à être transférée de la **simulation vers le monde réel**. Il implémente une méthodologie progressive en 4 phases, de la simulation pure au déploiement réel, avec des mécanismes d'adaptation en ligne pour réduire l'écart Sim-to-Real.

##  Objectifs

- Quantifier l'écart de performance entre simulation et réalité
- Implémenter une méthodologie de validation Hardware-in-the-Loop (HIL)
- Développer des mécanismes d'adaptation en ligne
- Garantir le maintien de la sécurité à 100% sur le système réel

##  Structure du Code
livrable4/

├── dashboard_livrable4.py # Dashboard Streamlit

├── domain_randomization.py # Randomisation de domaine

├── hil_environment.py # Environnement HIL

├── online_adapter.py # Adaptation en ligne

└── train_hil.py # Script principal d'entraînement

##  Installation

```
# Activer l'environnement conda
conda activate yucca_adv
```
```
# Dépendances optionnelles pour le hardware réel
pip install pyserial     # Pour communication série (Arduino)
pip install RPi.GPIO     # Pour Raspberry Pi (optionnel)
pip install paho-mqtt    # Pour MQTT (optionnel)
```

## Exécution
1. Validation complète (4 phases)
```
python train_hil.py
```

2. Exécuter une phase spécifique
```
# Phase 1: Simulation pure
python -c "from train_hil import HILTrainer; t = HILTrainer(); t.train_phase1_simulation(50)"
```
```
# Phase 2: Domain Randomization
python -c "from train_hil import HILTrainer; t = HILTrainer(); t.train_phase2_domain_randomization(50)"
```
```
# Phase 3: Hardware-in-the-Loop
python -c "from train_hil import HILTrainer; t = HILTrainer(); t.train_phase3_hil(30)"
```
```
# Phase 4: Déploiement réel
python -c "from train_hil import HILTrainer; t = HILTrainer(); t.train_phase4_real(20)"
```

## 3. Lancer le dashboard
```
streamlit run dashboard_livrable4.py
```

## Résultats Sauvegardés

results/livrable4/

├── hil_validation_results.json   # Résultats complets

├── adaptation_history.json        # Historique des adaptations

├── gap_history.json               # Évolution des gaps

├── models/

│   ├── phase1_simulation_agent_0.pt

│   ├── phase2_domain_rand_agent_0.pt

│   ├── phase3_hil_agent_0.pt

│   └── phase4_real_agent_0.pt

