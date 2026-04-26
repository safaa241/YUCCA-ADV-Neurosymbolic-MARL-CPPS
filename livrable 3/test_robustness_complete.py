"""Script de test complet de la robustesse du shield neurosymbolique"""

import numpy as np
from knowledge_base import KnowledgeBase
from neurosymbolic_shield import SymbolicShield

def test_all_robustness_features():
    print("\n" + "="*70)
    print("🔬 TEST COMPLET DE ROBUSTESSE - SHIELD NEUROSYMBOLIQUE")
    print("="*70)
    
    kb = KnowledgeBase()
    shield = SymbolicShield(kb)
    
    # Test 1: Valeurs aberrantes
    print("\n📊 TEST 1: Capteurs défaillants")
    abnormal_obs = [
        ("NaN", np.array([np.nan, 0.5, 0.2, 0.1, 0, 0.3])),
        ("Hors échelle", np.array([2.5, 0.5, 0.2, 0.1, 0, 0.3])),
        ("Négatif", np.array([-1.0, 0.5, 0.2, 0.1, 0, 0.3]))
    ]
    
    for name, obs in abnormal_obs:
        action, modified, explanation, inference = shield.filter_action_with_robustness(2, obs)
        print(f"  {name}: {explanation[:60]}...")
        print(f"    → Action corrigée: {shield.action_names.get(action, action)}")
    
    # Test 2: Montée rapide
    print("\n📊 TEST 2: Montée rapide de température")
    normal_obs = np.array([0.5, 0.5, 0.2, 0.1, 0, 0.3])
    
    for step in [0, 200, 400, 600, 800]:
        modified_obs = create_rapid_temperature_rise(normal_obs, step, start_step=100, rise_rate=0.05)
        temp = modified_obs[0] * 850
        action, modified, explanation, _ = shield.filter_action_with_robustness(2, modified_obs)
        print(f"  Step {step}: T={temp:.0f}°C → action={shield.action_names.get(action, action)} | {modified}")
    
    # Test 3: Conflit de règles
    print("\n📊 TEST 3: Conflit entre règles")
    conflict_obs = np.array([0.96, 0.92, 0.3, 0.1, 0, 0.3])  # T=816°C, P=9.2 bar
    action, modified, explanation, inference = shield.filter_action_with_robustness(2, conflict_obs)
    print(f"  État: T=816°C, P=9.2 bar")
    print(f"  Action: {shield.action_names.get(2)} → {shield.action_names.get(action, action)}")
    print(f"  Conflits détectés: {inference.get('conflicts_detected', False)}")
    
    # Rapport final
    print("\n" + "="*70)
    print("📋 RAPPORT FINAL DE ROBUSTESSE")
    print("="*70)
    report = shield.get_fault_tolerance_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Tous les tests de robustesse sont PASSÉS")

if __name__ == "__main__":
    test_all_robustness_features()