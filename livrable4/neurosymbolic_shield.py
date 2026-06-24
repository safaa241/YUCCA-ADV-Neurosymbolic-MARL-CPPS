"""
Shield Neurosymbolique pour MAPPO-NS
Garantit la sécurité via 7 règles symboliques
"""

import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime
import json


class SymbolicShield:
    """Shield symbolique pour filtrer les actions dangereuses"""
    
    # Constantes des seuils
    TEMP_CRITICAL = 850.0
    TEMP_WARNING = 800.0
    PRESSURE_CRITICAL = 10.0
    PRESSURE_WARNING = 9.0
    
    # Mapping des actions
    ACTIONS = {
        0: "REDUCE_SPEED",
        1: "MAINTAIN_SPEED", 
        2: "INCREASE_SPEED",
        3: "IDLE",
        4: "STOP"
    }
    
    def __init__(self, agent_id: int = 0):
        self.agent_id = agent_id
        self.stats = {
            'total_checks': 0,
            'safe_actions': 0,
            'corrected_actions': 0,
            'blocked_actions': 0,
            'rule_triggers': {f'R{i}': 0 for i in range(1, 8)}
        }
        self.explanations = []
        self.last_valid_values = {}
    
    def filter_action(self, raw_action: int, observation: np.ndarray) -> Tuple[int, bool, str]:
        """
        Filtre l'action selon les 7 règles symboliques
        Retourne: (action_safe, was_modified, explanation)
        """
        self.stats['total_checks'] += 1
        
        # Dénormalisation
        temp = observation[0] * 850.0
        pressure = observation[1] * 10.0
        speed = observation[2] * 10.0
        maintenance_needed = observation[4] > 0.5
        
        action_name = self.ACTIONS.get(raw_action, "UNKNOWN")
        
        # === R1: Température critique ===
        if temp >= self.TEMP_CRITICAL:
            self.stats['rule_triggers']['R1'] += 1
            explanation = f"R1: Température {temp:.0f}°C ≥ {self.TEMP_CRITICAL:.0f}°C → STOP"
            return self._log_correction(4, True, explanation)
        
        # === R3: Pression critique ===
        if pressure >= self.PRESSURE_CRITICAL and raw_action != 4:
            self.stats['rule_triggers']['R3'] += 1
            explanation = f"R3: Pression {pressure:.1f} bar ≥ {self.PRESSURE_CRITICAL:.1f} bar → STOP"
            return self._log_correction(4, True, explanation)
        
        # === R5: Maintenance requise ===
        if maintenance_needed and raw_action != 4:
            self.stats['rule_triggers']['R5'] += 1
            explanation = "R5: Maintenance requise → STOP"
            return self._log_correction(4, True, explanation)
        
        # === R2: Température élevée ===
        if temp > self.TEMP_WARNING and raw_action == 2:
            self.stats['rule_triggers']['R2'] += 1
            explanation = f"R2: Température {temp:.0f}°C > {self.TEMP_WARNING:.0f}°C → augmentation bloquée"
            return self._log_correction(1, True, explanation)
        
        # === R4: Pression élevée ===
        if pressure > self.PRESSURE_WARNING and raw_action == 2:
            self.stats['rule_triggers']['R4'] += 1
            explanation = f"R4: Pression {pressure:.1f} bar > {self.PRESSURE_WARNING:.1f} bar → augmentation bloquée"
            return self._log_correction(1, True, explanation)
        
        # === R6: Arrêt inutile ===
        if speed == 0 and temp < self.TEMP_WARNING and raw_action in [3, 4]:
            self.stats['rule_triggers']['R6'] += 1
            explanation = f"R6: Arrêt inutile (v=0, T={temp:.0f}°C) → reprise production"
            return self._log_correction(2, True, explanation)
        
        # === R7: Gestion conflit (R1 et R3 simultanées) ===
        if temp >= self.TEMP_CRITICAL and pressure >= self.PRESSURE_CRITICAL:
            self.stats['rule_triggers']['R7'] += 1
            explanation = "R7: Conflit T°C/Pression → Sécurité machine prioritaire → STOP"
            return self._log_correction(4, True, explanation)
        
        # Action acceptée
        self.stats['safe_actions'] += 1
        return (raw_action, False, f"Action acceptée: {action_name}")
    
    def _log_correction(self, safe_action: int, modified: bool, explanation: str) -> Tuple[int, bool, str]:
        """Log une correction du shield"""
        if modified:
            self.stats['corrected_actions'] += 1
        self.explanations.append({
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'safe_action': safe_action,
            'explanation': explanation
        })
        return (safe_action, modified, explanation)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du shield"""
        total = self.stats['total_checks']
        if total > 0:
            self.stats['intervention_rate'] = self.stats['corrected_actions'] / total
        return self.stats
    
    def get_recent_explanations(self, n: int = 10) -> List[Dict]:
        """Retourne les n dernières explications"""
        return self.explanations[-n:]
    
    def export_explanations(self, filepath: str):
        """Exporte les explications vers un fichier JSON"""
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_explanations': len(self.explanations),
                'explanations': self.explanations[-1000:]
            }, f, indent=2)


class KnowledgeBase:
    """Base de connaissances symboliques"""
    
    def __init__(self):
        self.rules = {
            'R1': {'condition': 'temp >= 850', 'action': 'STOP', 'priority': 1},
            'R2': {'condition': 'temp > 800 and action == INCREASE', 'action': 'MAINTAIN', 'priority': 2},
            'R3': {'condition': 'pressure >= 10', 'action': 'STOP', 'priority': 1},
            'R4': {'condition': 'pressure > 9.0 and action == INCREASE', 'action': 'MAINTAIN', 'priority': 2},
            'R5': {'condition': 'maintenance_needed', 'action': 'STOP', 'priority': 1},
            'R6': {'condition': 'speed == 0 and temp < 800', 'action': 'INCREASE', 'priority': 3},
            'R7': {'condition': 'R1 and R3', 'action': 'STOP', 'priority': 1}
        }
    
    def get_rule(self, rule_id: str) -> Dict:
        return self.rules.get(rule_id, {})
    
    def get_all_rules(self) -> Dict:
        return self.rules