"""
LIVRABLE 3 - Shield Neurosymbolique pour CPPS
Version complète avec validation des capteurs et gestion des conflits

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path

from knowledge_base import KnowledgeBase, SafetyLevel


class SymbolicShield:
    """
    Shield neurosymbolique avec:
    - Validation des observations aberrantes
    - Détection et résolution de conflits
    - Journalisation des interventions
    - Rapport de tolérance aux pannes
    """
    
    def __init__(self, knowledge_base: KnowledgeBase = None, 
                 record_explanations: bool = True,
                 config_path: Optional[str] = None):
        
        self.kb = knowledge_base or KnowledgeBase(config_path)
        self.record_explanations = record_explanations
        
        # Statistiques
        self.blocked_actions = 0
        self.corrected_actions = 0
        self.safe_actions = 0
        self.total_checks = 0
        self.sensor_error_corrections = 0
        self.conflict_resolutions = 0
        
        # Historique
        self.explanations = []
        self.intervention_history = []
        self.sensor_error_log = []
        self.conflict_resolution_log = []
        
        # Mapping des actions
        self.action_names = {
            0: "reduce_speed",
            1: "maintain_speed",
            2: "increase_speed",
            3: "idle",
            4: "emergency_stop"
        }
        
        # Métadonnées
        self.simulation_mode = True
        self.fault_injection_enabled = False
    
    def export_explanations(self, filepath: str):
        """Exporte les explications vers un fichier JSON"""
        import numpy as np
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fonction récursive de conversion
        def clean_data(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: clean_data(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_data(item) for item in obj]
            elif obj is None:
                return None
            else:
                return obj
    
    # Nettoyer les explications
        cleaned_explanations = []
        for exp in self.explanations[-200:]:
            try:
                cleaned_explanations.append(clean_data(exp))
            except Exception:
                cleaned_explanations.append({"error": "Cannot serialize"})
    
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "total_explanations": len(self.explanations),
            "explanations": cleaned_explanations
        }
    
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
        print(f"✅ Explications sauvegardées dans {filepath}")
        
    def validate_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        """
        Valide et corrige les observations aberrantes
        Problèmes détectés: NaN, valeurs hors [0,1], valeurs physiquement impossibles
        """
        obs_corrected = observation.copy()
        corrected = False
        errors = []
        
        # Valeurs par défaut sûres
        default_values = [0.35, 0.5, 0.2, 0.1, 0.0, 0.5]
        field_names = ["temperature", "pressure", "speed", "production", "maintenance", "time"]
        
        for i, val in enumerate(observation):
            # NaN
            if np.isnan(val):
                obs_corrected[i] = default_values[i] if i < len(default_values) else 0.5
                corrected = True
                errors.append(f"{field_names[i]}: NaN → valeur par défaut")
            
            # Hors [0,1]
            elif val < 0 or val > 1:
                obs_corrected[i] = max(0, min(1, default_values[i] if i < len(default_values) else 0.5))
                corrected = True
                errors.append(f"{field_names[i]}: {val:.2f} hors [0,1] → corrigé")
        
        # Vérification physique (température dénormalisée)
        if len(observation) >= 2:
            temp_sim = obs_corrected[0] * 850
            if temp_sim > 1200:
                obs_corrected[0] = 0.35
                corrected = True
                errors.append(f"température {temp_sim:.0f}°C > 1200°C (impossible)")
        
        error_msg = "; ".join(errors) if errors else None
        
        if corrected:
            self.sensor_error_corrections += 1
            self.sensor_error_log.append({
                "timestamp": datetime.now().isoformat(),
                "original": observation.tolist(),
                "corrected": obs_corrected.tolist(),
                "errors": errors
            })
            print(f"⚠️ [CAPTEUR] {error_msg}")
        
        return obs_corrected, corrected, error_msg
    
    def _observation_to_state(self, obs: np.ndarray, agent_id: int) -> Dict:
        """Convertit une observation normalisée en état interprétable"""
        # Validation des capteurs
        obs, was_corrected, error_msg = self.validate_observation(obs)
        
        # Dénormalisation
        temperature = obs[0] * 850
        pressure = obs[1] * 10
        speed = obs[2] * 10
        production = obs[3] * 100
        maintenance_needed = obs[4] > 0.5
        time_step = obs[5] * 500
        
        # Sécurisation des valeurs
        temperature = max(0, min(1000, temperature))
        pressure = max(0, min(15, pressure))
        speed = max(0, min(12, speed))
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'speed': speed,
            'production': production,
            'maintenance_needed': maintenance_needed,
            'time_step': time_step,
            'agent_id': agent_id,
            'sensor_error_corrected': was_corrected,
            'sensor_error_message': error_msg
        }
    
    def filter_action(self, action: int, observation: np.ndarray, 
                      agent_id: int = 0) -> Tuple[int, bool, Optional[str], Optional[Dict]]:
        """
        Filtre une action avec validation des capteurs
        """
        self.total_checks += 1
        
        # 1. Convertir l'observation (avec validation)
        try:
            state_dict = self._observation_to_state(observation, agent_id)
            sensor_error = state_dict.pop('sensor_error_corrected', False)
            sensor_error_msg = state_dict.pop('sensor_error_message', None)
        except Exception as e:
            # Erreur critique → STOP forcé
            self.blocked_actions += 1
            error_inference = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "error": str(e),
                "critical_fallback": True
            }
            return 4, True, f"🔴 ERREUR CRITIQUE CAPTEUR → STOP", error_inference
        
        # 2. Si erreur capteur en mode simulation, on continue avec état corrigé
        if sensor_error and not self.simulation_mode:
            self.blocked_actions += 1
            return 4, True, f"🔴 CAPTEUR DÉFAILLANT EN MODE REEL → STOP PREVENTIF", {"sensor_error": True}
        
        # 3. Détecter les conflits entre règles
        conflicts = self.kb.detect_conflicts(state_dict)
        if conflicts:
            self.conflict_resolutions += 1
        
        # 4. Obtenir l'action sûre
        safe_action, explanation, triggering_rule = self.kb.get_safe_action(state_dict, action)
        was_modified = (safe_action != action)
        
        # 5. Construction de l'inférence
        inference = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "state": state_dict,
            "proposed_action": action,
            "proposed_action_name": self.action_names.get(action, "unknown"),
            "safe_action": safe_action,
            "safe_action_name": self.action_names.get(safe_action, "unknown"),
            "was_modified": was_modified,
            "explanation": explanation,
            "triggering_rule": triggering_rule.name if triggering_rule else None,
            "sensor_error": sensor_error,
            "sensor_error_msg": sensor_error_msg,
            "conflicts_detected": len(conflicts) > 0,
            "conflict_count": len(conflicts),
            "simulation_mode": self.simulation_mode
        }
        
        # 6. Mise à jour des statistiques
        if was_modified:
            if safe_action == 4:
                self.blocked_actions += 1
                inference["intervention_type"] = "blocked"
            else:
                self.corrected_actions += 1
                inference["intervention_type"] = "corrected"
        else:
            self.safe_actions += 1
            inference["intervention_type"] = "safe"
        
        # 7. Enregistrer l'explication
        if (was_modified or sensor_error or conflicts) and self.record_explanations:
            self.explanations.append(inference)
            if was_modified or sensor_error:
                self.intervention_history.append(inference)
        
        return safe_action, was_modified, explanation, inference
    
    def get_safe_actions(self, observation: np.ndarray, agent_id: int = 0) -> List[int]:
        """Retourne la liste des actions sûres pour l'état courant"""
        state_dict = self._observation_to_state(observation, agent_id)
        
        if state_dict.get('sensor_error_corrected', False) and not self.simulation_mode:
            return [4]
        
        active_rules = self.kb.get_all_active_rules(state_dict)
        safe_actions = set([0, 1, 2, 3, 4])
        
        for rule in active_rules:
            if rule.rule_type.value == "blocking" and rule.action is not None:
                return [rule.action]
            elif rule.rule_type.value == "corrective" and rule.forbidden_actions:
                safe_actions -= set(rule.forbidden_actions)
        
        if not safe_actions:
            return [4]
        
        return list(safe_actions)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques complètes"""
        return {
            'total_checks': self.total_checks,
            'safe_actions': self.safe_actions,
            'corrected_actions': self.corrected_actions,
            'blocked_actions': self.blocked_actions,
            'sensor_error_corrections': self.sensor_error_corrections,
            'conflict_resolutions': self.conflict_resolutions,
            'intervention_rate': (self.corrected_actions + self.blocked_actions) / max(1, self.total_checks),
            'safety_rate': self.safe_actions / max(1, self.total_checks),
            'total_interventions': len(self.intervention_history),
            'simulation_mode': self.simulation_mode
        }
    
    def get_fault_tolerance_report(self) -> dict:
        """Rapport complet sur la tolérance aux pannes"""
        return {
            "total_sensor_errors_detected": self.sensor_error_corrections,
            "total_conflicts_detected": self.conflict_resolutions,
            "fallback_actions_taken": self.blocked_actions,
            "fault_tolerance_rate": (self.sensor_error_corrections + self.conflict_resolutions) / max(1, self.total_checks),
            "is_fault_tolerant": True,
            "default_fallback_action": "emergency_stop (4)",
            "recommendation": "Ajouter un filtre médian et un vote majoritaire pour déploiement réel"
        }
    
    def get_recent_explanations(self, n: int = 10) -> List[Dict]:
        """Retourne les n dernières explications"""
        return self.explanations[-n:] if self.explanations else []
    
    def export_explanations(self, filepath: str):
        """Exporte toutes les explications et logs"""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "total_explanations": len(self.explanations),
            "total_sensor_errors": len(self.sensor_error_log),
            "total_conflicts": len(self.conflict_resolution_log),
            "explanations": self.explanations[-200:],
            "sensor_errors": self.sensor_error_log[-100:],
            "conflict_resolutions": self.conflict_resolution_log[-100:],
            "statistics": self.get_stats(),
            "fault_tolerance": self.get_fault_tolerance_report()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exports sauvegardés dans {filepath}")
    
    def set_simulation_mode(self, enabled: bool = True):
        """Active/désactive le mode simulation (affecte la tolérance aux pannes)"""
        self.simulation_mode = enabled
        print(f"Mode simulation: {'ACTIF' if enabled else 'INACTIF (mode réel)'}")
    
    def reset(self):
        """Réinitialise toutes les statistiques"""
        self.blocked_actions = 0
        self.corrected_actions = 0
        self.safe_actions = 0
        self.total_checks = 0
        self.sensor_error_corrections = 0
        self.conflict_resolutions = 0
        self.explanations = []
        self.intervention_history = []
        self.sensor_error_log = []
        self.conflict_resolution_log = []
        if hasattr(self.kb, 'reset_statistics'):
            self.kb.reset_statistics()