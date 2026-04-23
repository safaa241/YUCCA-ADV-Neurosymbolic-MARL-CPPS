"""
LIVRABLE 3 - Shield Neurosymbolique pour CPPS
Filtre les actions des agents avec des règles symboliques

"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path

from knowledge_base import KnowledgeBase, RuleEngine, SafetyLevel, SafetyRule


class SymbolicShield:
    """
    Shield neurosymbolique pour la sécurité garantie
    
    Combine:
    - Apprentissage neuronal (MAPPO) pour la performance
    - Raisonnement symbolique (règles) pour la sécurité
    
    Garantit que toutes les actions exécutées sont sûres
    """
    
    def __init__(self, knowledge_base: KnowledgeBase = None, 
                 record_explanations: bool = True):
        
        self.kb = knowledge_base or KnowledgeBase()
        self.rule_engine = RuleEngine(self.kb)
        self.record_explanations = record_explanations
        
        # Statistiques
        self.blocked_actions = 0      # Actions bloquées (STOP forcé)
        self.corrected_actions = 0    # Actions corrigées
        self.safe_actions = 0         # Actions déjà sûres
        self.total_checks = 0
        
        # Historique des explications
        self.explanations = []
        self.intervention_history = []
        
        # Mapping des actions
        self.action_names = {
            0: "reduce_speed",
            1: "maintain_speed",
            2: "increase_speed",
            3: "idle",
            4: "emergency_stop"
        }
    
    # Méthode principale pour filtrer une action proposée par l'agent en fonction de l'observation courante
    def filter_action(self, action: int, observation: np.ndarray, 
                      agent_id: int = 0) -> Tuple[int, bool, Optional[str], Optional[Dict]]:
        
        self.total_checks += 1
        
        # 1. Convertir l'observation en état compréhensible
        state_dict = self._observation_to_state(observation, agent_id)
        
        # 2. Effectuer l'inférence symbolique
        inference = self.rule_engine.infer(state_dict, action)
        
        # 3. Ajouter les métadonnées
        inference["timestamp"] = datetime.now().isoformat()
        inference["agent_id"] = agent_id
        
        safe_action = inference["safe_action"]
        was_modified = inference["was_modified"]
        explanation = inference["explanation"]
        
        # 4. Mettre à jour les statistiques
        if was_modified:
            if safe_action == 4:  # emergency_stop
                self.blocked_actions += 1
            else:
                self.corrected_actions += 1
        else:
            self.safe_actions += 1
        
        # 5. Enregistrer l'explication
        if was_modified and self.record_explanations:
            explanation_record = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "state": state_dict,
                "original_action": action,
                "original_action_name": self.action_names.get(action, "unknown"),
                "safe_action": safe_action,
                "safe_action_name": self.action_names.get(safe_action, "unknown"),
                "explanation": explanation,
                "triggering_rule": inference["triggering_rule"],
                "active_rules": inference["active_rules"],
                "safety_level": inference["safety_level"]
            }
            self.explanations.append(explanation_record)
            self.intervention_history.append(explanation_record)
        
        return safe_action, was_modified, explanation, inference
    
    def _observation_to_state(self, obs: np.ndarray, agent_id: int) -> Dict:
        """
        Convertit une observation normalisée en état interprétable
        
        Observation format: [temp_norm, pressure_norm, speed_norm, 
                            production_norm, maintenance_norm, time_norm]
        """
        # Dénormalisation
        temperature = obs[0] * 850  # 0-1 → 0-850°C
        pressure = obs[1] * 10       # 0-1 → 0-10 bar
        speed = obs[2] * 10          # 0-1 → 0-10 m/s
        production = obs[3] * 100    # 0-1 → 0-100 pièces
        maintenance_needed = obs[4] > 0.5
        time_step = obs[5] * 500     # 0-1 → 0-500 steps
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'speed': speed,
            'production': production,
            'maintenance_needed': maintenance_needed,
            'time_step': time_step,
            'agent_id': agent_id
        }
    
    def get_safe_actions(self, observation: np.ndarray, agent_id: int = 0) -> List[int]:
        """
        Retourne la liste des actions sûres pour l'état courant
        
        Args:
            observation: Observation normalisée
            agent_id: Identifiant de l'agent
            
        Returns:
            Liste des actions sûres
        """
        state_dict = self._observation_to_state(observation, agent_id)
        active_rules = self.rule_engine.kb.get_all_active_rules(state_dict)
        
        # Commencer avec toutes les actions
        safe_actions = set([0, 1, 2, 3, 4])
        
        for rule in active_rules:
            if rule.rule_type.value == "blocking":
                # Règle bloquante: seulement l'action forcée
                if rule.action is not None:
                    return [rule.action]
            
            elif rule.rule_type.value == "corrective":
                # Règle corrective: supprimer les actions interdites
                if rule.forbidden_actions:
                    safe_actions -= set(rule.forbidden_actions)
        
        # Si aucune action n'est sûre, retourner STOP
        if not safe_actions:
            return [4]
        
        return list(safe_actions)
    
    def get_safety_level(self, observation: np.ndarray, agent_id: int = 0) -> str:
        """Retourne le niveau de sécurité de l'état courant"""
        state_dict = self._observation_to_state(observation, agent_id)
        return self.rule_engine.kb.get_safety_level(state_dict).value
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques complètes du shield"""
        return {
            'total_checks': self.total_checks,
            'safe_actions': self.safe_actions,
            'corrected_actions': self.corrected_actions,
            'blocked_actions': self.blocked_actions,
            'intervention_rate': (self.corrected_actions + self.blocked_actions) / max(1, self.total_checks),
            'safety_rate': self.safe_actions / max(1, self.total_checks),
            'rule_statistics': self.rule_engine.kb.get_rule_statistics(),
            'total_interventions': len(self.intervention_history)
        }
    
    def get_recent_explanations(self, n: int = 10) -> List[Dict]:
        """Retourne les n dernières explications"""
        return self.explanations[-n:] if self.explanations else []
    
    def get_explanations_by_agent(self, agent_id: int) -> List[Dict]:
        """Retourne les explications pour un agent spécifique"""
        return [exp for exp in self.explanations if exp.get('agent_id') == agent_id]
    
    def get_explanations_by_rule(self, rule_name: str) -> List[Dict]:
        """Retourne les explications pour une règle spécifique"""
        return [exp for exp in self.explanations if exp.get('triggering_rule') == rule_name]
    
    def export_explanations(self, filepath: str):
        """Exporte les explications au format JSON"""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "total_explanations": len(self.explanations),
            "explanations": self.explanations,
            "statistics": self.get_stats()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Explications exportées dans {filepath}")
    
    def reset(self):
        """Réinitialise les statistiques du shield"""
        self.blocked_actions = 0
        self.corrected_actions = 0
        self.safe_actions = 0
        self.total_checks = 0
        self.explanations = []
        self.intervention_history = []
        self.rule_engine.kb.reset_statistics()
    
    def get_action_mask(self, observation: np.ndarray, agent_id: int = 0) -> np.ndarray:
        """
        Retourne un masque d'actions binaire (1 = sûre, 0 = dangereuse)
        
        Utile pour l'action masking dans l'agent
        """
        safe_actions = self.get_safe_actions(observation, agent_id)
        mask = np.zeros(5, dtype=np.float32)
        for action in safe_actions:
            mask[action] = 1.0
        return mask


class NeurosymbolicWrapper:
    """
    Wrapper qui ajoute le shield neurosymbolique à n'importe quel agent
    """
    
    def __init__(self, base_agent, shield: SymbolicShield = None):
        self.base_agent = base_agent
        self.shield = shield or SymbolicShield()
        self.name = f"Neurosymbolic_{getattr(base_agent, 'name', 'Agent')}"
        
        # Historique
        self.action_history = []
        self.shield_interventions = []
    
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        """
        Sélectionne une action avec filtrage neurosymbolique
        
        Args:
            observation: Observation normalisée
            explore: Mode exploration (True) ou exploitation (False)
            
        Returns:
            Action filtrée sûre
        """
        # 1. L'agent de base propose une action
        if hasattr(self.base_agent, 'select_action'):
            if explore:
                raw_action, _, _ = self.base_agent.select_action(observation)
            else:
                raw_action, _, _ = self.base_agent.select_action(observation, explore=False)
        else:
            raw_action = self.base_agent.compute_action(observation)
        
        # 2. Le shield filtre l'action
        safe_action, was_modified, explanation, inference = self.shield.filter_action(
            raw_action, observation
        )
        
        # 3. Historique
        self.action_history.append({
            'timestamp': datetime.now().isoformat(),
            'observation': observation.copy(),
            'raw_action': raw_action,
            'safe_action': safe_action,
            'was_modified': was_modified,
            'explanation': explanation
        })
        
        if was_modified:
            self.shield_interventions.append({
                'raw_action': raw_action,
                'safe_action': safe_action,
                'explanation': explanation
            })
        
        return safe_action
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques complètes"""
        return {
            'shield_stats': self.shield.get_stats(),
            'total_actions': len(self.action_history),
            'intervention_count': len(self.shield_interventions),
            'intervention_rate': len(self.shield_interventions) / max(1, len(self.action_history))
        }
    
    def reset_history(self):
        """Réinitialise l'historique"""
        self.action_history = []
        self.shield_interventions = []
        self.shield.reset()