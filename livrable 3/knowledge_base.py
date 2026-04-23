"""
LIVRABLE 3 - Base de connaissances symboliques pour CPPS
Contient les règles métier et normes de sécurité industrielles

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime


class RuleType(Enum):
    """Type de règle symbolique"""
    BLOCKING = "blocking"      # Bloque complètement l'action (STOP forcé)
    CORRECTIVE = "corrective"  # Corrige l'action vers une action sûre
    INFORMATIVE = "informative" # Information seulement, ne modifie pas l'action


class SafetyLevel(Enum):
    """Niveau de sécurité"""
    CRITICAL = "critical"   # Danger immédiat - nécessite STOP
    HIGH = "high"           # Risque élevé - action corrective nécessaire
    MEDIUM = "medium"       # Risque modéré - surveillance recommandée
    LOW = "low"             # Risque faible
    SAFE = "safe"           # Sécurisé - opération normale


@dataclass
class SafetyRule:
    """Structure d'une règle de sécurité"""
    name: str
    condition: Callable[[Dict], bool]
    priority: int
    rule_type: RuleType
    message: str
    description: str = ""
    action: Optional[int] = None
    forbidden_actions: Optional[List[int]] = None
    safe_action: Optional[int] = None
    allowed_actions: Optional[List[int]] = None
    safety_level: SafetyLevel = SafetyLevel.SAFE
    category: str = "general"  # temperature, pressure, maintenance, production
    
    def evaluate(self, state: Dict) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Évalue la règle sur un état donné
        
        Returns:
            (déclenchée, action_forcée, message)
        """
        if not self.condition(state):
            return False, None, None
        
        if self.rule_type == RuleType.BLOCKING and self.action is not None:
            return True, self.action, self.message
        
        return True, None, self.message


class KnowledgeBase:
    """
    Base de connaissances symboliques pour l'industrie 4.0
    
    Contient les règles métier et normes de sécurité:
    - Normes ISO 13849 (sécurité des machines)
    - Règles spécifiques CPPS
    - Contraintes opérationnelles
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.rules = self._initialize_rules()
        self.rule_statistics = {rule.name: 0 for rule in self.rules}
        self.total_evaluations = 0
        self.last_triggered_rule = None
        
    def _initialize_rules(self) -> List[SafetyRule]:
        """Initialise les règles de sécurité"""
        
        rules = []
        
        # ========== RÈGLES DE NIVEAU CRITIQUE (Priority 100-90) ==========
        
        # R1: Température critique
        rules.append(SafetyRule(
            name="temperature_critical",
            condition=lambda s: s.get('temperature', 0) >= 850,
            priority=100,
            rule_type=RuleType.BLOCKING,
            action=4,  # emergency_stop
            message="🔴 TEMPÉRATURE CRITIQUE > 850°C → ARRÊT D'URGENCE",
            description="La température a dépassé la limite critique de sécurité (850°C). "
                       "Un arrêt d'urgence immédiat est requis pour éviter tout dommage.",
            safety_level=SafetyLevel.CRITICAL,
            category="temperature"
        ))
        
        # R2: Pression critique
        rules.append(SafetyRule(
            name="pressure_critical",
            condition=lambda s: s.get('pressure', 0) >= 10,
            priority=95,
            rule_type=RuleType.BLOCKING,
            action=4,
            message="🔴 PRESSION CRITIQUE > 10 bar → ARRÊT D'URGENCE",
            description="La pression a dépassé la limite critique de sécurité (10 bar).",
            safety_level=SafetyLevel.CRITICAL,
            category="pressure"
        ))
        
        # R3: Maintenance requise
        rules.append(SafetyRule(
            name="maintenance_required",
            condition=lambda s: s.get('maintenance_needed', False),
            priority=90,
            rule_type=RuleType.BLOCKING,
            action=4,
            message="🔧 MAINTENANCE REQUISE → ARRÊT OBLIGATOIRE",
            description="La machine nécessite une maintenance. L'opération ne peut pas continuer.",
            safety_level=SafetyLevel.CRITICAL,
            category="maintenance"
        ))
        
        # ========== RÈGLES DE NIVEAU ÉLEVÉ (Priority 80-70) ==========
        
        # R4: Température élevée
        rules.append(SafetyRule(
            name="temperature_high",
            condition=lambda s: s.get('temperature', 0) > 800,
            priority=80,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],  # increase_speed
            safe_action=0,  # reduce_speed
            message="⚠️ Température > 800°C → Augmentation vitesse interdite",
            description="La température est élevée (>800°C). L'augmentation de vitesse "
                       "pourrait entraîner une surchauffe critique.",
            safety_level=SafetyLevel.HIGH,
            category="temperature"
        ))
        
        # R5: Pression élevée
        rules.append(SafetyRule(
            name="pressure_high",
            condition=lambda s: s.get('pressure', 0) > 9.0,
            priority=75,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=0,
            message="⚠️ Pression > 9.0 bar → Augmentation vitesse interdite",
            description="La pression est élevée (>9.0 bar). L'augmentation de vitesse est risquée.",
            safety_level=SafetyLevel.HIGH,
            category="pressure"
        ))
        
        # ========== RÈGLES DE NIVEAU MOYEN (Priority 60-50) ==========
        
        # R6: Température haute (alerte)
        rules.append(SafetyRule(
            name="temperature_warning",
            condition=lambda s: 750 < s.get('temperature', 0) <= 800,
            priority=60,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=1,  # maintain_speed
            message="⚠️ Température élevée (750-800°C) → Maintien de la vitesse",
            description="La température est dans la zone d'alerte (750-800°C). "
                       "Il est recommandé de ne pas augmenter la vitesse.",
            safety_level=SafetyLevel.MEDIUM,
            category="temperature"
        ))
        
        # R7: Pression haute (alerte)
        rules.append(SafetyRule(
            name="pressure_warning",
            condition=lambda s: 8.5 < s.get('pressure', 0) <= 9.0,
            priority=55,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=1,
            message="⚠️ Pression élevée (8.5-9.0 bar) → Maintien de la vitesse",
            description="La pression est dans la zone d'alerte (8.5-9.0 bar).",
            safety_level=SafetyLevel.MEDIUM,
            category="pressure"
        ))
        
        # R8: Limite de vitesse
        rules.append(SafetyRule(
            name="speed_limit",
            condition=lambda s: s.get('speed', 0) >= 9.5,
            priority=50,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=1,
            message="⚠️ Vitesse proche limite → Augmentation interdite",
            description="La vitesse est proche de la limite maximale (10 m/s).",
            safety_level=SafetyLevel.MEDIUM,
            category="speed"
        ))
        
        # ========== RÈGLES DE NIVEAU BAS (Priority 40-20) ==========
        
        # R9: Température optimale
        rules.append(SafetyRule(
            name="temperature_optimal",
            condition=lambda s: s.get('temperature', 0) < 700,
            priority=30,
            rule_type=RuleType.INFORMATIVE,
            message="✅ Température optimale (<700°C)",
            description="La température est dans la zone optimale pour la production.",
            safety_level=SafetyLevel.SAFE,
            category="temperature"
        ))
        
        # R10: Pression optimale
        rules.append(SafetyRule(
            name="pressure_optimal",
            condition=lambda s: s.get('pressure', 0) < 8,
            priority=25,
            rule_type=RuleType.INFORMATIVE,
            message="✅ Pression optimale (<8 bar)",
            description="La pression est dans la zone optimale.",
            safety_level=SafetyLevel.SAFE,
            category="pressure"
        ))
        
        # ========== RÈGLES DE NIVEAU INFORMATION (Priority <20) ==========
        
        # R11: Conditions optimales complètes
        rules.append(SafetyRule(
            name="optimal_conditions",
            condition=lambda s: (s.get('temperature', 0) < 700 and 
                                s.get('pressure', 0) < 8),
            priority=10,
            rule_type=RuleType.INFORMATIVE,
            allowed_actions=[0, 1, 2, 3],
            message="✅ Conditions optimales → Toutes actions permises",
            description="Les conditions sont optimales pour la production. "
                       "Toutes les actions sont autorisées.",
            safety_level=SafetyLevel.SAFE,
            category="general"
        ))
        
        # R12: Boost de production possible
        rules.append(SafetyRule(
            name="production_boost",
            condition=lambda s: (s.get('temperature', 0) < 600 and 
                                s.get('pressure', 0) < 7 and
                                s.get('speed', 0) < 8),
            priority=5,
            rule_type=RuleType.INFORMATIVE,
            message="📈 Conditions excellentes → Production optimale possible",
            description="Les conditions sont excellentes. La production peut être maximisée.",
            safety_level=SafetyLevel.SAFE,
            category="production"
        ))
        
        return rules
    
    def get_safe_action(self, state_dict: Dict, proposed_action: int) -> Tuple[int, Optional[str], Optional[SafetyRule]]:
        """
        Retourne une action sûre basée sur l'état et les règles
        
        Args:
            state_dict: État du système (température, pression, vitesse, etc.)
            proposed_action: Action proposée par l'agent (0-4)
            
        Returns:
            (action_sûre, explication, règle_déclenchée)
        """
        self.total_evaluations += 1
        
        # Trier les règles par priorité décroissante
        sorted_rules = sorted(self.rules, key=lambda x: x.priority, reverse=True)
        
        for rule in sorted_rules:
            try:
                if rule.condition(state_dict):
                    # Incrémenter le compteur de la règle
                    self.rule_statistics[rule.name] = self.rule_statistics.get(rule.name, 0) + 1
                    self.last_triggered_rule = rule
                    
                    # Règle avec action forcée (BLOCKING)
                    if rule.rule_type == RuleType.BLOCKING and rule.action is not None:
                        return rule.action, rule.message, rule
                    
                    # Règle avec actions interdites (CORRECTIVE)
                    if rule.rule_type == RuleType.CORRECTIVE:
                        if rule.forbidden_actions and proposed_action in rule.forbidden_actions:
                            if rule.safe_action is not None:
                                return rule.safe_action, rule.message, rule
                            else:
                                # Action par défaut: maintenir
                                return 1, rule.message, rule
                    
                    # Règle informative - ne modifie pas l'action
                    if rule.rule_type == RuleType.INFORMATIVE:
                        # Continue à vérifier d'autres règles de priorité supérieure
                        pass
            
            except Exception as e:
                print(f"Erreur lors de l'évaluation de la règle {rule.name}: {e}")
                continue
        
        # Aucune règle déclenchée
        return proposed_action, None, None
    
    def get_all_active_rules(self, state_dict: Dict) -> List[SafetyRule]:
        """
        Retourne toutes les règles actives pour un état donné
        """
        active_rules = []
        
        for rule in self.rules:
            try:
                if rule.condition(state_dict):
                    active_rules.append(rule)
            except Exception:
                pass
        
        # Trier par priorité
        active_rules.sort(key=lambda x: x.priority, reverse=True)
        
        return active_rules
    
    def get_highest_priority_rule(self, state_dict: Dict) -> Optional[SafetyRule]:
        """
        Retourne la règle avec la plus haute priorité active
        """
        active_rules = self.get_all_active_rules(state_dict)
        return active_rules[0] if active_rules else None
    
    def get_safety_level(self, state_dict: Dict) -> SafetyLevel:
        """
        Retourne le niveau de sécurité global de l'état
        """
        active_rules = self.get_all_active_rules(state_dict)
        
        if not active_rules:
            return SafetyLevel.SAFE
        
        # Prendre le niveau le plus critique
        levels = [rule.safety_level for rule in active_rules]
        if SafetyLevel.CRITICAL in levels:
            return SafetyLevel.CRITICAL
        elif SafetyLevel.HIGH in levels:
            return SafetyLevel.HIGH
        elif SafetyLevel.MEDIUM in levels:
            return SafetyLevel.MEDIUM
        elif SafetyLevel.LOW in levels:
            return SafetyLevel.LOW
        else:
            return SafetyLevel.SAFE
    
    def get_safety_message(self, state_dict: Dict) -> str:
        """
        Retourne un message de sécurité basé sur l'état
        """
        highest_rule = self.get_highest_priority_rule(state_dict)
        
        if highest_rule:
            return highest_rule.message
        else:
            return "✅ Conditions normales de sécurité"
    
    def is_action_allowed(self, state_dict: Dict, action: int) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si une action est autorisée dans l'état donné
        
        Returns:
            (autorisée, raison)
        """
        safe_action, message, rule = self.get_safe_action(state_dict, action)
        
        if safe_action != action:
            return False, message or f"Action {action} interdite par la règle {rule.name if rule else 'inconnue'}"
        
        return True, None
    
    def get_allowed_actions(self, state_dict: Dict) -> List[int]:
        """
        Retourne la liste de toutes les actions autorisées
        """
        allowed = []
        
        for action in range(5):  # 0-4
            is_allowed, _ = self.is_action_allowed(state_dict, action)
            if is_allowed:
                allowed.append(action)
        
        # Si aucune action n'est autorisée, retourner STOP
        if not allowed:
            return [4]
        
        return allowed
    
    def get_rule_statistics(self) -> Dict:
        """
        Retourne les statistiques d'utilisation des règles
        """
        return {
            "total_evaluations": self.total_evaluations,
            "rules_triggered": sum(self.rule_statistics.values()),
            "rule_breakdown": self.rule_statistics.copy(),
            "most_triggered": max(self.rule_statistics.items(), key=lambda x: x[1])[0] if self.rule_statistics else None,
            "most_triggered_count": max(self.rule_statistics.values()) if self.rule_statistics else 0
        }
    
    def reset_statistics(self):
        """Réinitialise les statistiques"""
        self.rule_statistics = {rule.name: 0 for rule in self.rules}
        self.total_evaluations = 0
        self.last_triggered_rule = None
    
    def export_knowledge(self, filepath: str):
        """
        Exporte la base de connaissances au format JSON
        """
        export_data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "total_rules": len(self.rules),
            "rules": [
                {
                    "name": rule.name,
                    "priority": rule.priority,
                    "type": rule.rule_type.value,
                    "message": rule.message,
                    "description": rule.description,
                    "safety_level": rule.safety_level.value,
                    "category": rule.category
                }
                for rule in self.rules
            ],
            "statistics": self.get_rule_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Base de connaissances exportée dans {filepath}")
    
    def add_custom_rule(self, rule: SafetyRule):
        """
        Ajoute une règle personnalisée à la base de connaissances
        """
        self.rules.append(rule)
        self.rule_statistics[rule.name] = 0
        # Retrier les règles par priorité
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        print(f"✅ Règle ajoutée: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """
        Supprime une règle de la base de connaissances
        """
        self.rules = [r for r in self.rules if r.name != rule_name]
        if rule_name in self.rule_statistics:
            del self.rule_statistics[rule_name]
        print(f"✅ Règle supprimée: {rule_name}")
    
    def get_rules_by_category(self, category: str) -> List[SafetyRule]:
        """
        Retourne les règles d'une catégorie spécifique
        """
        return [r for r in self.rules if r.category == category]
    
    def get_rules_by_safety_level(self, level: SafetyLevel) -> List[SafetyRule]:
        """
        Retourne les règles d'un niveau de sécurité spécifique
        """
        return [r for r in self.rules if r.safety_level == level]
    
    def __len__(self) -> int:
        return len(self.rules)
    
    def __repr__(self) -> str:
        return f"KnowledgeBase(rules={len(self.rules)}, evaluations={self.total_evaluations})"


class RuleEngine:
    """
    Moteur d'inférence pour les règles symboliques
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.inference_history = []
        self.max_history_size = 1000
    
    def infer(self, state_dict: Dict, proposed_action: int) -> Dict:
        """
        Effectue une inférence complète
        
        Returns:
            Dictionnaire avec action, explications, règles actives
        """
        # Obtenir toutes les règles actives
        active_rules = self.kb.get_all_active_rules(state_dict)
        
        # Déterminer l'action sûre
        safe_action, explanation, triggering_rule = self.kb.get_safe_action(state_dict, proposed_action)
        
        # Niveau de sécurité
        safety_level = self.kb.get_safety_level(state_dict)
        
        # Enregistrer l'inférence
        inference = {
            "timestamp": datetime.now().isoformat(),
            "state": {
                "temperature": state_dict.get('temperature', 0),
                "pressure": state_dict.get('pressure', 0),
                "speed": state_dict.get('speed', 0)
            },
            "proposed_action": proposed_action,
            "safe_action": safe_action,
            "was_modified": safe_action != proposed_action,
            "active_rules": [rule.name for rule in active_rules],
            "triggering_rule": triggering_rule.name if triggering_rule else None,
            "explanation": explanation,
            "safety_level": safety_level.value
        }
        
        # Gérer l'historique
        self.inference_history.append(inference)
        if len(self.inference_history) > self.max_history_size:
            self.inference_history = self.inference_history[-self.max_history_size:]
        
        return inference
    
    def get_last_inferences(self, n: int = 10) -> List[Dict]:
        """Retourne les dernières inférences"""
        return self.inference_history[-n:] if self.inference_history else []
    
    def get_inferences_by_rule(self, rule_name: str) -> List[Dict]:
        """Retourne les inférences pour une règle spécifique"""
        return [inf for inf in self.inference_history if inf.get('triggering_rule') == rule_name]
    
    def get_inferences_with_modifications(self) -> List[Dict]:
        """Retourne les inférences où l'action a été modifiée"""
        return [inf for inf in self.inference_history if inf.get('was_modified', False)]
    
    def clear_history(self):
        """Efface l'historique d'inférence"""
        self.inference_history = []
    
    def export_history(self, filepath: str):
        """Exporte l'historique des inférences"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "total_inferences": len(self.inference_history),
            "inferences": self.inference_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Historique exporté dans {filepath}")


# Raccourci pour créer une base de connaissances par défaut
def create_default_knowledge_base() -> KnowledgeBase:
    """Crée une base de connaissances avec les règles par défaut"""
    return KnowledgeBase()