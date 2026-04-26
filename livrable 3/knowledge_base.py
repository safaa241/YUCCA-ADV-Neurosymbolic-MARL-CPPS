"""
LIVRABLE 3 - Base de connaissances symboliques pour CPPS
Version complète avec seuils dynamiques, gestion des conflits et configuration externe

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path


class RuleType(Enum):
    """Type de règle symbolique"""
    BLOCKING = "blocking"
    CORRECTIVE = "corrective"
    INFORMATIVE = "informative"


class SafetyLevel(Enum):
    """Niveau de sécurité"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


@dataclass
class SafetyRule:
    """Structure d'une règle de sécurité avec seuils dynamiques"""
    name: str
    condition_func: Callable[[Dict], bool]
    priority: int
    rule_type: RuleType
    message: str
    description: str = ""
    action: Optional[int] = None
    forbidden_actions: Optional[List[int]] = None
    safe_action: Optional[int] = None
    allowed_actions: Optional[List[int]] = None
    safety_level: SafetyLevel = SafetyLevel.SAFE
    category: str = "general"
    thresholds: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    
    def evaluate(self, state: Dict) -> Tuple[bool, Optional[int], Optional[str]]:
        """Évalue la règle sur un état donné"""
        if not self.is_active:
            return False, None, None
        try:
            if not self.condition_func(state):
                return False, None, None
        except Exception:
            return False, None, None
        
        if self.rule_type == RuleType.BLOCKING and self.action is not None:
            return True, self.action, self.message
        
        return True, None, self.message
    
    def update_threshold(self, param: str, new_value: float):
        """Met à jour un seuil dynamiquement"""
        self.thresholds[param] = new_value


class KnowledgeBase:
    """
    Base de connaissances symboliques avec:
    - Seuils dynamiques paramétrables
    - Détection et résolution de conflits
    - Configuration externe (JSON)
    - Validation des observations
    """
    
    # Définition des contextes de production (constante de classe)
    PRODUCTION_CONTEXTS = {
        "steel": {
            "name": "Acier",
            "temp_critical": 850,
            "temp_high": 800,
            "temp_warning_low": 750,
            "press_critical": 10,
            "press_high": 9.0,
            "press_warning_low": 8.5,
            "temp_optimal": 700,
            "press_optimal": 8.0
        },
        "aluminium": {
            "name": "Aluminium",
            "temp_critical": 650,
            "temp_high": 600,
            "temp_warning_low": 550,
            "press_critical": 8,
            "press_high": 7.0,
            "press_warning_low": 6.5,
            "temp_optimal": 500,
            "press_optimal": 6.0
        },
        "titanium": {
            "name": "Titane",
            "temp_critical": 950,
            "temp_high": 900,
            "temp_warning_low": 850,
            "press_critical": 12,
            "press_high": 11.0,
            "press_warning_low": 10.5,
            "temp_optimal": 800,
            "press_optimal": 9.0
        },
        "plastic": {
            "name": "Plastique",
            "temp_critical": 400,
            "temp_high": 350,
            "temp_warning_low": 300,
            "press_critical": 5,
            "press_high": 4.5,
            "press_warning_low": 4.0,
            "temp_optimal": 250,
            "press_optimal": 3.0
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        self._dynamic_thresholds = {}
        self.rules = self._initialize_rules()
        self.rule_statistics = {rule.name: 0 for rule in self.rules}
        self.total_evaluations = 0
        self.last_triggered_rule = None
        self.conflict_log = []
        
        # Charger la configuration externe si fournie
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def _initialize_rules(self) -> List[SafetyRule]:
        """Initialise les règles avec seuils paramétrables"""
        
        rules = []
        
        # R1: Température critique
        rules.append(SafetyRule(
            name="temperature_critical",
            condition_func=lambda s: s.get('temperature', 0) >= self._get_threshold('temp_critical', 850),
            priority=100,
            rule_type=RuleType.BLOCKING,
            action=4,
            message="🔴 TEMPÉRATURE CRITIQUE > SEUIL → ARRÊT D'URGENCE",
            description="La température a dépassé la limite critique de sécurité.",
            safety_level=SafetyLevel.CRITICAL,
            category="temperature",
            thresholds={"temp_critical": 850}
        ))
        
        # R2: Pression critique
        rules.append(SafetyRule(
            name="pressure_critical",
            condition_func=lambda s: s.get('pressure', 0) >= self._get_threshold('press_critical', 10),
            priority=95,
            rule_type=RuleType.BLOCKING,
            action=4,
            message="🔴 PRESSION CRITIQUE > SEUIL → ARRÊT D'URGENCE",
            safety_level=SafetyLevel.CRITICAL,
            category="pressure",
            thresholds={"press_critical": 10}
        ))
        
        # R3: Maintenance requise
        rules.append(SafetyRule(
            name="maintenance_required",
            condition_func=lambda s: s.get('maintenance_needed', False),
            priority=90,
            rule_type=RuleType.BLOCKING,
            action=4,
            message="🔧 MAINTENANCE REQUISE → ARRÊT OBLIGATOIRE",
            safety_level=SafetyLevel.CRITICAL,
            category="maintenance"
        ))
        
        # R4: Température élevée
        rules.append(SafetyRule(
            name="temperature_high",
            condition_func=lambda s: s.get('temperature', 0) > self._get_threshold('temp_high', 800),
            priority=80,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=0,
            message="⚠️ Température > SEUIL → Augmentation vitesse interdite",
            safety_level=SafetyLevel.HIGH,
            category="temperature",
            thresholds={"temp_high": 800}
        ))
        
        # R5: Pression élevée
        rules.append(SafetyRule(
            name="pressure_high",
            condition_func=lambda s: s.get('pressure', 0) > self._get_threshold('press_high', 9.0),
            priority=75,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=0,
            message="⚠️ Pression > SEUIL → Augmentation vitesse interdite",
            safety_level=SafetyLevel.HIGH,
            category="pressure",
            thresholds={"press_high": 9.0}
        ))
        
        # R6: Température haute (alerte)
        rules.append(SafetyRule(
            name="temperature_warning",
            condition_func=lambda s: (self._get_threshold('temp_warning_low', 750) < s.get('temperature', 0) <= 
                                      self._get_threshold('temp_high', 800)),
            priority=60,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=1,
            message="⚠️ Température zone alerte → Maintien recommandé",
            safety_level=SafetyLevel.MEDIUM,
            category="temperature",
            thresholds={"temp_warning_low": 750, "temp_warning_high": 800}
        ))
        
        # R7: Pression haute (alerte)
        rules.append(SafetyRule(
            name="pressure_warning",
            condition_func=lambda s: (self._get_threshold('press_warning_low', 8.5) < s.get('pressure', 0) <= 
                                      self._get_threshold('press_high', 9.0)),
            priority=55,
            rule_type=RuleType.CORRECTIVE,
            forbidden_actions=[2],
            safe_action=1,
            message="⚠️ Pression zone alerte → Maintien recommandé",
            safety_level=SafetyLevel.MEDIUM,
            category="pressure",
            thresholds={"press_warning_low": 8.5, "press_warning_high": 9.0}
        ))
        
        # R8: Conditions optimales
        rules.append(SafetyRule(
            name="optimal_conditions",
            condition_func=lambda s: (s.get('temperature', 0) < self._get_threshold('temp_optimal', 700) and
                                      s.get('pressure', 0) < self._get_threshold('press_optimal', 8)),
            priority=10,
            rule_type=RuleType.INFORMATIVE,
            allowed_actions=[0, 1, 2, 3],
            message="✅ Conditions optimales → Toutes actions permises",
            safety_level=SafetyLevel.SAFE,
            category="general",
            thresholds={"temp_optimal": 700, "press_optimal": 8}
        ))
        
        return rules
    
    def _get_threshold(self, param: str, default: float) -> float:
        """Récupère un seuil avec valeur par défaut"""
        return self._dynamic_thresholds.get(param, default)
    
    def set_threshold(self, param: str, value: float):
        """Définit un seuil dynamiquement"""
        self._dynamic_thresholds[param] = value
        print(f"✅ Seuil mis à jour: {param} = {value}")
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """Définit plusieurs seuils à la fois"""
        for param, value in thresholds.items():
            self._dynamic_thresholds[param] = value
        print(f"✅ {len(thresholds)} seuils mis à jour")
    
    def load_config(self, config_path: str):
        """Charge la configuration depuis un fichier JSON externe"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Charger les seuils
            thresholds = config.get('thresholds', {})
            for param, value in thresholds.items():
                self._dynamic_thresholds[param] = value
            
            # Désactiver certaines règles si spécifié
            disabled_rules = config.get('disabled_rules', [])
            for rule in self.rules:
                if rule.name in disabled_rules:
                    rule.is_active = False
            
            print(f"✅ Configuration chargée depuis {config_path}")
            print(f"   Seuils chargés: {len(thresholds)}")
            print(f"   Règles désactivées: {len(disabled_rules)}")
            
        except FileNotFoundError:
            print(f"⚠️ Fichier de configuration non trouvé: {config_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Erreur JSON dans {config_path}: {e}")
    
    def save_config(self, config_path: str):
        """Sauvegarde la configuration actuelle"""
        config = {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "description": "Configuration dynamique des seuils pour le shield neurosymbolique",
            "thresholds": self._dynamic_thresholds.copy(),
            "disabled_rules": [r.name for r in self.rules if not r.is_active]
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration sauvegardée dans {config_path}")
    
    def resolve_conflicts(self, conflicts: List[Dict], proposed_action: int) -> Tuple[int, str]:
        """
        Résout les conflits détectés
        
        Returns:
            (action_résolue, message_de_résolution)
        """
        if not conflicts:
            return proposed_action, "Aucun conflit"
        
        # Trier les conflits par sévérité
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_conflicts = sorted(conflicts, key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
        
        for conflict in sorted_conflicts:
            if 'resolved_action' in conflict:
                return conflict['resolved_action'], f"Conflit résolu: {conflict['type']} → action={conflict['resolved_action']}"
        
        return proposed_action, "Conflit non résolu, action conservée"
    
    def validate_observation(self, observation: np.ndarray) -> Tuple[np.ndarray, bool, List[str]]:
        """Valide et corrige les observations aberrantes"""
        obs_corrected = observation.copy()
        corrected = False
        errors = []
        
        # Valeurs par défaut pour chaque capteur (sûres)
        default_values = [0.35, 0.5, 0.2, 0.1, 0.0, 0.5]
        
        for i, val in enumerate(observation):
            # Détection NaN
            if np.isnan(val):
                obs_corrected[i] = default_values[i] if i < len(default_values) else 0.5
                corrected = True
                errors.append(f"capteur[{i}] = NaN → remplacé par défaut")
            
            # Détection hors intervalle [0,1]
            elif val < 0 or val > 1:
                obs_corrected[i] = max(0, min(1, default_values[i] if i < len(default_values) else 0.5))
                corrected = True
                errors.append(f"capteur[{i}] = {val:.3f} (hors [0,1]) → remplacé")
        
        # Vérification des valeurs après dénormalisation (simulée)
        if not corrected:
            temp_sim = obs_corrected[0] * 850 if len(obs_corrected) > 0 else 0
            press_sim = obs_corrected[1] * 10 if len(obs_corrected) > 1 else 0
            
            # Valeurs physiquement impossibles
            if temp_sim > 1200:
                obs_corrected[0] = 0.5
                corrected = True
                errors.append(f"température simulée {temp_sim:.0f}°C > 1200°C (impossible)")
            if press_sim > 20:
                obs_corrected[1] = 0.5
                corrected = True
                errors.append(f"pression simulée {press_sim:.1f} bar > 20 bar (impossible)")
        
        if corrected:
            print(f"⚠️ Correction d'observations aberrantes: {errors}")
        
        return obs_corrected, corrected, errors
    
    def get_safe_action(self, state_dict: Dict, proposed_action: int) -> Tuple[int, Optional[str], Optional[SafetyRule]]:
        """
        Retourne une action sûre avec:
        - Détection des conflits
        - Application des règles par priorité
        - Résolution des conflits
        """
        self.total_evaluations += 1
        
        # 1. Détecter les conflits
        conflicts = self.detect_conflicts(state_dict)
        
        # 2. Trier les règles par priorité
        sorted_rules = sorted(self.rules, key=lambda x: x.priority, reverse=True)
        
        # 3. Appliquer les règles par ordre de priorité
        triggered_rule = None
        safe_action = proposed_action
        explanation = None
        
        for rule in sorted_rules:
            try:
                if not rule.is_active:
                    continue
                    
                if rule.condition_func(state_dict):
                    triggered_rule = rule
                    self.rule_statistics[rule.name] = self.rule_statistics.get(rule.name, 0) + 1
                    
                    if rule.rule_type == RuleType.BLOCKING and rule.action is not None:
                        safe_action = rule.action
                        explanation = rule.message
                        break
                    
                    if rule.rule_type == RuleType.CORRECTIVE:
                        if rule.forbidden_actions and proposed_action in rule.forbidden_actions:
                            safe_action = rule.safe_action if rule.safe_action is not None else 1
                            explanation = rule.message
                            # Ne pas break, on continue pour voir les autres règles
            except Exception as e:
                print(f"Erreur évaluation règle {rule.name}: {e}")
                continue
        
        # 4. Résoudre les conflits
        resolved_action, resolution_msg = self.resolve_conflicts(conflicts, safe_action)
        
        if resolved_action != safe_action and conflicts:
            explanation = f"{explanation or ''} [Résolution conflit: {resolution_msg}]"
        
        return resolved_action, explanation, triggered_rule
    
    def get_all_active_rules(self, state_dict: Dict) -> List[SafetyRule]:
        """Retourne toutes les règles actives pour un état donné"""
        active = []
        for rule in self.rules:
            try:
                if rule.is_active and rule.condition_func(state_dict):
                    active.append(rule)
            except Exception:
                continue
        return sorted(active, key=lambda x: x.priority, reverse=True)
    
    def get_rule_statistics(self) -> Dict:
        """Retourne les statistiques d'utilisation"""
        total_triggered = sum(self.rule_statistics.values())
        return {
            "total_evaluations": self.total_evaluations,
            "rules_triggered": total_triggered,
            "trigger_rate": total_triggered / max(1, self.total_evaluations),
            "rule_breakdown": self.rule_statistics.copy(),
            "conflicts_detected": len(self.conflict_log),
            "most_triggered": max(self.rule_statistics.items(), key=lambda x: x[1])[0] if self.rule_statistics else None
        }
    
    # ========== 1. SEUILS DYNAMIQUES PAR PRODUIT ==========
    
    def set_production_context(self, context_name: str) -> dict:
        """
        Change dynamiquement tous les seuils selon le contexte de production
        
        Args:
            context_name: 'steel', 'aluminium', 'titanium', 'plastic'
        """
        if context_name not in self.PRODUCTION_CONTEXTS:
            print(f"⚠️ Contexte '{context_name}' inconnu, utilisation de 'steel' par défaut")
            context_name = "steel"
        
        context = self.PRODUCTION_CONTEXTS[context_name]
        
        # Mettre à jour tous les seuils
        for param, value in context.items():
            if param != "name":
                self.set_threshold(param, value)
        
        print(f"✅ Contexte de production changé: {context['name']}")
        print(f"   Température critique: {context['temp_critical']}°C")
        print(f"   Pression critique: {context['press_critical']} bar")
        
        return context
    
    def get_current_context_info(self) -> dict:
        """Retourne les informations du contexte actuel"""
        return {
            "temp_critical": self._get_threshold('temp_critical', 850),
            "temp_high": self._get_threshold('temp_high', 800),
            "press_critical": self._get_threshold('press_critical', 10),
            "press_high": self._get_threshold('press_high', 9.0)
        }
    
    def update_thresholds_for_product(self, product_type: str) -> dict:
        """
        Change dynamiquement tous les seuils selon le produit fabriqué
        À appeler à chaque changement de contexte dans l'environnement
        """
        # Éviter l'import circulaire
        try:
            from cpps_environment import PRODUCT_THRESHOLDS
        except ImportError:
            # Fallback sur les contextes internes
            if product_type not in self.PRODUCTION_CONTEXTS:
                product_type = "steel"
            thresholds = self.PRODUCTION_CONTEXTS[product_type]
            
            # Mise à jour des seuils dynamiques
            threshold_mapping = {
                "temp_critical": thresholds["temp_critical"],
                "temp_high": thresholds["temp_high"],
                "temp_warning_low": thresholds["temp_warning_low"],
                "press_critical": thresholds["press_critical"],
                "press_high": thresholds["press_high"],
                "press_warning_low": thresholds["press_warning_low"],
                "temp_optimal": thresholds["temp_optimal"],
                "press_optimal": thresholds["press_optimal"]
            }
            
            for param, value in threshold_mapping.items():
                self._dynamic_thresholds[param] = value
            
            print(f"✅ Seuils mis à jour pour {thresholds['name']}")
            print(f"   Température critique: {thresholds['temp_critical']}°C")
            print(f"   Pression critique: {thresholds['press_critical']} bar")
            
            return thresholds
        
        if product_type not in PRODUCT_THRESHOLDS:
            product_type = "steel"
        
        thresholds = PRODUCT_THRESHOLDS[product_type]
        
        # Mise à jour des seuils dynamiques
        threshold_mapping = {
            "temp_critical": thresholds["temperature_max"],
            "temp_high": thresholds["temperature_warning"],
            "temp_warning_low": thresholds["temperature_high"],
            "press_critical": thresholds["pressure_max"],
            "press_high": thresholds["pressure_warning"],
            "press_warning_low": thresholds["pressure_high"],
            "temp_optimal": thresholds["temperature_high"] - 50,
            "press_optimal": thresholds["pressure_high"] - 1
        }
        
        for param, value in threshold_mapping.items():
            self._dynamic_thresholds[param] = value
        
        print(f"✅ Seuils mis à jour pour {thresholds['name']}")
        print(f"   Température critique: {thresholds['temperature_max']}°C")
        print(f"   Pression critique: {thresholds['pressure_max']} bar")
        
        return thresholds
    
    def get_current_thresholds(self) -> dict:
        """Retourne les seuils actifs"""
        return {
            "temp_critical": self._get_threshold("temp_critical", 850),
            "temp_high": self._get_threshold("temp_high", 800),
            "temp_warning_low": self._get_threshold("temp_warning_low", 750),
            "press_critical": self._get_threshold("press_critical", 10),
            "press_high": self._get_threshold("press_high", 9.0),
            "press_warning_low": self._get_threshold("press_warning_low", 8.5)
        }
    
    # ========== 2. DÉTECTION DES CONFLITS ENTRE RÈGLES ==========
    
    def detect_conflicts(self, state_dict: Dict) -> List[Dict]:
        """
        Détecte les conflits entre règles actives
        Types: multiple_blocking, conflicting_corrective, priority_conflict
        """
        conflicts = []
        active_rules = self.get_all_active_rules(state_dict)
        
        if len(active_rules) < 2:
            return conflicts
        
        # 1. Conflit: plusieurs règles bloquantes
        blocking_rules = [r for r in active_rules if r.rule_type == RuleType.BLOCKING]
        if len(blocking_rules) > 1:
            # Prendre la règle avec la plus haute priorité
            best_rule = max(blocking_rules, key=lambda x: x.priority)
            conflicts.append({
                "type": "multiple_blocking_rules",
                "severity": "high",
                "rules": [r.name for r in blocking_rules],
                "priorities": [r.priority for r in blocking_rules],
                "resolution": "highest_priority",
                "resolved_action": best_rule.action,
                "resolved_rule": best_rule.name
            })
        
        # 2. Conflit: actions correctives contradictoires
        corrective_rules = [r for r in active_rules if r.rule_type == RuleType.CORRECTIVE]
        if len(corrective_rules) > 1:
            safe_actions = {}
            for rule in corrective_rules:
                if rule.safe_action is not None:
                    safe_actions[rule.name] = rule.safe_action
            
            if len(set(safe_actions.values())) > 1:
                # Résolution: prendre l'action la plus sûre (réduire > maintenir > augmenter)
                safest_action = 0  # reduce_speed par défaut
                conflicts.append({
                    "type": "conflicting_corrective_actions",
                    "severity": "medium",
                    "rules": list(safe_actions.keys()),
                    "conflicting_actions": list(set(safe_actions.values())),
                    "resolution": "most_restrictive",
                    "resolved_action": safest_action
                })
        
        # 3. Conflit: priorité égale
        priority_groups = {}
        for rule in active_rules:
            priority_groups.setdefault(rule.priority, []).append(rule.name)
        
        for priority, rule_names in priority_groups.items():
            if len(rule_names) > 1:
                conflicts.append({
                    "type": "priority_conflict",
                    "severity": "low",
                    "priority": priority,
                    "rules": rule_names,
                    "resolution": "first_match"
                })
        
        # Journaliser les conflits
        for conflict in conflicts:
            self.conflict_log.append({
                "timestamp": datetime.now().isoformat(),
                "state": {k: v for k, v in state_dict.items() if k not in ['agent_id']},
                "conflict": conflict
            })
        
        return conflicts
    
    def get_conflict_summary(self) -> Dict:
        """Retourne un résumé des conflits détectés"""
        if not self.conflict_log:
            return {"total_conflicts": 0, "conflicts_by_type": {}}
        
        by_type = {}
        for entry in self.conflict_log:
            ctype = entry['conflict']['type']
            by_type[ctype] = by_type.get(ctype, 0) + 1
        
        return {
            "total_conflicts": len(self.conflict_log),
            "conflicts_by_type": by_type,
            "recent_conflicts": self.conflict_log[-5:] if self.conflict_log else []
        }
    
    def export_knowledge(self, filepath: str):
        """Exporte la base de connaissances au complet"""
        export_data = {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "total_rules": len(self.rules),
            "dynamic_thresholds": self._dynamic_thresholds,
            "rules": [
                {
                    "name": rule.name,
                    "priority": rule.priority,
                    "type": rule.rule_type.value,
                    "message": rule.message,
                    "is_active": rule.is_active,
                    "thresholds": rule.thresholds,
                    "category": rule.category
                }
                for rule in self.rules
            ],
            "statistics": self.get_rule_statistics(),
            "conflicts": self.get_conflict_summary()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Base de connaissances exportée dans {filepath}")
    
    def reset_statistics(self):
        """Réinitialise toutes les statistiques"""
        self.rule_statistics = {rule.name: 0 for rule in self.rules}
        self.total_evaluations = 0
        self.last_triggered_rule = None
        self.conflict_log = []


def create_default_knowledge_base(config_path: Optional[str] = None) -> KnowledgeBase:
    """Crée une base de connaissances par défaut"""
    return KnowledgeBase(config_path)