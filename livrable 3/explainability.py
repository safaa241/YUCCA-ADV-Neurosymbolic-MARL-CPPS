"""
LIVRABLE 3 - MODULE D'EXPLICABILITÉ AVANCÉ
Génération d'explications pour anomalies, conflits et mode dégradé

Auteur: FEKNI Safaa
Projet: YUCCA-ADV
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum


class ExplanationType(Enum):
    """Types d'explications disponibles"""
    ACTION_MODIFIED = "action_modified"
    ACTION_BLOCKED = "action_blocked"
    ACTION_SAFE = "action_safe"
    RULE_TRIGGERED = "rule_triggered"
    ANOMALY_DETECTED = "anomaly_detected"
    RAPID_CHANGE = "rapid_change"
    CONFLICT_RESOLVED = "conflict_resolved"
    EMERGENCY_MODE = "emergency_mode"
    CONTEXT_CHANGED = "context_changed"
    SHIELD_STATS = "shield_stats"


class ExplanationSeverity(Enum):
    """Niveaux de sévérité des explications"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class Explanation:
    """Structure complète d'une explication"""
    
    timestamp: str
    agent_id: int
    type: str
    severity: str
    title: str
    description: str
    icon: str = "📝"
    
    details: Dict[str, Any] = field(default_factory=dict)
    
    original_action: Optional[int] = None
    safe_action: Optional[int] = None
    triggering_rule: Optional[str] = None
    context: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convertit l'explication en dictionnaire"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convertit l'explication en JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convertit l'explication en Markdown"""
        result = f"\n### {self.icon} {self.title}\n\n"
        result += f"**Agent:** {self.agent_id}\n"
        result += f"**Type:** {self.type}\n"
        result += f"**Sévérité:** {self.severity}\n"
        result += f"**Timestamp:** {self.timestamp}\n\n"
        result += f"{self.description}\n\n"
        result += "**Détails:**\n"
        result += "```json\n"
        result += json.dumps(self.details, indent=2, ensure_ascii=False)
        result += "\n```\n"
        return result
    
    def to_html(self) -> str:
        """Convertit l'explication en HTML avec style"""
        colors = {
            "critical": "#e74c3c",
            "warning": "#f39c12",
            "info": "#3498db",
            "success": "#2ecc71"
        }
        color = colors.get(self.severity, "#666")
        
        details_str = json.dumps(self.details, indent=2, ensure_ascii=False)
        
        return f"""
<div style="border-left: 4px solid {color}; padding: 12px; margin: 10px 0; background-color: #f8f9fa; border-radius: 8px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.5rem;">{self.icon}</span>
        <strong>{self.title}</strong>
        <span style="margin-left: auto; font-size: 0.8rem; color: #666;">{self.timestamp}</span>
    </div>
    <div style="margin-top: 8px; color: #444;">
        {self.description}
    </div>
    <details style="margin-top: 8px;">
        <summary style="cursor: pointer; color: {color};">Details</summary>
        <pre style="margin-top: 8px; background: #2d2d2d; color: #f8f8f2; padding: 8px; border-radius: 4px; overflow-x: auto;">{details_str}</pre>
    </details>
</div>
        """


class ExplanationGenerator:
    """
    Générateur d'explications avancé pour le shield neurosymbolique
    """
    
    def __init__(self):
        self.explanation_history: List[Explanation] = []
        
        self.action_names = {
            0: "Reduire la vitesse",
            1: "Maintenir la vitesse",
            2: "Augmenter la vitesse",
            3: "Mettre en attente",
            4: "Arret d'urgence"
        }
        
        self.rule_names = {
            "temperature_critical": "Temperature Critique",
            "temperature_high": "Temperature Elevee",
            "temperature_warning": "Temperature Haute",
            "pressure_critical": "Pression Critique",
            "pressure_high": "Pression Elevee",
            "pressure_warning": "Pression Haute",
            "maintenance_required": "Maintenance Requise",
            "optimal_conditions": "Conditions Optimales",
            "speed_limit": "Limite de Vitesse"
        }
        
        self.rule_severity = {
            "temperature_critical": "critical",
            "pressure_critical": "critical",
            "maintenance_required": "critical",
            "temperature_high": "warning",
            "pressure_high": "warning",
            "temperature_warning": "warning",
            "pressure_warning": "warning",
            "speed_limit": "warning"
        }
    
    def generate_anomaly_explanation(self, agent_id: int, sensor: str,
                                     original_value: float, corrected_value: float,
                                     detection_method: str = "median_filter") -> Explanation:
        """Génère une explication pour une anomalie capteur"""
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.ANOMALY_DETECTED.value,
            severity=ExplanationSeverity.WARNING.value,
            icon="🔄",
            title=f"Anomalie capteur detectee: {sensor}",
            description=f"La valeur du capteur {sensor} ({original_value:.1f}) a ete corrigee a {corrected_value:.1f}.",
            details={
                "sensor": sensor,
                "original_value": original_value,
                "corrected_value": corrected_value,
                "detection_method": detection_method,
                "correction_applied": True
            }
        )
    
    def generate_rapid_change_explanation(self, agent_id: int, 
                                          change_rate: float,
                                          current_temp: float,
                                          previous_temp: float,
                                          time_window: int = 5) -> Explanation:
        """Génère une explication pour un changement trop rapide"""
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.RAPID_CHANGE.value,
            severity=ExplanationSeverity.CRITICAL.value,
            icon="⚡",
            title="Montee en temperature rapide detectee",
            description=f"Temperature: {previous_temp:.0f}C -> {current_temp:.0f}C (taux: {change_rate:.1f}C/step)",
            details={
                "change_rate": change_rate,
                "current_temperature": current_temp,
                "previous_temperature": previous_temp,
                "time_window": time_window,
                "emergency_protocol": "Active - STOP immediat"
            }
        )
    
    def generate_conflict_explanation(self, agent_id: int,
                                      conflicting_rules: List[str],
                                      resolution_strategy: str,
                                      final_action: int,
                                      original_action: int,
                                      resolution_reason: str = "") -> Explanation:
        """Génère une explication pour un conflit entre regles"""
        rule_names = [self.rule_names.get(r, r) for r in conflicting_rules]
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.CONFLICT_RESOLVED.value,
            severity=ExplanationSeverity.WARNING.value,
            icon="⚖️",
            title="Conflit entre regles de securite",
            description=f"{len(conflicting_rules)} regles en conflit: {', '.join(rule_names)}",
            details={
                "conflicting_rules": conflicting_rules,
                "conflicting_rules_display": rule_names,
                "resolution_strategy": resolution_strategy,
                "resolution_reason": resolution_reason,
                "original_action": original_action,
                "original_action_name": self.action_names.get(original_action, "inconnue"),
                "final_action": final_action,
                "final_action_name": self.action_names.get(final_action, "inconnue")
            }
        )
    
    def generate_emergency_mode_explanation(self, agent_id: int,
                                            reason: str,
                                            temperature: float,
                                            previous_mode: str) -> Explanation:
        """Génère une explication pour l'activation du mode degrade"""
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.EMERGENCY_MODE.value,
            severity=ExplanationSeverity.CRITICAL.value,
            icon="🚨",
            title="MODE DEGRADE ACTIVE",
            description=f"Mode degrade active: {reason}",
            details={
                "reason": reason,
                "temperature": temperature,
                "previous_mode": previous_mode,
                "current_mode": "emergency",
                "allowed_actions": ["Arret d'urgence uniquement"]
            }
        )
    
    def generate_context_change_explanation(self, agent_id: int,
                                            old_context: str,
                                            new_context: str,
                                            material: str) -> Explanation:
        """Génère une explication pour un changement de contexte"""
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.CONTEXT_CHANGED.value,
            severity=ExplanationSeverity.INFO.value,
            icon="🏭",
            title="Contexte de production modifie",
            description=f"Contexte: {old_context} -> {new_context} (materiau: {material})",
            details={
                "old_context": old_context,
                "new_context": new_context,
                "material": material,
                "thresholds_updated": True
            }
        )
    
    def generate_action_modification_explanation(self, agent_id: int,
                                                  original_action: int,
                                                  safe_action: int,
                                                  rule_name: str,
                                                  state: Dict,
                                                  reason: str) -> Explanation:
        """Génère une explication pour modification d'action"""
        original_name = self.action_names.get(original_action, "inconnue")
        safe_name = self.action_names.get(safe_action, "inconnue")
        rule_display = self.rule_names.get(rule_name, rule_name)
        
        if safe_action == 4:
            severity = ExplanationSeverity.CRITICAL.value
            icon = "🔴"
            title = f"Action BLOQUEE: {original_name}"
            description = f"Action remplacee par ARRET D'URGENCE. Regle: {rule_display}"
        elif original_action == 2 and safe_action == 0:
            severity = ExplanationSeverity.WARNING.value
            icon = "⚠️"
            title = f"Action corrigee: {original_name} -> {safe_name}"
            description = f"Augmentation de vitesse bloquee. Regle: {rule_display}"
        else:
            severity = ExplanationSeverity.INFO.value
            icon = "🔄"
            title = f"Action modifiee: {original_name} -> {safe_name}"
            description = reason if reason else f"Modification par regle {rule_display}"
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.ACTION_MODIFIED.value,
            severity=severity,
            icon=icon,
            title=title,
            description=description,
            original_action=original_action,
            safe_action=safe_action,
            triggering_rule=rule_name,
            details={
                "original_action_name": original_name,
                "safe_action_name": safe_name,
                "triggering_rule": rule_display,
                "reason": reason,
                "temperature": f"{state.get('temperature', 0):.1f}C",
                "pressure": f"{state.get('pressure', 0):.1f} bar"
            }
        )
    
    def generate_safe_action_explanation(self, agent_id: int,
                                         action: int,
                                         state: Dict) -> Explanation:
        """Génère une explication pour une action sûre"""
        action_name = self.action_names.get(action, "inconnue")
        
        temp = state.get('temperature', 0)
        pressure = state.get('pressure', 0)
        
        if temp < 700 and pressure < 8:
            status = "Conditions optimales"
            icon = "✅"
        elif temp < 800 and pressure < 9:
            status = "Conditions normales"
            icon = "✅"
        else:
            status = "Zone de surveillance"
            icon = "ℹ️"
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.ACTION_SAFE.value,
            severity=ExplanationSeverity.SUCCESS.value,
            icon=icon,
            title=f"Action sure: {action_name}",
            description=f"Action '{action_name}' executee. {status}",
            original_action=action,
            safe_action=action,
            details={
                "action_name": action_name,
                "temperature": f"{temp:.1f}C",
                "pressure": f"{pressure:.1f} bar",
                "safety_status": status
            }
        )
    
    def generate_rule_triggered_explanation(self, agent_id: int,
                                            rule_name: str,
                                            state: Dict,
                                            condition_values: Dict) -> Explanation:
        """Génère une explication pour une regle declenchee"""
        rule_display = self.rule_names.get(rule_name, rule_name)
        severity = self.rule_severity.get(rule_name, "info")
        
        severity_enum = ExplanationSeverity.WARNING if severity == "warning" else ExplanationSeverity.INFO
        icon = "⚠️" if severity == "warning" else "ℹ️"
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            type=ExplanationType.RULE_TRIGGERED.value,
            severity=severity_enum.value,
            icon=icon,
            title=f"Regle declenchee: {rule_display}",
            description=f"La regle '{rule_display}' a ete declenchee.",
            triggering_rule=rule_name,
            details={
                "rule": rule_name,
                "rule_display": rule_display,
                "condition_values": condition_values,
                "temperature": f"{state.get('temperature', 0):.1f}C",
                "pressure": f"{state.get('pressure', 0):.1f} bar"
            }
        )
    
    def add_explanation(self, explanation: Explanation):
        """Ajoute une explication à l'historique"""
        self.explanation_history.append(explanation)
    
    def get_explanations(self, 
                        type_filter: Optional[str] = None,
                        severity_filter: Optional[str] = None,
                        agent_filter: Optional[int] = None,
                        limit: int = 100) -> List[Explanation]:
        """Filtre et retourne les explications"""
        result = self.explanation_history
        
        if type_filter:
            result = [e for e in result if e.type == type_filter]
        if severity_filter:
            result = [e for e in result if e.severity == severity_filter]
        if agent_filter is not None:
            result = [e for e in result if e.agent_id == agent_filter]
        
        return result[-limit:] if limit else result
    
    def get_explanations_summary(self) -> Dict:
        """Retourne un résumé des explications"""
        total = len(self.explanation_history)
        
        if total == 0:
            return {"total": 0, "by_type": {}, "by_severity": {}, "by_agent": {}}
        
        by_type = {}
        by_severity = {}
        by_agent = {}
        
        for exp in self.explanation_history:
            by_type[exp.type] = by_type.get(exp.type, 0) + 1
            by_severity[exp.severity] = by_severity.get(exp.severity, 0) + 1
            by_agent[exp.agent_id] = by_agent.get(exp.agent_id, 0) + 1
        
        return {
            "total": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "by_agent": by_agent
        }
    
    def export_explanations(self, filepath: str, format: str = "json"):
        """Exporte les explications vers un fichier"""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total": len(self.explanation_history),
                "summary": self.get_explanations_summary(),
                "explanations": [exp.to_dict() for exp in self.explanation_history]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Explications du Shield Neurosymbolique\n\n")
                f.write(f"Genere le: {datetime.now().isoformat()}\n")
                f.write(f"Total: {len(self.explanation_history)} explications\n\n")
                f.write("---\n\n")
                for exp in self.explanation_history[-50:]:
                    f.write(exp.to_markdown())
                    f.write("\n---\n\n")
        
        elif format == "html":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                f.write("<title>Explications Shield Neurosymbolique</title>\n")
                f.write("<meta charset='UTF-8'>\n")
                f.write("<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>\n")
                f.write("</head>\n<body>\n")
                f.write(f"<h1>Explications du Shield Neurosymbolique</h1>\n")
                f.write(f"<p>Genere le: {datetime.now().isoformat()}</p>\n")
                f.write(f"<p>Total: {len(self.explanation_history)} explications</p>\n")
                f.write("<hr>\n")
                for exp in self.explanation_history[-50:]:
                    f.write(exp.to_html())
                    f.write("<hr>\n")
                f.write("</body>\n</html>")
        
        print(f"Explications exportees dans {filepath}")
    
    def clear_history(self):
        """Efface l'historique des explications"""
        self.explanation_history = []
    
    def get_last_n_explanations(self, n: int = 10) -> List[Dict]:
        """Retourne les n dernieres explications au format dictionnaire"""
        return [exp.to_dict() for exp in self.explanation_history[-n:]]


class ExplainableShield:
    """
    Shield explicable qui combine le filtrage des actions avec la generation d'explications
    """
    
    def __init__(self, knowledge_base, explanation_generator: ExplanationGenerator = None):
        self.kb = knowledge_base
        self.explanation_gen = explanation_generator or ExplanationGenerator()
        
        self.total_checks = 0
        self.modified_actions = 0
        self.blocked_actions = 0
    
    def filter_action_with_explanation(self, action: int, observation: np.ndarray,
                                        agent_id: int = 0):
        """
        Filtre une action et genere une explication
        A implementer selon les besoins
        """
        self.total_checks += 1
        
        state_dict = self._observation_to_state(observation)
        safe_action, reason, triggered_rule = self.kb.get_safe_action(state_dict, action)
        
        modified = (safe_action != action)
        
        if modified:
            self.modified_actions += 1
            if safe_action == 4:
                self.blocked_actions += 1
            
            explanation = self.explanation_gen.generate_action_modification_explanation(
                agent_id, action, safe_action,
                triggered_rule.name if triggered_rule else "unknown",
                state_dict, reason or "Securite"
            )
            self.explanation_gen.add_explanation(explanation)
        
        return safe_action, modified, reason
    
    def _observation_to_state(self, obs: np.ndarray) -> Dict:
        """Convertit l'observation en dictionnaire d'etat"""
        return {
            'temperature': obs[0] * 850,
            'pressure': obs[1] * 10,
            'speed': obs[2] * 10,
            'production': obs[3] * 100,
            'maintenance_needed': obs[4] > 0.5,
            'time_step': obs[5] * 500
        }
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du shield"""
        return {
            'total_checks': self.total_checks,
            'modified_actions': self.modified_actions,
            'blocked_actions': self.blocked_actions,
            'modification_rate': self.modified_actions / max(1, self.total_checks),
            'explanations_count': len(self.explanation_gen.explanation_history)
        }


# Fonction utilitaire pour creer un generateur d'explications par defaut
def create_explanation_generator() -> ExplanationGenerator:
    """Cree un generateur d'explications avec la configuration par defaut"""
    return ExplanationGenerator()


if __name__ == "__main__":
    # Test rapide du module
    print("Test du module d'explicabilite")
    print("=" * 50)
    
    gen = ExplanationGenerator()
    
    exp1 = gen.generate_safe_action_explanation(0, 1, {"temperature": 650, "pressure": 7.5})
    gen.add_explanation(exp1)
    
    exp2 = gen.generate_action_modification_explanation(
        0, 2, 0, "temperature_high", 
        {"temperature": 820, "pressure": 8.5}, 
        "Temperature elevee > 800C"
    )
    gen.add_explanation(exp2)
    
    exp3 = gen.generate_anomaly_explanation(1, "temperature", 950, 820)
    gen.add_explanation(exp3)
    
    exp4 = gen.generate_rapid_change_explanation(0, 65, 820, 650)
    gen.add_explanation(exp4)
    
    print(f"Total explications: {len(gen.explanation_history)}")
    print(gen.get_explanations_summary())
    print("\nTest reussi!")