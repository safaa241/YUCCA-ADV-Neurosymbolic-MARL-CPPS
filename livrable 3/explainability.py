"""
LIVRABLE 3 - Module d'Explicabilité pour MARL Neurosymbolique
Génère des explications lisibles pour les décisions des agents

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum


class ExplanationLevel(Enum):
    """Niveau de détail des explications"""
    BASIC = "basic"          # Explication simple
    DETAILED = "detailed"    # Explication détaillée
    FULL = "full"           # Explication complète avec état


class ExplanationType(Enum):
    """Type d'explication"""
    ACTION_MODIFIED = "action_modified"
    ACTION_BLOCKED = "action_blocked"
    ACTION_SAFE = "action_safe"
    RULE_TRIGGERED = "rule_triggered"
    STATE_CHANGE = "state_change"
    PERFORMANCE_ALERT = "performance_alert"


@dataclass
class Explanation:
    """Structure d'une explication"""
    timestamp: str
    agent_id: int
    explanation_type: str
    title: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, critical, success
    
    def to_dict(self) -> Dict:
        """Convertit l'explication en dictionnaire"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convertit l'explication en JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convertit l'explication en Markdown"""
        emoji = {
            "critical": "🔴",
            "warning": "⚠️",
            "info": "ℹ️",
            "success": "✅"
        }.get(self.severity, "📝")
        
        return f"""
### {emoji} {self.title}

**Agent:** {self.agent_id}
**Type:** {self.explanation_type}
**Timestamp:** {self.timestamp}

{self.description}

**Détails:**
{json.dumps(self.details, indent=2, ensure_ascii=False)}
        """
    
    def to_html(self) -> str:
        """Convertit l'explication en HTML"""
        colors = {
            "critical": "#e74c3c",
            "warning": "#f39c12",
            "info": "#3498db",
            "success": "#2ecc71"
        }
        color = colors.get(self.severity, "#666")
        
        return f"""
<div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
    <strong>{self.title}</strong><br>
    <small>Agent {self.agent_id} | {self.timestamp}</small><br>
    <p>{self.description}</p>
    <details>
        <summary>Détails</summary>
        <pre>{json.dumps(self.details, indent=2, ensure_ascii=False)}</pre>
    </details>
</div>
        """


class ExplanationGenerator:
    """
    Générateur d'explications pour les décisions du shield neurosymbolique
    """
    
    def __init__(self):
        self.explanation_history: List[Explanation] = []
        
        # Mapping des actions
        self.action_names = {
            0: "Réduire la vitesse",
            1: "Maintenir la vitesse",
            2: "Augmenter la vitesse",
            3: "Mettre en attente (idle)",
            4: "Arrêt d'urgence"
        }
        
        # Mapping des règles
        self.rule_names = {
            "temperature_critical": "Température Critique",
            "temperature_high": "Température Élevée",
            "temperature_warning": "Température Haute",
            "pressure_critical": "Pression Critique",
            "pressure_high": "Pression Élevée",
            "pressure_warning": "Pression Haute",
            "maintenance_required": "Maintenance Requise",
            "optimal_conditions": "Conditions Optimales",
            "speed_limit": "Limite de Vitesse",
            "production_boost": "Boost de Production"
        }
        
        # Mapping des niveaux de sévérité
        self.rule_severity = {
            "temperature_critical": "critical",
            "pressure_critical": "critical",
            "maintenance_required": "critical",
            "temperature_high": "warning",
            "pressure_high": "warning",
            "temperature_warning": "warning",
            "pressure_warning": "warning",
            "speed_limit": "warning",
            "optimal_conditions": "success",
            "production_boost": "success"
        }
    
    def generate_action_modification_explanation(
        self,
        agent_id: int,
        original_action: int,
        safe_action: int,
        triggered_rule: str,
        state: Dict[str, float],
        reason: str
    ) -> Explanation:
        """
        Génère une explication pour une action modifiée
        """
        original_name = self.action_names.get(original_action, "inconnue")
        safe_name = self.action_names.get(safe_action, "inconnue")
        rule_name = self.rule_names.get(triggered_rule, triggered_rule)
        severity = self.rule_severity.get(triggered_rule, "warning")
        
        # Description détaillée
        if safe_action == 4:  # emergency_stop
            description = f"L'action '{original_name}' a été remplacée par un ARRÊT D'URGENCE car {reason.lower()}"
        elif original_action == 2 and safe_action == 0:  # increase -> reduce
            description = f"L'augmentation de vitesse a été bloquée et remplacée par une réduction car {reason.lower()}"
        elif original_action == 2 and safe_action == 1:  # increase -> maintain
            description = f"L'augmentation de vitesse a été bloquée et remplacée par un maintien car {reason.lower()}"
        else:
            description = f"L'action '{original_name}' a été remplacée par '{safe_name}' car {reason.lower()}"
        
        # Détails
        details = {
            "original_action": original_action,
            "original_action_name": original_name,
            "safe_action": safe_action,
            "safe_action_name": safe_name,
            "triggered_rule": rule_name,
            "rule_reason": reason,
            "state": {
                "temperature": f"{state.get('temperature', 0):.1f}°C",
                "pressure": f"{state.get('pressure', 0):.1f} bar",
                "speed": f"{state.get('speed', 0):.1f} m/s",
                "production": state.get('production', 0),
                "maintenance_needed": state.get('maintenance_needed', False),
                "time_step": state.get('time_step', 0)
            }
        }
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            explanation_type=ExplanationType.ACTION_MODIFIED.value,
            title=f"Action modifiée: {original_name} → {safe_name}",
            description=description,
            details=details,
            severity=severity
        )
    
    def generate_action_blocked_explanation(
        self,
        agent_id: int,
        original_action: int,
        triggered_rule: str,
        state: Dict[str, float],
        reason: str
    ) -> Explanation:
        """
        Génère une explication pour une action bloquée
        """
        original_name = self.action_names.get(original_action, "inconnue")
        rule_name = self.rule_names.get(triggered_rule, triggered_rule)
        
        description = f"L'action '{original_name}' a été COMPLÈTEMENT BLOQUÉE et remplacée par un arrêt d'urgence car {reason.lower()}"
        
        details = {
            "blocked_action": original_action,
            "blocked_action_name": original_name,
            "triggered_rule": rule_name,
            "rule_reason": reason,
            "state": {
                "temperature": f"{state.get('temperature', 0):.1f}°C",
                "pressure": f"{state.get('pressure', 0):.1f} bar",
                "speed": f"{state.get('speed', 0):.1f} m/s"
            }
        }
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            explanation_type=ExplanationType.ACTION_BLOCKED.value,
            title=f"Action bloquée: {original_name}",
            description=description,
            details=details,
            severity="critical"
        )
    
    def generate_safe_action_explanation(
        self,
        agent_id: int,
        action: int,
        state: Dict[str, float]
    ) -> Explanation:
        """
        Génère une explication pour une action sûre
        """
        action_name = self.action_names.get(action, "inconnue")
        
        # Déterminer le niveau de sécurité
        temp = state.get('temperature', 0)
        pressure = state.get('pressure', 0)
        
        if temp < 700 and pressure < 8:
            safety_status = "✅ Conditions optimales"
            severity = "success"
        elif temp < 750 and pressure < 8.5:
            safety_status = "✅ Conditions normales"
            severity = "success"
        elif temp < 800 and pressure < 9.0:
            safety_status = "⚠️ Conditions acceptables (surveillance recommandée)"
            severity = "warning"
        else:
            safety_status = "⚠️ Zone dangereuse (l'action est sûre mais la situation est critique)"
            severity = "warning"
        
        description = f"L'action '{action_name}' a été exécutée. {safety_status}"
        
        details = {
            "action": action,
            "action_name": action_name,
            "state": {
                "temperature": f"{temp:.1f}°C",
                "pressure": f"{pressure:.1f} bar",
                "speed": f"{state.get('speed', 0):.1f} m/s"
            }
        }
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            explanation_type=ExplanationType.ACTION_SAFE.value,
            title=f"Action sûre: {action_name}",
            description=description,
            details=details,
            severity=severity
        )
    
    def generate_rule_triggered_explanation(
        self,
        agent_id: int,
        rule_name: str,
        state: Dict[str, float],
        condition_values: Dict[str, float]
    ) -> Explanation:
        """
        Génère une explication pour une règle déclenchée
        """
        rule_display = self.rule_names.get(rule_name, rule_name)
        severity = self.rule_severity.get(rule_name, "info")
        
        description = f"La règle '{rule_display}' a été déclenchée."
        
        details = {
            "rule": rule_name,
            "rule_display": rule_display,
            "condition_values": condition_values,
            "state": {
                "temperature": f"{state.get('temperature', 0):.1f}°C",
                "pressure": f"{state.get('pressure', 0):.1f} bar",
                "speed": f"{state.get('speed', 0):.1f} m/s"
            }
        }
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            explanation_type=ExplanationType.RULE_TRIGGERED.value,
            title=f"Règle déclenchée: {rule_display}",
            description=description,
            details=details,
            severity=severity
        )
    
    def generate_performance_alert(
        self,
        agent_id: int,
        metric: str,
        current_value: float,
        threshold: float,
        trend: str
    ) -> Explanation:
        """
        Génère une alerte de performance
        """
        description = f"La métrique '{metric}' a atteint {current_value:.1f} (seuil: {threshold:.1f}). Tendance: {trend}"
        
        details = {
            "metric": metric,
            "current_value": current_value,
            "threshold": threshold,
            "trend": trend
        }
        
        severity = "warning" if current_value > threshold else "info"
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            explanation_type=ExplanationType.PERFORMANCE_ALERT.value,
            title=f"Alerte performance: {metric}",
            description=description,
            details=details,
            severity=severity
        )
    
    def generate_state_change_explanation(
        self,
        agent_id: int,
        previous_state: Dict[str, float],
        new_state: Dict[str, float],
        action_taken: int
    ) -> Explanation:
        """
        Génère une explication pour un changement d'état
        """
        action_name = self.action_names.get(action_taken, "inconnue")
        
        # Calculer les deltas
        temp_delta = new_state.get('temperature', 0) - previous_state.get('temperature', 0)
        pressure_delta = new_state.get('pressure', 0) - previous_state.get('pressure', 0)
        
        temp_change = f"+{temp_delta:.1f}°C" if temp_delta >= 0 else f"{temp_delta:.1f}°C"
        pressure_change = f"+{pressure_delta:.1f} bar" if pressure_delta >= 0 else f"{pressure_delta:.1f} bar"
        
        description = f"Suite à l'action '{action_name}', la température a varié de {temp_change} et la pression de {pressure_change}."
        
        details = {
            "action_taken": action_taken,
            "action_name": action_name,
            "previous_state": {
                "temperature": f"{previous_state.get('temperature', 0):.1f}°C",
                "pressure": f"{previous_state.get('pressure', 0):.1f} bar",
                "speed": f"{previous_state.get('speed', 0):.1f} m/s"
            },
            "new_state": {
                "temperature": f"{new_state.get('temperature', 0):.1f}°C",
                "pressure": f"{new_state.get('pressure', 0):.1f} bar",
                "speed": f"{new_state.get('speed', 0):.1f} m/s"
            },
            "deltas": {
                "temperature": temp_delta,
                "pressure": pressure_delta
            }
        }
        
        return Explanation(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            explanation_type=ExplanationType.STATE_CHANGE.value,
            title=f"Changement d'état après {action_name}",
            description=description,
            details=details,
            severity="info"
        )
    
    def add_explanation(self, explanation: Explanation):
        """Ajoute une explication à l'historique"""
        self.explanation_history.append(explanation)
    
    def get_explanations(self, 
                        agent_id: Optional[int] = None,
                        explanation_type: Optional[str] = None,
                        severity: Optional[str] = None,
                        limit: int = 100) -> List[Explanation]:
        """
        Récupère les explications avec filtres
        """
        explanations = self.explanation_history
        
        if agent_id is not None:
            explanations = [e for e in explanations if e.agent_id == agent_id]
        
        if explanation_type is not None:
            explanations = [e for e in explanations if e.explanation_type == explanation_type]
        
        if severity is not None:
            explanations = [e for e in explanations if e.severity == severity]
        
        return explanations[-limit:]
    
    def get_explanations_summary(self) -> Dict:
        """
        Retourne un résumé des explications
        """
        total = len(self.explanation_history)
        
        if total == 0:
            return {"total": 0, "by_type": {}, "by_severity": {}, "by_agent": {}}
        
        # Compter par type
        by_type = {}
        for exp in self.explanation_history:
            by_type[exp.explanation_type] = by_type.get(exp.explanation_type, 0) + 1
        
        # Compter par sévérité
        by_severity = {}
        for exp in self.explanation_history:
            by_severity[exp.severity] = by_severity.get(exp.severity, 0) + 1
        
        # Compter par agent
        by_agent = {}
        for exp in self.explanation_history:
            by_agent[exp.agent_id] = by_agent.get(exp.agent_id, 0) + 1
        
        return {
            "total": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "by_agent": by_agent
        }
    
    def export_explanations(self, filepath: str, format: str = "json"):
        """
        Exporte les explications vers un fichier
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total": len(self.explanation_history),
                "explanations": [exp.to_dict() for exp in self.explanation_history]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Explications du Shield Neurosymbolique\n\n")
                f.write(f"Généré le: {datetime.now().isoformat()}\n")
                f.write(f"Total: {len(self.explanation_history)} explications\n\n")
                f.write("---\n\n")
                for exp in self.explanation_history[-50:]:  # Dernières 50
                    f.write(exp.to_markdown())
                    f.write("\n---\n\n")
        
        elif format == "html":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                f.write("<title>Explications Shield Neurosymbolique</title>\n")
                f.write("<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>\n")
                f.write("</head>\n<body>\n")
                f.write(f"<h1>Explications du Shield Neurosymbolique</h1>\n")
                f.write(f"<p>Généré le: {datetime.now().isoformat()}</p>\n")
                f.write(f"<p>Total: {len(self.explanation_history)} explications</p>\n")
                f.write("<hr>\n")
                for exp in self.explanation_history[-50:]:
                    f.write(exp.to_html())
                    f.write("<hr>\n")
                f.write("</body>\n</html>")
        
        print(f"✅ Explications exportées dans {filepath}")
    
    def clear_history(self):
        """Efface l'historique des explications"""
        self.explanation_history = []


class ExplainableShield:
    """
    Shield explicable qui combine le filtrage des actions avec la génération d'explications
    """
    
    def __init__(self, knowledge_base, explanation_generator: ExplanationGenerator = None):
        self.kb = knowledge_base
        self.explanation_gen = explanation_generator or ExplanationGenerator()
        
        # Statistiques
        self.total_checks = 0
        self.modified_actions = 0
        self.blocked_actions = 0
        
    def filter_action_with_explanation(
        self,
        action: int,
        observation: np.ndarray,
        agent_id: int = 0
    ) -> Tuple[int, bool, Optional[Explanation]]:
        """
        Filtre une action et génère une explication
        """
        self.total_checks += 1
        
        # Convertir l'observation en état
        state_dict = self._observation_to_state(observation)
        
        # Obtenir l'action sûre
        safe_action, reason, triggered_rule = self.kb.get_safe_action(state_dict, action)
        
        modified = (safe_action != action)
        
        explanation = None
        
        if modified:
            self.modified_actions += 1
            
            if safe_action == 4:  # emergency_stop
                self.blocked_actions += 1
                explanation = self.explanation_gen.generate_action_blocked_explanation(
                    agent_id, action, triggered_rule.name if triggered_rule else "unknown",
                    state_dict, reason or "violation de sécurité critique"
                )
            else:
                explanation = self.explanation_gen.generate_action_modification_explanation(
                    agent_id, action, safe_action,
                    triggered_rule.name if triggered_rule else "unknown",
                    state_dict, reason or "violation de sécurité"
                )
            
            self.explanation_gen.add_explanation(explanation)
        else:
            # Optionnel: générer une explication pour les actions sûres
            pass
        
        return safe_action, modified, explanation
    
    def _observation_to_state(self, obs: np.ndarray) -> Dict:
        """Convertit l'observation en dictionnaire d'état"""
        return {
            'temperature': obs[0] * 850,
            'pressure': obs[1] * 10,
            'speed': obs[2] * 10,
            'production': obs[3] * 100,
            'maintenance_needed': obs[4] > 0.5,
            'time_step': obs[5] * 500
        }
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        return {
            'total_checks': self.total_checks,
            'modified_actions': self.modified_actions,
            'blocked_actions': self.blocked_actions,
            'modification_rate': self.modified_actions / max(1, self.total_checks),
            'explanations_count': len(self.explanation_gen.explanation_history)
        }