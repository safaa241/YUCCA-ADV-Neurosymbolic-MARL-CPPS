# Jumeau numérique (3 machines, physique simulée)
# Version avec contexte de production dynamique

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class MachineState:
    """Structure de données pour l'état d'une machine"""
    temperature: float = 20.0
    pressure: float = 5.0
    speed: float = 0.0
    production_count: int = 0
    maintenance_needed: bool = False


# Types de produits possibles pour les seuils dynamiques
PRODUCT_TYPES = ['steel', 'aluminium', 'titanium', 'plastic']

# Seuils par type de produit
PRODUCT_THRESHOLDS = {
    "steel": {
        "name": "Acier",
        "temperature_max": 850,
        "temperature_warning": 800,
        "temperature_high": 750,
        "pressure_max": 10,
        "pressure_warning": 9.0,
        "pressure_high": 8.5,
        "speed_max": 10,
        "production_target": 10
    },
    "aluminium": {
        "name": "Aluminium",
        "temperature_max": 650,
        "temperature_warning": 600,
        "temperature_high": 550,
        "pressure_max": 8,
        "pressure_warning": 7.0,
        "pressure_high": 6.5,
        "speed_max": 12,
        "production_target": 20
    },
    "titanium": {
        "name": "Titane",
        "temperature_max": 950,
        "temperature_warning": 900,
        "temperature_high": 850,
        "pressure_max": 12,
        "pressure_warning": 11.0,
        "pressure_high": 10.5,
        "speed_max": 8,
        "production_target": 5
    },
    "plastic": {
        "name": "Plastique",
        "temperature_max": 400,
        "temperature_warning": 350,
        "temperature_high": 300,
        "pressure_max": 5,
        "pressure_warning": 4.5,
        "pressure_high": 4.0,
        "speed_max": 6,
        "production_target": 15
    }
}


class CPPSProductionEnv(gym.Env):
    """
    Environnement de production CPPS avec contexte dynamique
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, num_agents=3, episode_length=500, render_mode=None, product_type='steel'):
        super().__init__()
        
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.current_product = product_type
        
        # Configuration des machines
        self.machines = {
            0: {"name": "Welding Robot", "type": "temperature_sensitive"},
            1: {"name": "Painting Robot", "type": "pressure_sensitive"},
            2: {"name": "Quality Control", "type": "precision_sensitive"}
        }
        
        # États et actions
        self.action_spaces = {
            i: spaces.Discrete(5) for i in range(num_agents)
        }
        # Actions: 0=reduce_speed, 1=maintain_speed, 2=increase_speed, 3=idle, 4=emergency_stop
        
        self.observation_spaces = {
            i: spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
            for i in range(num_agents)
        }
        
        self.machine_states = {i: MachineState() for i in range(num_agents)}
        
        # Paramètres de simulation
        self.current_step = 0
        self.total_production = 0
        self.total_reward = 0
        self.violations_log = []
        
        # Initialiser les seuils selon le produit
        self._update_thresholds_from_product()
        
        # Debug
        self.debug_counter = 0
        self.context_change_history = []
    
    def _update_thresholds_from_product(self):
        """Met à jour les seuils selon le type de produit actuel"""
        thresholds = PRODUCT_THRESHOLDS.get(self.current_product, PRODUCT_THRESHOLDS['steel'])
        
        self.safety_limits = {
            "temperature_max": thresholds["temperature_max"],
            "temperature_warning": thresholds["temperature_warning"],
            "temperature_high": thresholds["temperature_high"],
            "pressure_max": thresholds["pressure_max"],
            "pressure_warning": thresholds["pressure_warning"],
            "pressure_high": thresholds["pressure_high"],
            "speed_max": thresholds["speed_max"],
            "speed_min": 0
        }
        
        self.production_target = {
            0: thresholds["production_target"],
            1: thresholds["production_target"],
            2: thresholds["production_target"]
        }
        
        return thresholds
    
    def set_product_type(self, product_type: str) -> str:
        """
        Change le type de produit à fabriquer
        
        Args:
            product_type: 'steel', 'aluminium', 'titanium', 'plastic'
        
        Returns:
            Nom du produit (pour confirmation)
        """
        if product_type not in PRODUCT_TYPES:
            product_type = 'steel'
        
        old_product = self.current_product
        self.current_product = product_type
        thresholds = self._update_thresholds_from_product()
        
        self.context_change_history.append({
            "step": self.current_step,
            "episode": getattr(self, '_current_episode', 0),
            "old_product": old_product,
            "new_product": product_type,
            "new_thresholds": thresholds
        })
        
        print(f"🏭 [ENV] Changement de production: {old_product} → {product_type}")
        print(f"   Température max: {thresholds['temperature_max']}°C")
        print(f"   Pression max: {thresholds['pressure_max']} bar")
        
        return product_type
    
    def get_product_type(self) -> str:
        """Retourne le type de produit actuel"""
        return self.current_product
    
    def get_current_thresholds(self) -> dict:
        """Retourne les seuils actuels"""
        return self.safety_limits.copy()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Enregistrer l'épisode pour le contexte
        self._current_episode = getattr(self, '_current_episode', 0) + 1
        
        self.current_step = 0
        self.total_production = 0
        self.total_reward = 0
        self.violations_log = []
        self.debug_counter = 0
        
        # Réinitialiser les machines avec des valeurs aléatoires selon le produit
        thresholds = self.safety_limits
        self.machine_states = {
            i: MachineState(
                temperature=20.0 + np.random.rand() * min(50, thresholds["temperature_warning"] / 10),
                pressure=5.0 + np.random.rand() * min(2, thresholds["pressure_warning"] / 5),
                speed=0.0,
                production_count=0,
                maintenance_needed=False
            ) for i in range(self.num_agents)
        }
        
        observations = {i: self._get_observation(i) for i in range(self.num_agents)}
        info = {"product_type": self.current_product, "thresholds": self.safety_limits}
        
        return observations, info
    
    def _get_observation(self, agent_id):
        """Calcule l'observation normalisée"""
        machine = self.machine_states[agent_id]
        limits = self.safety_limits
        
        obs = np.array([
            machine.temperature / limits["temperature_max"],
            machine.pressure / limits["pressure_max"],
            machine.speed / limits["speed_max"],
            min(1.0, machine.production_count / self.production_target.get(agent_id, 10)),
            float(machine.maintenance_needed),
            self.current_step / self.episode_length
        ], dtype=np.float32)
        
        # Sécurisation des valeurs
        obs = np.clip(obs, 0, 1)
        obs = np.nan_to_num(obs, nan=0.0)
        
        return obs
    
    def step(self, actions: Dict[int, int]):
        self.current_step += 1
        
        observations = {}
        rewards = {}
        dones = {}
        truncated = {}
        info = {}
        
        for agent_id, action in actions.items():
            self._apply_action(agent_id, action)
            
            violations = self._check_safety(agent_id)
            info[agent_id] = {"violations": violations}
            
            reward = self._calculate_reward(agent_id, violations)
            rewards[agent_id] = reward
            self.total_reward += reward
            
            observations[agent_id] = self._get_observation(agent_id)
            
            done = self.current_step >= self.episode_length
            dones[agent_id] = done
            truncated[agent_id] = False
        
        info["global"] = {
            "total_production": self.total_production,
            "total_reward": self.total_reward,
            "violations": len(self.violations_log),
            "step": self.current_step,
            "product_type": self.current_product
        }
        
        return observations, rewards, dones, truncated, info
    
    def _apply_action(self, agent_id: int, action: int):
        machine = self.machine_states[agent_id]
        limits = self.safety_limits
        dt = 0.01
        
        # Application de l'action
        if action == 0:  # reduce_speed
            machine.speed = max(0, machine.speed - 0.5)
        elif action == 1:  # maintain_speed
            pass
        elif action == 2:  # increase_speed
            max_speed = limits["speed_max"]
            machine.speed = min(max_speed, machine.speed + 0.5)
        elif action == 3:  # idle
            machine.speed = 0
        elif action == 4:  # emergency_stop
            machine.speed = 0
            machine.maintenance_needed = True
        
        # Physique simulée
        if machine.speed > 0:
            # La température augmente avec la vitesse
            machine.temperature += machine.speed * dt * 100
            machine.temperature = min(limits["temperature_max"], machine.temperature)
            
            # Pression pour le robot de peinture (agent 1)
            if agent_id == 1:
                machine.pressure = 5 + machine.speed * 0.3
                machine.pressure = min(limits["pressure_max"], machine.pressure)
            
            # Production (uniquement si conditions sûres)
            if (machine.temperature < limits["temperature_warning"] and 
                machine.pressure < limits["pressure_warning"]):
                production_increment = int(machine.speed * 0.5)
                if production_increment > 0:
                    machine.production_count += production_increment
                    self.total_production += production_increment
        else:
            # Refroidissement
            machine.temperature = max(20, machine.temperature - 1.0)
            if agent_id == 1:
                machine.pressure = max(0, machine.pressure - 0.2)
    
    def _check_safety(self, agent_id: int) -> list:
        machine = self.machine_states[agent_id]
        limits = self.safety_limits
        violations = []
        
        # Vérification température
        if machine.temperature >= limits["temperature_max"]:
            severity = min(1.0, (machine.temperature - limits["temperature_max"]) / 50)
            violations.append({
                "type": "temperature_critical",
                "value": machine.temperature,
                "limit": limits["temperature_max"],
                "severity": severity
            })
            self.violations_log.append(violations[-1])
        elif machine.temperature > limits["temperature_warning"]:
            severity = (machine.temperature - limits["temperature_warning"]) / 50
            violations.append({
                "type": "temperature_warning",
                "value": machine.temperature,
                "limit": limits["temperature_warning"],
                "severity": severity
            })
            self.violations_log.append(violations[-1])
        
        # Vérification pression (agent 1)
        if agent_id == 1:
            if machine.pressure >= limits["pressure_max"]:
                severity = min(1.0, (machine.pressure - limits["pressure_max"]) / 2)
                violations.append({
                    "type": "pressure_critical",
                    "value": machine.pressure,
                    "limit": limits["pressure_max"],
                    "severity": severity
                })
                self.violations_log.append(violations[-1])
            elif machine.pressure > limits["pressure_warning"]:
                severity = (machine.pressure - limits["pressure_warning"]) / 1.0
                violations.append({
                    "type": "pressure_warning",
                    "value": machine.pressure,
                    "limit": limits["pressure_warning"],
                    "severity": severity
                })
                self.violations_log.append(violations[-1])
        
        # Vérification maintenance
        if machine.maintenance_needed:
            violations.append({
                "type": "maintenance_required",
                "severity": 0.5
            })
        
        return violations
    
    def _calculate_reward(self, agent_id: int, violations: list) -> float:
        machine = self.machine_states[agent_id]
        reward = 0.0
        
        # Récompense pour production
        reward += machine.production_count * 0.3
        
        # Bonus pour vitesse positive
        if machine.speed > 0:
            reward += 1.0
        else:
            reward -= 2.0
        
        # Sécurité
        if len(violations) == 0:
            reward += 2.0
        else:
            for v in violations:
                if "critical" in v["type"]:
                    reward -= 100.0
                elif "warning" in v["type"]:
                    reward -= 50.0
                else:
                    reward -= 20.0
        
        # Pénalité stop inutile
        thresholds = self.safety_limits
        if machine.speed == 0 and not machine.maintenance_needed and machine.temperature < thresholds["temperature_warning"]:
            reward -= 3.0
        
        return reward
    
    def render(self):
        if self.render_mode == "human":
            print(f"\n--- Step {self.current_step} (Produit: {self.current_product}) ---")
            for i, machine in self.machine_states.items():
                print(f"Agent {i} ({self.machines[i]['name']}):")
                print(f"  Temperature: {machine.temperature:.1f}°C")
                print(f"  Pressure: {machine.pressure:.1f} bar")
                print(f"  Speed: {machine.speed:.1f}")
                print(f"  Production: {machine.production_count}")
    
    def close(self):
        pass
    
    def get_violations_summary(self):
        summary = {
            "total": len(self.violations_log),
            "by_type": {},
            "by_severity": {"low": 0, "medium": 0, "high": 0}
        }
        
        for v in self.violations_log:
            vtype = v["type"]
            summary["by_type"][vtype] = summary["by_type"].get(vtype, 0) + 1
            
            severity = v.get("severity", 0)
            if severity < 0.3:
                summary["by_severity"]["low"] += 1
            elif severity < 0.7:
                summary["by_severity"]["medium"] += 1
            else:
                summary["by_severity"]["high"] += 1
        
        return summary
    
    def get_context_change_history(self):
        """Retourne l'historique des changements de contexte"""
        return self.context_change_history