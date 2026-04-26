# Jumeau numérique (3 machines, physique simulée)
# Ce qu'il produit: États, récompenses, violations de sûreté, etc.


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass


@dataclass

# Structure de données pour l'état d'une machine
class MachineState:
    temperature: float = 20.0 # Température initiale (°C)
    pressure: float = 5.0 # Pression initiale (bar)
    speed: float = 0.0 # Vitesse initiale (pièces/minute)
    production_count: int = 0 # Nombre de pièces produites
    maintenance_needed: bool = False # Indique si la machine nécessite une maintenance

# Environnement de production CPPS
class CPPSProductionEnv(gym.Env):
    
    metadata = {'render_modes': ['human', 'rgb_array']} # Modes de rendu disponibles
    # Version 1.0 - Initialisation de l'environnement avec des machines de base et des règles simples
    def __init__(self, num_agents=3, episode_length=500, render_mode=None):
        
        super().__init__() # Appel du constructeur parent
        
        self.num_agents = num_agents # Nombre d'agents (machines)
        self.episode_length = episode_length # Nombre de steps par épisode
        self.render_mode = render_mode # Mode de rendu (human, rgb_array, etc.)
        
        # Configuration des machines
        self.machines = {
            0: {"name": "Welding Robot", "max_temp": 850, "max_speed": 10}, # Robot de soudure
            1: {"name": "Painting Robot", "max_pressure": 10, "max_speed": 10}, # Robot de peinture
            2: {"name": "Quality Control", "max_precision": 100, "max_speed": 5} # Contrôle qualité
        }
        
        # États et actions
        self.action_spaces = { # Chaque agent peut choisir parmi 5 actions
            i: spaces.Discrete(5) for i in range(num_agents) 
        }
        # Actions: 0=décrémenter, 1=maintenir, 2=incrémenter, 3=idle, 4=stop
        
        self.observation_spaces = { # Chaque agent observe 6 variables normalisées entre 0 et 1
            i: spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) 
            for i in range(num_agents) 
        }
        # Observation: [temperature/max_temp, pressure/max_pressure, speed/max_speed,
        #               production_count/target, maintenance_needed, time_step/episode_length]
        
        
        self.machine_states = {i: MachineState() for i in range(num_agents)} # Initialisation des états des machines
        
        # Paramètres de simulation
        self.current_step = 0 # Compteur de steps dans l'épisode
        self.total_production = 0 # Compteur de pièces produites
        self.total_reward = 0 # Compteur de récompenses accumulées
        self.violations_log = [] # Journal des violations de sûreté (température, pression, etc.)
        
        # Cibles de production
        self.production_target = {0: 10, 1: 10, 2: 10}  # pièces/minute
        
        # Limites de sûreté
        self.safety_limits = {
            "temperature_max": 850,
            "temperature_min": 0,
            "pressure_max": 10,
            "pressure_min": 0,
            "speed_max": 10,
            "speed_min": 0
        }
        
        # Debug: compteur pour affichage
        self.debug_counter = 0
    
    # RÉINITIALISATION DE L'ENVIRONNEMENT
    def reset(self, seed=None, options=None): 
        
        super().reset(seed=seed) # Appel du reset du parent pour la gestion des seeds
        
        self.current_step = 0 # Réinitialiser le compteur de steps
        self.total_production = 0 # Réinitialiser le compteur de production
        self.total_reward = 0 # Réinitialiser le compteur de récompenses
        self.violations_log = []
        self.debug_counter = 0
        
        # Réinitialiser les machines avec des valeurs aléatoires
        self.machine_states = {
            i: MachineState(
                temperature=20.0 + np.random.rand() * 50, # Température initiale entre 20 et 70°C
                pressure=5.0 + np.random.rand() * 2, # Pression initiale entre 5 et 7 bar
                speed=0.0, # Vitesse initiale à 0
                production_count=0, # Compteur de production à 0
                maintenance_needed=False # Pas de maintenance nécessaire au départ
            ) for i in range(self.num_agents) # Réinitialisation pour chaque agent
        }
        
        # Observations initiales
        observations = {i: self._get_observation(i) for i in range(self.num_agents)}
        info = {} # Informations supplémentaires (vide pour l'instant)
        
        return observations, info
    # CALCUL DE L'OBSERVATION POUR UN AGENT
    def _get_observation(self, agent_id):
        
        machine = self.machine_states[agent_id] # Récupérer l'état de la machine pour l'agent donné
        # Normaliser les observations par rapport aux limites de sûreté et aux cibles de production
        obs = np.array([
            machine.temperature / self.safety_limits["temperature_max"], 
            machine.pressure / self.safety_limits["pressure_max"], 
            machine.speed / self.safety_limits["speed_max"],# Normalisation de la vitesse
            min(1.0, machine.production_count / self.production_target.get(agent_id, 10)), # Normalisation de la production par rapport à la cible
            float(machine.maintenance_needed), # 1.0 si maintenance nécessaire, sinon 0.0
            self.current_step / self.episode_length # Normalisation du temps écoulé dans l'épisode
        ], dtype=np.float32) # Normalisation des observations pour les rendre compatibles avec les réseaux de neurones
        
        return obs
    # EXÉCUTION D'UNE ÉTAPE DE SIMULATION
    def step(self, actions: Dict[int, int]):

        self.current_step += 1 # Incrémenter le compteur de steps
        
        observations = {} # Observations pour chaque agent
        rewards = {} # Récompenses pour chaque agent
        dones = {} # Indique si l'épisode est terminé pour chaque agent
        truncated = {} # Indique si l'épisode a été tronqué (terminé prématurément)
        info = {} # Informations supplémentaires pour chaque agent
        
        # Traiter les actions de chaque agent
        for agent_id, action in actions.items():
            self._apply_action(agent_id, action)
            
            # Vérifier les violations de sûreté
            violations = self._check_safety(agent_id)
            info[agent_id] = {"violations": violations}
            
            # Calculer les récompenses
            reward = self._calculate_reward(agent_id, violations)
            rewards[agent_id] = reward
            self.total_reward += reward
            
            # Observations
            observations[agent_id] = self._get_observation(agent_id)
            
            # Condition de fin
            done = self.current_step >= self.episode_length
            dones[agent_id] = done
            truncated[agent_id] = False
        
        # Informations globales
        info["global"] = {
            "total_production": self.total_production,
            "total_reward": self.total_reward,
            "violations": len(self.violations_log),
            "step": self.current_step
        }
        
        return observations, rewards, dones, truncated, info
    # APPLICATION D'UNE ACTION POUR UN AGENT
    def _apply_action(self, agent_id: int, action: int):
        
        machine = self.machine_states[agent_id]
        
        dt = 0.01  # Pas de temps simulé (secondes)
        
        # ========== 1. APPLIQUER L'ACTION ==========
        if action == 0:  # Décrémenter la vitesse
            machine.speed = max(0, machine.speed - 0.5)
        elif action == 1:  # Maintenir
            pass
        elif action == 2:  # Augmenter la vitesse
            max_speed = self.machines[agent_id]["max_speed"]
            machine.speed = min(max_speed, machine.speed + 0.5)
        elif action == 3:  # Mode idle
            machine.speed = 0
        elif action == 4:  # Stop d'urgence
            machine.speed = 0
            machine.maintenance_needed = True
        
        # ========== 2. PHYSIQUE SIMULÉE (CORRIGÉE) ==========
        if machine.speed > 0:
            # La température AUGMENTE avec la vitesse
            # Facteur x100 pour que la température atteigne 800°C pendant l'épisode
            machine.temperature += machine.speed * dt * 100
            machine.temperature = min(self.safety_limits["temperature_max"], machine.temperature)
            
            # Pression pour le robot de peinture (agent 1)
            if agent_id == 1:
                machine.pressure = 5 + machine.speed * 0.3
                machine.pressure = min(self.safety_limits["pressure_max"], machine.pressure)
            
            # Production (uniquement si conditions raisonnablement sûres)
            # Seuils plus réalistes pour permettre la production
            if machine.temperature < 800 and machine.pressure < 9.5:
                production_increment = int(machine.speed * 0.5)
                if production_increment > 0:
                    machine.production_count += production_increment
                    self.total_production += production_increment
                    
                    # DEBUG: Afficher quand on produit (tous les 100 steps)
                    if self.current_step % 100 == 0 and agent_id == 0:
                        print(f"[PROD] Step {self.current_step}: T={machine.temperature:.1f}°C, "
                              f"speed={machine.speed:.1f}, prod={production_increment}")
        else:
            # Refroidissement (plus rapide quand on arrête)
            machine.temperature = max(20, machine.temperature - 1.0)
            if agent_id == 1:
                machine.pressure = max(0, machine.pressure - 0.2)
        
        # ========== 3. DEBUG: Afficher la température périodiquement ==========
        if self.current_step % 100 == 0 and agent_id == 0:
            print(f"[ENV] Step {self.current_step}: T={machine.temperature:.1f}°C, "
                  f"speed={machine.speed:.1f}, action={action}")
    # VÉRIFICATION DES VIOLATIONS DE SÛRETÉ
    def _check_safety(self, agent_id: int) -> list:
        
        machine = self.machine_states[agent_id]
        violations = []
        
        # Vérifier température (CRITIQUE)
        if machine.temperature >= self.safety_limits["temperature_max"]:
            severity = min(1.0, (machine.temperature - self.safety_limits["temperature_max"]) / 50)
            violations.append({
                "type": "temperature_critical",
                "value": machine.temperature,
                "limit": self.safety_limits["temperature_max"],
                "severity": severity
            })
            self.violations_log.append(violations[-1])
        elif machine.temperature > 800:  # Alerte température élevée
            severity = (machine.temperature - 800) / 50
            violations.append({
                "type": "temperature_warning",
                "value": machine.temperature,
                "limit": 800,
                "severity": severity
            })
            self.violations_log.append(violations[-1])
        
        # Vérifier pression (pour agent 1)
        if agent_id == 1 and machine.pressure >= self.safety_limits["pressure_max"]:
            severity = min(1.0, (machine.pressure - self.safety_limits["pressure_max"]) / 2)
            violations.append({
                "type": "pressure_critical",
                "value": machine.pressure,
                "limit": self.safety_limits["pressure_max"],
                "severity": severity
            })
            self.violations_log.append(violations[-1])
        elif agent_id == 1 and machine.pressure > 9.0:
            severity = (machine.pressure - 9.0) / 1.0
            violations.append({
                "type": "pressure_warning",
                "value": machine.pressure,
                "limit": 9.0,
                "severity": severity
            })
            self.violations_log.append(violations[-1])
        
        # Vérifier maintenance
        if machine.maintenance_needed:
            violations.append({
                "type": "maintenance_required",
                "severity": 0.5
            })
        
        return violations
    
    # CALCUL DE LA RÉCOMPENSE POUR UN AGENT
    def _calculate_reward(self, agent_id: int, violations: list) -> float:
        machine = self.machine_states[agent_id]
        reward = 0.0
    
    # RÉCOMPENSE POUR PRODUCTION (forte)
        reward += machine.production_count * 0.3
    
    # BONUS POUR VITESSE POSITIVE
        if machine.speed > 0:
            reward += 1.0
        else:
            reward -= 2.0  # Pénalité forte pour rester à l'arrêt
    
    # SÉCURITÉ
        if len(violations) == 0:
            reward += 2.0
        else:
            for v in violations:
                reward -= 50.0
    
    # PÉNALITÉ STOP INUTILE
        if machine.speed == 0 and not machine.maintenance_needed and machine.temperature < 800:
            reward -= 3.0
    
        return reward

    # RENDU DE L'ENVIRONNEMENT
    def render(self):
        
        if self.render_mode == "human":
            print(f"\n--- Step {self.current_step} ---")
            for i, machine in self.machine_states.items():
                print(f"Agent {i} ({self.machines[i]['name']}):")
                print(f"  Temperature: {machine.temperature:.1f}°C")
                print(f"  Pressure: {machine.pressure:.1f} bar")
                print(f"  Speed: {machine.speed:.1f}")
                print(f"  Production: {machine.production_count}")

    # fermeture de l'environnement (vide pour l'instant)
    def close(self):
        
        pass
    
    # RÉSUMÉ DES VIOLATIONS
    def get_violations_summary(self):

        summary = {
            "total": len(self.violations_log),
            "by_type": {},
            "by_severity": {"low": 0, "medium": 0, "high": 0}
        }
        
        # Compter les violations par type et par sévérité
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