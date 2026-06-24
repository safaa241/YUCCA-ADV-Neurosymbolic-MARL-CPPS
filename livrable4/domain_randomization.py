"""
LIVRABLE 4 - Domain Randomization pour CPPS
Randomisation des paramètres pour robustesse sim-to-real

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random


class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    DISCRETE = "discrete"


@dataclass
class RandomizationRange:
    """Définit une plage de randomisation"""
    min_val: float
    max_val: float
    distribution: DistributionType = DistributionType.UNIFORM
    mean: Optional[float] = None
    std: Optional[float] = None
    
    def sample(self) -> float:
        """Échantillonne une valeur dans la plage"""
        if self.distribution == DistributionType.UNIFORM:
            return random.uniform(self.min_val, self.max_val)
        elif self.distribution == DistributionType.NORMAL:
            mean = self.mean if self.mean is not None else (self.min_val + self.max_val) / 2
            std = self.std if self.std is not None else (self.max_val - self.min_val) / 4
            # S'assurer que std est positif
            std = max(0.01, std)
            return np.clip(random.gauss(mean, std), self.min_val, self.max_val)
        elif self.distribution == DistributionType.LOG_NORMAL:
            mean = self.mean if self.mean is not None else (self.min_val + self.max_val) / 2
            std = self.std if self.std is not None else (self.max_val - self.min_val) / 4
            std = max(0.01, std)
            return np.clip(random.lognormvariate(mean, std), self.min_val, self.max_val)
        else:
            return random.uniform(self.min_val, self.max_val)


class DomainRandomizer:
    """
    Randomisation de domaine pour la robustesse sim-to-real
    
    Principe: Entraîner l'agent sur une large gamme de paramètres
    pour qu'il soit robuste aux variations du monde réel.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_default_ranges()
        
        # Historique des randomisations
        self.history = []
        self.current_params = {}
        
    def _init_default_ranges(self):
        """Initialise les plages de randomisation par défaut"""
        
        # Paramètres physiques
        self.physical_params = {
            'temperature_gain': RandomizationRange(0.8, 1.2),
            'pressure_gain': RandomizationRange(0.85, 1.15),
            'speed_gain': RandomizationRange(0.9, 1.1),
            'heating_coefficient': RandomizationRange(0.7, 1.3),
            'cooling_coefficient': RandomizationRange(0.7, 1.3),
        }
        
        # Paramètres de bruit (valeurs absolues, pas des pourcentages)
        self.noise_params = {
            'sensor_noise_temperature': RandomizationRange(0.0, 2.0),  # °C
            'sensor_noise_pressure': RandomizationRange(0.0, 0.2),     # bar
            'sensor_noise_speed': RandomizationRange(0.0, 0.1),        # m/s
            'actuator_noise': RandomizationRange(0.0, 0.03),
        }
        
        # Paramètres temporels
        self.temporal_params = {
            'latency_ms': RandomizationRange(0, 100),
            'dt_variation': RandomizationRange(0.8, 1.2),
            'step_jitter': RandomizationRange(0.0, 0.05),
        }
        
        # Paramètres environnementaux
        self.environment_params = {
            'ambient_temperature': RandomizationRange(15, 35),
            'supply_voltage': RandomizationRange(4.5, 5.5),
            'humidity': RandomizationRange(30, 80),
        }
        
        # Paramètres de calibration
        self.calibration_params = {
            'temp_offset': RandomizationRange(-2.0, 2.0),
            'pressure_offset': RandomizationRange(-0.5, 0.5),
            'speed_offset': RandomizationRange(-0.5, 0.5),
        }
    
    def randomize(self, state: np.ndarray, action: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Applique la randomisation à l'état et aux paramètres de l'environnement"""
        # Échantillonner les paramètres
        params = self.sample_parameters()
        self.current_params = params
        self.history.append(params)
        
        # Copier l'état
        randomized = state.copy()
        
        # Dénormalisation temporaire pour appliquer les gains
        temp_actual = state[0] * 850
        pressure_actual = state[1] * 10
        speed_actual = state[2] * 10
        
        # Appliquer les gains physiques
        temp_actual *= params.get('temperature_gain', 1.0)
        pressure_actual *= params.get('pressure_gain', 1.0)
        speed_actual *= params.get('speed_gain', 1.0)
        
        # Ajouter les offsets de calibration
        temp_actual += params.get('temp_offset', 0.0)
        pressure_actual += params.get('pressure_offset', 0.0)
        speed_actual += params.get('speed_offset', 0.0)
        
        # Ajouter le bruit des capteurs (avec valeur absolue pour éviter scale < 0)
        temp_noise_std = max(0, params.get('sensor_noise_temperature', 0.5))
        pressure_noise_std = max(0, params.get('sensor_noise_pressure', 0.05))
        speed_noise_std = max(0, params.get('sensor_noise_speed', 0.03))
        
        temp_noise = np.random.normal(0, temp_noise_std)
        pressure_noise = np.random.normal(0, pressure_noise_std)
        speed_noise = np.random.normal(0, speed_noise_std)
        
        temp_actual += temp_noise
        pressure_actual += pressure_noise
        speed_actual += speed_noise
        
        # Appliquer la randomisation temporelle
        dt_factor = params.get('dt_variation', 1.0)
        
        # Renormaliser en s'assurant de rester dans les limites
        randomized[0] = np.clip(temp_actual / 850, 0, 1)
        randomized[1] = np.clip(pressure_actual / 10, 0, 1)
        randomized[2] = np.clip(speed_actual / 10, 0, 1)
        
        # Ajouter du bruit sur le temps
        time_jitter = params.get('step_jitter', 0.0) * np.random.randn()
        randomized[5] = np.clip(randomized[5] + time_jitter, 0, 1)
        
        return randomized, params
    
    def sample_parameters(self) -> Dict:
        """Échantillonne tous les paramètres"""
        params = {}
        
        for category in [self.physical_params, self.noise_params, 
                        self.temporal_params, self.environment_params,
                        self.calibration_params]:
            for name, range_def in category.items():
                params[name] = range_def.sample()
        
        return params
    
    def get_physics_modifiers(self) -> Dict:
        """Retourne les modificateurs physiques pour l'environnement"""
        params = self.current_params
        return {
            'heating_coefficient': params.get('heating_coefficient', 1.0),
            'cooling_coefficient': params.get('cooling_coefficient', 1.0),
            'dt_multiplier': params.get('dt_variation', 1.0),
            'ambient_temp': params.get('ambient_temperature', 25.0),
        }
    
    def get_sensor_model(self) -> Dict:
        """Retourne le modèle de capteur"""
        params = self.current_params
        return {
            'temperature_bias': params.get('temp_offset', 0.0),
            'pressure_bias': params.get('pressure_offset', 0.0),
            'temperature_noise': max(0, params.get('sensor_noise_temperature', 0.5)),
            'pressure_noise': max(0, params.get('sensor_noise_pressure', 0.05)),
            'speed_noise': max(0, params.get('sensor_noise_speed', 0.03)),
        }
    
    def get_latency(self) -> float:
        """Retourne la latence simulée en secondes"""
        return self.current_params.get('latency_ms', 0) / 1000.0
    
    def reset(self):
        """Réinitialise l'historique"""
        self.history = []
        self.current_params = {}
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques de randomisation"""
        if not self.history:
            return {}
        
        stats = {}
        for param in self.history[0].keys():
            values = [h[param] for h in self.history]
            stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats


class HardwareSimulator:
    """
    Simulateur matériel pour HIL - Simule le comportement de composants réels
    """
    
    def __init__(self, use_real_hardware: bool = False):
        self.use_real_hardware = use_real_hardware
        self.hardware_connected = False
        
        # Modèles de composants
        self.sensor_models = {
            'temperature': self._simulate_temperature_sensor,
            'pressure': self._simulate_pressure_sensor,
            'speed': self._simulate_speed_sensor,
        }
        
        # Filtres pour lisser les mesures
        self.filters = {
            'temperature': [],
            'pressure': [],
            'speed': []
        }
        self.filter_size = 5
        self.last_valid_values = {}
        
    def _simulate_temperature_sensor(self, real_temp: float, params: Dict) -> float:
        """Simule un capteur de température réel"""
        # Inertie thermique
        inertia = params.get('thermal_inertia', 0.1)
        
        # Bruit de mesure (positif)
        noise_std = max(0.1, abs(params.get('temp_noise_std', 0.5)))
        noise = np.random.normal(0, noise_std)
        
        # Dérive possible
        drift = params.get('temp_drift', 0.0) * np.sin(np.random.rand() * np.pi)
        
        measured = real_temp * (1 - inertia) + noise + drift
        
        # Appliquer filtre médian
        self.filters['temperature'].append(measured)
        if len(self.filters['temperature']) > self.filter_size:
            self.filters['temperature'].pop(0)
        
        return np.median(self.filters['temperature']) if self.filters['temperature'] else measured
    
    def _simulate_pressure_sensor(self, real_pressure: float, params: Dict) -> float:
        """Simule un capteur de pression réel"""
        # Hystérésis
        hysteresis = params.get('pressure_hysteresis', 0.02)
        
        # Bruit (positif)
        noise_std = max(0.01, abs(params.get('pressure_noise_std', 0.05)))
        noise = np.random.normal(0, noise_std)
        
        measured = real_pressure * (1 + hysteresis * np.random.randn()) + noise
        
        self.filters['pressure'].append(measured)
        if len(self.filters['pressure']) > self.filter_size:
            self.filters['pressure'].pop(0)
        
        return np.median(self.filters['pressure']) if self.filters['pressure'] else measured
    
    def _simulate_speed_sensor(self, real_speed: float, params: Dict) -> float:
        """Simule un capteur de vitesse réel"""
        # Quantification
        quantization = params.get('speed_quantization', 0.1)
        
        # Bruit (positif)
        noise_std = max(0.01, abs(params.get('speed_noise_std', 0.03)))
        noise = np.random.normal(0, noise_std)
        
        measured = np.round(real_speed / quantization) * quantization + noise
        
        self.filters['speed'].append(measured)
        if len(self.filters['speed']) > self.filter_size:
            self.filters['speed'].pop(0)
        
        return np.median(self.filters['speed']) if self.filters['speed'] else measured
    
    def measure(self, true_state: Dict, params: Dict) -> Dict:
        """
        Simule la mesure d'un capteur réel
        
        Args:
            true_state: État réel du système
            params: Paramètres de simulation
            
        Returns:
            Mesures bruitées
        """
        measurements = {}
        
        for key, simulator in self.sensor_models.items():
            if key in true_state:
                measurements[key] = simulator(true_state[key], params)
        
        # Latence de mesure
        latency = params.get('measurement_latency', 0.01)
        if latency > 0:
            # Simuler latence (serait géré par le buffer dans la réalité)
            pass
        
        return measurements


class ActuatorSimulator:
    """
    Simulateur d'actionneurs pour HIL
    """
    
    def __init__(self):
        self.last_command = 0
        self.actuator_state = 0
        
    def apply_command(self, command: int, params: Dict) -> int:
        """
        Applique une commande à l'actionneur
        
        Args:
            command: Commande (0-4)
            params: Paramètres de simulation
            
        Returns:
            Commande effective après simulation
        """
        # Délai de réponse
        response_delay = params.get('actuator_delay', 0)
        
        # Non-linéarité
        nonlinearity = params.get('actuator_nonlinearity', 0.05)
        
        # Bruit d'exécution (positif)
        noise_std = max(0, abs(params.get('actuator_noise', 0.01)))
        noise = np.random.normal(0, noise_std)
        
        # Appliquer les effets
        if response_delay > 0:
            # Simuler délai
            effective_command = self.last_command
        else:
            effective_command = command
        
        # Ajouter bruit
        if np.random.rand() < nonlinearity:
            effective_command = np.clip(effective_command + int(np.sign(noise)), 0, 4)
        
        self.last_command = command
        self.actuator_state = effective_command
        
        return effective_command