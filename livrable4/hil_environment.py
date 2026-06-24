"""
LIVRABLE 4 - Environnement Hardware-in-the-Loop (HIL)
Interface entre la simulation et le matériel réel

Version étendue avec simulation réaliste de la communication hardware
et génération de logs de preuve pour la soutenance.

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
"""

import numpy as np
import time
import threading
import queue
import random
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

# ============================================================================
# UTILITAIRE POUR LA SÉRIALISATION JSON
# ============================================================================

def convert_to_serializable(obj):
    """
    Convertit les types non sérialisables (numpy, etc.) en types Python natifs
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj
    
# Tentative d'import des bibliothèques hardware (optionnel)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ PySerial non installé - Utilisation du mode simulation")

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False
    print("⚠️ RPi.GPIO non disponible - Utilisation du mode simulation")


class HardwareMode(Enum):
    """Modes de fonctionnement HIL"""
    SIMULATION = "simulation"      # 100% logiciel
    HIL = "hil"                    # Hybride (simulation + hardware)
    REAL = "real"                  # 100% matériel réel


@dataclass
class HardwareConfig:
    """Configuration du matériel HIL"""
    mode: HardwareMode = HardwareMode.SIMULATION
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    gpio_pins: Dict[str, int] = None
    sampling_rate: float = 100.0
    use_mqtt: bool = False
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    simulate_latency: bool = True
    simulate_errors: bool = True
    log_hardware: bool = True
    
    def __post_init__(self):
        if self.gpio_pins is None:
            self.gpio_pins = {
                'temp_sensor': 4,
                'pressure_sensor': 17,
                'motor_pwm': 18,
                'led_status': 27,
                'led_error': 22
            }


class HardwareSimulator:
    """
    Simulateur matériel réaliste pour tests HIL sans hardware réel
    
    Simule:
    - Communication série (Arduino)
    - GPIO (Raspberry Pi)
    - Capteurs avec bruit réaliste
    - Latences et erreurs
    """
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.is_connected = False
        self.running = False
        
        # Valeurs simulées des capteurs
        self.sensor_values = {
            'temperature': 25.0,
            'pressure': 5.0,
            'speed': 0.0,
            'vibration': 0.0,
            'current': 0.0
        }
        
        # État des actionneurs
        self.actuator_state = {
            'motor_pwm': 0,
            'led_status': False,
            'led_error': False,
            'last_command': 0
        }
        
        # Métriques de simulation
        self.latency_history = []
        self.error_history = []
        self.communication_log = []
        
        # Paramètres de bruit réaliste
        self.noise_params = {
            'temperature_std': 0.5,
            'pressure_std': 0.05,
            'speed_std': 0.1,
            'latency_mean': 0.012,  # 12ms
            'latency_std': 0.003,   # 3ms
            'packet_loss_rate': 0.01  # 1%
        }
        
        # Historique pour les dérives
        self.temp_drift = 0.0
        self.pressure_drift = 0.0
        
        # CORRECTION: Initialiser last_valid_values
        self.last_valid_values = {
            'temperature': 25.0,
            'pressure': 5.0,
            'speed': 0.0
        }
        
    def connect(self) -> bool:
        """Simule la connexion au matériel"""
        self.is_connected = True
        self.running = True
        
        log_msg = f"[HIL_SIM] Connexion établie - Port: {self.config.serial_port}, Baud: {self.config.baud_rate}"
        self.communication_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'CONNECTION',
            'message': log_msg
        })
        
        if self.config.log_hardware:
            print(f"✅ {log_msg}")
            print(f"[HIL_SIM] GPIO initialisé - Pins: {self.config.gpio_pins}")
            print(f"[HIL_SIM] Capteurs simulés: DS18B20 (température), BMP180 (pression)")
        
        return True
    
    def disconnect(self):
        """Simule la déconnexion"""
        self.running = False
        self.is_connected = False
        print(f"[HIL_SIM] Déconnexion du matériel")
    
    def read_sensor(self, sensor_name: str) -> float:
        """Lit un capteur simulé"""
        if not self.is_connected:
            return self.sensor_values.get(sensor_name, 0)
        
        real_value = self.sensor_values.get(sensor_name, 0)
        noise_std = self.noise_params.get(f'{sensor_name}_std', 0.1)
        noise = np.random.normal(0, noise_std)
        measured = real_value + noise
    
        measured = self.validate_sensor_reading(measured, sensor_name)
    
        return max(0, measured)

    def validate_sensor_reading(self, value: float, sensor_type: str) -> float:
        """Valide et corrige les lectures aberrantes des capteurs"""
    
        # Seuils physiques
        limits = {
            'temperature': (0, 1000, 50),   # min, max, variation_max
            'pressure': (0, 15, 2),
            'speed': (0, 10, 3)
        }
    
        min_val, max_val, max_var = limits.get(sensor_type, (0, 100, 10))
    
        # 1. Vérification plage physique
        if value < min_val or value > max_val:
            print(f"[ANOMALIE] {sensor_type}: {value} hors plage [{min_val},{max_val}]")
            return self.last_valid_values.get(sensor_type, (min_val+max_val)/2)
    
        # 2. Vérification variation trop rapide
        last_val = self.last_valid_values.get(sensor_type, value)
        if abs(value - last_val) > max_var:
            print(f"[ANOMALIE] {sensor_type}: variation {abs(value-last_val):.1f} > {max_var}")
            return last_val
    
        # 3. Valeur valide
        self.last_valid_values[sensor_type] = value
        return value

    def read_all_sensors(self) -> Dict[str, float]:
        """Lit tous les capteurs"""
        return {
            'temperature': self.read_sensor('temperature'),
            'pressure': self.read_sensor('pressure'),
            'speed': self.read_sensor('speed'),
            'vibration': self.read_sensor('vibration'),
            'current': self.read_sensor('current')
        }
    
    def write_actuator(self, actuator: str, value: Any) -> bool:
        """
        Simule l'envoi d'une commande à un actionneur
        
        Args:
            actuator: Nom de l'actionneur ('motor', 'led_status', 'led_error')
            value: Commande à envoyer
            
        Returns:
            Succès de l'envoi
        """
        if not self.is_connected:
            return False
        
        # Simuler la latence
        if self.config.simulate_latency:
            latency = np.random.normal(
                self.noise_params['latency_mean'],
                self.noise_params['latency_std']
            )
            latency = max(0, latency)
            self.latency_history.append(latency)
            time.sleep(latency / 10)  # Latence réduite pour ne pas ralentir la simu
        
        # Mettre à jour l'état
        if actuator == 'motor':
            self.actuator_state['motor_pwm'] = value
            self.actuator_state['last_command'] = value
        elif actuator == 'led_status':
            self.actuator_state['led_status'] = value
        elif actuator == 'led_error':
            self.actuator_state['led_error'] = value
        
        # Log de la commande
        if self.config.log_hardware:
            log_msg = f"[HIL_SIM] Commande envoyée - {actuator}={value}"
            self.communication_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'COMMAND',
                'message': log_msg
            })
        
        return True
    
    def update_sensor_from_simulation(self, sim_state: Dict[str, float]):
        """
        Met à jour les valeurs simulées des capteurs à partir de la simulation
        
        Args:
            sim_state: État de la simulation (température, pression, vitesse)
        """
        # Mise à jour avec un facteur d'inertie (simule la physique réelle)
        alpha = 0.85  # Facteur d'inertie (plus proche de 1 = plus lent)
        
        if 'temperature' in sim_state:
            self.sensor_values['temperature'] = alpha * self.sensor_values['temperature'] + (1 - alpha) * sim_state['temperature']
        
        if 'pressure' in sim_state:
            self.sensor_values['pressure'] = alpha * self.sensor_values['pressure'] + (1 - alpha) * sim_state['pressure']
        
        if 'speed' in sim_state:
            self.sensor_values['speed'] = alpha * self.sensor_values['speed'] + (1 - alpha) * sim_state['speed']
    
    def get_sim_to_real_gap(self, sim_state: Dict[str, float]) -> Dict[str, float]:
        """
        Calcule l'écart entre simulation et capteurs simulés
        
        Returns:
            Dictionnaire des écarts par capteur
        """
        gaps = {}
        for key in ['temperature', 'pressure', 'speed']:
            if key in sim_state and key in self.sensor_values:
                if key == 'temperature':
                    max_val = 850
                elif key == 'pressure':
                    max_val = 10
                else:
                    max_val = 10
                
                gap = abs(self.sensor_values[key] - sim_state[key]) / max_val
                gaps[key] = gap
                self.error_history.append(gap)
        
        return gaps
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques de la simulation HIL"""
        return {
            'total_commands': len([l for l in self.communication_log if l['type'] == 'COMMAND']),
            'total_errors': len([l for l in self.communication_log if l['type'] == 'ERROR']),
            'mean_latency': np.mean(self.latency_history) if self.latency_history else 0,
            'std_latency': np.std(self.latency_history) if self.latency_history else 0,
            'mean_error': np.mean(self.error_history) if self.error_history else 0,
            'max_error': np.max(self.error_history) if self.error_history else 0,
            'is_connected': self.is_connected,
            'sensor_values': self.sensor_values.copy(),
            'actuator_state': self.actuator_state.copy()
        }
    
    def export_logs(self, filepath: str):
        """Exporte les logs de communication"""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'mode': self.config.mode.value,
                'serial_port': self.config.serial_port,
                'baud_rate': self.config.baud_rate
            },
            'statistics': self.get_statistics(),
            'logs': self.communication_log[-100:]  # Derniers 100 logs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Logs HIL exportés dans {filepath}")


class HardwareInterface:
    """
    Interface matérielle pour HIL avec support réel et simulé
    
    Supporte:
    - Communication série (Arduino) - réel ou simulé
    - GPIO (Raspberry Pi) - réel ou simulé
    - MQTT (communication réseau)
    """
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.simulator = HardwareSimulator(config) if config.mode != HardwareMode.REAL else None
        self.serial_conn = None
        self.mqtt_client = None
        self.running = False
        self.data_queue = queue.Queue()
        
        self._init_hardware()
    
    def _init_hardware(self):
        """Initialise la connexion matérielle ou le simulateur"""
        
        if self.config.mode == HardwareMode.SIMULATION:
            print("🔬 Mode SIMULATION - Utilisation du simulateur HIL")
            self.simulator.connect()
            return
        
        if self.config.mode == HardwareMode.HIL:
            print("🖥️ Mode HIL - Tentative de connexion au matériel")
            
            # Tentative de connexion au matériel réel
            hardware_detected = False
            
            # Test série (Arduino)
            if SERIAL_AVAILABLE:
                try:
                    self.serial_conn = serial.Serial(
                        port=self.config.serial_port,
                        baudrate=self.config.baud_rate,
                        timeout=1
                    )
                    hardware_detected = True
                    print(f"✅ Connexion série établie sur {self.config.serial_port}")
                except Exception as e:
                    print(f"⚠️ Connexion série impossible: {e}")
                    print("   → Passage en mode simulation HIL")
            
            # Test GPIO (Raspberry Pi)
            if RPI_AVAILABLE and not hardware_detected:
                try:
                    GPIO.setmode(GPIO.BCM)
                    for pin in self.config.gpio_pins.values():
                        GPIO.setup(pin, GPIO.OUT)
                    hardware_detected = True
                    print("✅ GPIO Raspberry Pi initialisé")
                except Exception as e:
                    print(f"⚠️ GPIO impossible: {e}")
            
            # Si aucun hardware détecté, on utilise le simulateur
            if not hardware_detected:
                print("⚠️ Aucun matériel détecté - Utilisation du simulateur HIL")
                if self.simulator:
                    self.simulator.connect()
        
        elif self.config.mode == HardwareMode.REAL:
            print("🏭 Mode RÉEL - Tentative de déploiement complet")
            if SERIAL_AVAILABLE:
                try:
                    self.serial_conn = serial.Serial(
                        port=self.config.serial_port,
                        baudrate=self.config.baud_rate,
                        timeout=0.5
                    )
                    print(f"✅ [REAL] Connexion série sur {self.config.serial_port}")
                except Exception as e:
                    print(f"❌ [REAL] Échec connexion série: {e}")
            
            if RPI_AVAILABLE:
                try:
                    GPIO.setmode(GPIO.BCM)
                    for pin in self.config.gpio_pins.values():
                        GPIO.setup(pin, GPIO.OUT)
                    print("✅ [REAL] GPIO initialisé")
                except Exception as e:
                    print(f"❌ [REAL] Échec GPIO: {e}")
    
    def read_sensors(self) -> Dict[str, float]:
        """
        Lit les valeurs des capteurs (réels ou simulés)
        
        Returns:
            Dictionnaire des mesures
        """
        # Mode simulation
        if self.config.mode == HardwareMode.SIMULATION and self.simulator:
            return self.simulator.read_all_sensors()
        
        # Mode HIL ou REAL - tentative de lecture réelle
        measurements = {}
        
        # Lecture via série (Arduino)
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(b'READ\n')
                line = self.serial_conn.readline().decode().strip()
                if line:
                    data = json.loads(line)
                    measurements.update(data)
            except Exception as e:
                print(f"⚠️ Erreur lecture série: {e}")
        
        # Lecture GPIO (Raspberry Pi) - simulation si pas de capteur
        if RPI_AVAILABLE and not measurements:
            try:
                measurements['temperature'] = 25.0 + np.random.randn() * 2
                measurements['pressure'] = 5.0 + np.random.randn() * 0.5
            except Exception as e:
                print(f"⚠️ Erreur lecture GPIO: {e}")
        
        # Si aucune mesure réelle, utiliser le simulateur
        if not measurements and self.simulator:
            measurements = self.simulator.read_all_sensors()
            print("ℹ️ Utilisation des capteurs simulés (fallback)")
        
        return measurements
    
    def write_actuator(self, action: int):
        """
        Envoie une commande à l'actionneur (réel ou simulé)
        
        Args:
            action: Action (0-4)
        """
        # Mapping action -> PWM
        pwm_values = {
            0: 0,     # reduce_speed
            1: 128,   # maintain_speed
            2: 255,   # increase_speed
            3: 0,     # idle
            4: 0      # emergency_stop
        }
        
        pwm = pwm_values.get(action, 0)
        
        # En mode simulation, juste mettre à jour le simulateur
        if self.config.mode == HardwareMode.SIMULATION and self.simulator:
            self.simulator.write_actuator('motor', pwm)
            self.simulator.write_actuator('led_status', action != 4)
            self.simulator.write_actuator('led_error', action == 4)
            
            # Log pour la preuve
            action_names = {0: "REDUCE", 1: "MAINTAIN", 2: "INCREASE", 3: "IDLE", 4: "STOP"}
            print(f"[HIL_SIM] Action exécutée: {action_names.get(action, 'UNKNOWN')} (PWM={pwm})")
            return
        
        # Mode HIL/REAL - tentative d'envoi réel
        
        # Commande via série
        if self.serial_conn and self.serial_conn.is_open:
            try:
                command = json.dumps({'action': action, 'pwm': pwm})
                self.serial_conn.write(f"{command}\n".encode())
                if self.config.log_hardware:
                    print(f"[HARDWARE] Commande série envoyée: {command}")
            except Exception as e:
                print(f"⚠️ Erreur écriture série: {e}")
        
        # Commande via GPIO (PWM)
        if RPI_AVAILABLE:
            try:
                motor_pin = self.config.gpio_pins.get('motor_pwm')
                if motor_pin:
                    p = GPIO.PWM(motor_pin, 1000)
                    p.start(pwm / 255 * 100)
                
                # LED de statut
                status_pin = self.config.gpio_pins.get('led_status')
                if status_pin:
                    GPIO.output(status_pin, action != 4)
                
                # LED d'erreur
                error_pin = self.config.gpio_pins.get('led_error')
                if error_pin:
                    GPIO.output(error_pin, action == 4)
                    
                if self.config.log_hardware:
                    print(f"[HARDWARE] GPIO mis à jour: PWM={pwm}, LED_status={action!=4}")
            except Exception as e:
                print(f"⚠️ Erreur écriture GPIO: {e}")
        
        # Fallback vers simulateur
        elif self.simulator:
            self.simulator.write_actuator('motor', pwm)
    
    def update_from_simulation(self, sim_state: Dict[str, float]):
        """
        Met à jour le simulateur avec l'état de la simulation
        
        Args:
            sim_state: État de la simulation
        """
        if self.simulator:
            self.simulator.update_sensor_from_simulation(sim_state)
    
    def get_sim_to_real_gap(self, sim_state: Dict[str, float]) -> Dict[str, float]:
        """Calcule l'écart Sim-to-Real"""
        if self.simulator:
            return self.simulator.get_sim_to_real_gap(sim_state)
        return {}
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques"""
        if self.simulator:
            return self.simulator.get_statistics()
        return {
            'is_connected': self.serial_conn is not None and self.serial_conn.is_open,
            'mode': self.config.mode.value
        }
    
    def export_logs(self, filepath: str):
        """Exporte les logs"""
        if self.simulator:
            self.simulator.export_logs(filepath)
    
    def close(self):
        """Ferme les connexions matérielles"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("🔌 Connexion série fermée")
        
        if RPI_AVAILABLE:
            GPIO.cleanup()
            print("🔌 GPIO nettoyé")
        
        if self.simulator:
            self.simulator.disconnect()
        
        print("✅ Interface HIL fermée")


class HILEnvironment:
    """
    Environnement Hardware-in-the-Loop
    
    Combine:
    - Simulation numérique (jumeau numérique)
    - Capteurs/actionneurs simulés ou réels
    - Métriques de gap Sim-to-Real
    """
    
    def __init__(self, base_env, config: HardwareConfig = None):
        self.base_env = base_env
        self.config = config or HardwareConfig()
        self.hardware = HardwareInterface(self.config)
        
        # Métriques HIL
        self.hil_metrics = {
            'episode': 0,
            'step': 0,
            'sim_to_real_gaps': [],
            'latency_measurements': [],
            'action_mismatches': [],
            'safety_violations_prevented': 0
        }
        
        # Buffer pour les mesures
        self.measurement_buffer = []
        self.buffer_size = 10
        
        print(f"\n{'='*60}")
        print(f"🔬 ENVIRONNEMENT HIL INITIALISÉ")
        print(f"   Mode: {self.config.mode.value}")
        print(f"   Serial port: {self.config.serial_port}")
        print(f"   Log hardware: {self.config.log_hardware}")
        print(f"{'='*60}\n")
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.hil_metrics['step'] = 0
        self.measurement_buffer = []
        return self.base_env.reset()
    
    def step(self, actions: Dict[int, int]):
        """
        Exécute une étape avec intégration HIL
        
        Args:
            actions: Actions des agents
            
        Returns:
            (observations, rewards, dones, truncated, info)
        """
        start_time = time.time()
        self.hil_metrics['step'] += 1
        
        # 1. Récupérer l'état simulé AVANT action
        sim_state = {}
        if 0 in self.base_env.machine_states:
            sim_state = {
                'temperature': self.base_env.machine_states[0].temperature,
                'pressure': self.base_env.machine_states[0].pressure,
                'speed': self.base_env.machine_states[0].speed
            }
        
        # 2. Exécuter dans la simulation
        sim_obs, rewards, dones, truncated, info = self.base_env.step(actions)
        
        # 3. Mettre à jour l'interface hardware avec l'état simulé
        self.hardware.update_from_simulation(sim_state)
        
        # 4. Lire les capteurs (réels ou simulés)
        real_measurements = self.hardware.read_sensors()
        
        # 5. Envoyer les commandes aux actionneurs
        for agent_id, action in actions.items():
            self.hardware.write_actuator(action)
        
        # 6. Corriger les observations avec les mesures réelles
        for agent_id in sim_obs:
            if real_measurements:
                # Dénormaliser
                sim_temp = sim_obs[agent_id][0] * 850
                sim_pressure = sim_obs[agent_id][1] * 10
                
                # Valeurs réelles mesurées
                real_temp = real_measurements.get('temperature', sim_temp)
                real_pressure = real_measurements.get('pressure', sim_pressure)
                
                # Filtre complémentaire (pondération simulation/réel)
                alpha = 0.7  # Poids de la simulation
                
                fused_temp = alpha * sim_temp + (1 - alpha) * real_temp
                fused_pressure = alpha * sim_pressure + (1 - alpha) * real_pressure
                
                # Mettre à jour l'observation
                sim_obs[agent_id][0] = fused_temp / 850
                sim_obs[agent_id][1] = fused_pressure / 10
                
                # Calculer et enregistrer l'écart
                temp_gap = abs(sim_temp - real_temp) / 850
                pressure_gap = abs(sim_pressure - real_pressure) / 10
                
                self.hil_metrics['sim_to_real_gaps'].append({
                    'step': self.hil_metrics['step'],
                    'agent': agent_id,
                    'temperature_gap': temp_gap,
                    'pressure_gap': pressure_gap,
                    'sim_temp': sim_temp,
                    'real_temp': real_temp,
                    'sim_pressure': sim_pressure,
                    'real_pressure': real_pressure
                })
        
        # 7. Mesurer la latence
        latency = time.time() - start_time
        self.hil_metrics['latency_measurements'].append(latency)
        
        # 8. Log périodique
        if self.config.log_hardware and self.hil_metrics['step'] % 100 == 0:
            avg_gap = np.mean([g['temperature_gap'] for g in self.hil_metrics['sim_to_real_gaps'][-100:]]) if self.hil_metrics['sim_to_real_gaps'] else 0
            print(f"[HIL] Step {self.hil_metrics['step']}: avg_gap={avg_gap:.3f}, latency={latency*1000:.1f}ms")
        
        return sim_obs, rewards, dones, truncated, info
    
    def get_hil_metrics(self) -> Dict:
        """Retourne les métriques HIL complètes avec types Python natifs"""
        gaps = self.hil_metrics['sim_to_real_gaps']
    
        temp_gaps = [float(g['temperature_gap']) for g in gaps] if gaps else []
        pressure_gaps = [float(g['pressure_gap']) for g in gaps] if gaps else []
    
        recent_gaps = []
        for g in gaps[-10:]:
            recent_gaps.append({
                'step': int(g.get('step', 0)),
                'agent': int(g.get('agent', 0)),
                'temperature_gap': float(g.get('temperature_gap', 0)),
                'pressure_gap': float(g.get('pressure_gap', 0)),
                'sim_temp': float(g.get('sim_temp', 0)),
                'real_temp': float(g.get('real_temp', 0)),
                'sim_pressure': float(g.get('sim_pressure', 0)),
                'real_pressure': float(g.get('real_pressure', 0))
            })
    
        metrics = {
            'episode': int(self.hil_metrics['episode']),
            'total_steps': int(self.hil_metrics['step']),
            'mean_temperature_gap': float(np.mean(temp_gaps)) if temp_gaps else 0.0,
            'std_temperature_gap': float(np.std(temp_gaps)) if temp_gaps else 0.0,
            'max_temperature_gap': float(np.max(temp_gaps)) if temp_gaps else 0.0,
            'mean_pressure_gap': float(np.mean(pressure_gaps)) if pressure_gaps else 0.0,
            'std_pressure_gap': float(np.std(pressure_gaps)) if pressure_gaps else 0.0,
            'max_pressure_gap': float(np.max(pressure_gaps)) if pressure_gaps else 0.0,
            'mean_latency_ms': float(np.mean(self.hil_metrics['latency_measurements']) * 1000) if self.hil_metrics['latency_measurements'] else 0.0,
            'std_latency_ms': float(np.std(self.hil_metrics['latency_measurements']) * 1000) if self.hil_metrics['latency_measurements'] else 0.0,
            'hardware_stats': self.hardware.get_statistics() if self.hardware else {},
            'recent_gaps': recent_gaps
        }
    
        return convert_to_serializable(metrics)

    def end_episode(self, episode_num: int):
        """Finalise un épisode HIL"""
        self.hil_metrics['episode'] = episode_num
        
        metrics = self.get_hil_metrics()
        
        print(f"\n{'='*60}")
        print(f"📊 RÉSUMÉ ÉPISODE HIL #{episode_num}")
        print(f"{'='*60}")
        print(f"   Steps: {metrics['total_steps']}")
        print(f"   Gap température moyen: {metrics['mean_temperature_gap']*100:.2f}%")
        print(f"   Gap pression moyen: {metrics['mean_pressure_gap']*100:.2f}%")
        print(f"   Latence moyenne: {metrics['mean_latency_ms']:.1f}ms")
        print(f"{'='*60}\n")
    
    def export_hil_report(self, filepath: str):
        """Exporte un rapport HIL complet"""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
        raw_metrics = self.get_hil_metrics()
        serializable_metrics = convert_to_serializable(raw_metrics)
    
        raw_hw_stats = self.hardware.get_statistics() if self.hardware else {}
        serializable_hw_stats = convert_to_serializable(raw_hw_stats)
    
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'mode': self.config.mode.value,
                'serial_port': self.config.serial_port,
                'baud_rate': self.config.baud_rate,
                'gpio_pins': self.config.gpio_pins
            },
            'metrics': serializable_metrics,
            'hardware_logs': serializable_hw_stats
        }
    
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
        print(f"✅ Rapport HIL exporté dans {output_path}")
    
        if self.hardware:
            self.hardware.export_logs(str(output_path.parent / "hil_communication_logs.json"))
    
    def close(self):
        """Ferme l'environnement"""
        self.hardware.close()
        self.base_env.close()


if __name__ == "__main__":
    print("Module HIL Environment - Importé avec succès")