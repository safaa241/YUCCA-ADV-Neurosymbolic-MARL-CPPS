# Base du shield (sera enrichi en livrable 3)
# Ce qu’il produit: Un module de filtrage d’actions basé sur des règles symboliques, qui peut être intégré à l’agent pour éviter les actions dangereuses. Il fournira également des statistiques sur les actions bloquées et corrigées.

from typing import Dict, List, Tuple
import numpy as np


class SymbolicShield:
    """
    Shield symbolique pour filtrer les actions dangereuses
    et les remplacer par des actions sûres
    """
     
    def __init__(self, knowledge_base):
        """
        Initialisation du shield
        
        Args:
            knowledge_base: Instance de KnowledgeBase
        """
        self.kb = knowledge_base
        self.blocked_action_count = 0
        self.corrected_action_count = 0
        self.action_mapping = {
            0: "reduce_speed",
            1: "maintain",
            2: "increase_speed",
            3: "idle",
            4: "stop"
        }
    
    def filter_action(self, action: int, state: np.ndarray) -> Tuple[int, bool, str]:
        """
        Filtre une action selon les règles symboliques
        
        Args:
            action: Action proposée (0-4)
            state: État du système
            
        Returns:
            (filtered_action, was_modified, reason)
        """
        # Convertir l'état en dictionnaire
        state_dict = self._state_to_dict(state)
        
        # ========== CORRECTION: Règles plus strictes ==========
        
        # Règle 1: Température critique → STOP FORCÉ
        if state_dict.get('temperature', 0) >= 850:
            if action != 4:
                self.corrected_action_count += 1
                self.blocked_action_count += 1
                return 4, True, "CRITICAL: Temperature > 850°C → EMERGENCY STOP"
        
        # Règle 2: Maintenance requise → STOP
        if state_dict.get('maintenance_needed', False):
            if action != 4:
                self.corrected_action_count += 1
                return 4, True, "Maintenance required → STOP"
        
        # Règle 3: Température élevée (>800°C)
        if state_dict.get('temperature', 0) > 800:
            if action == 2:  # Tentative d'augmentation
                self.corrected_action_count += 1
                return 0, True, "Safety: Temperature too high → reducing speed"
        
        # Règle 4: Pression élevée (>9.0 bar)
        if state_dict.get('pressure', 0) > 9.0:
            if action == 2:  # Tentative d'augmentation
                self.corrected_action_count += 1
                return 0, True, "Safety: Pressure too high → reducing speed"
        
        # Action sûre
        return action, False, "Action safe"
    
    def _state_to_dict(self, state: np.ndarray) -> Dict:
        """Convertit l'état en dictionnaire"""
        return {
            'temperature': state[0] * 850,  # Dénormaliser
            'pressure': state[1] * 10,
            'speed': state[2] * 10,
            'production_count': state[3],
            'maintenance_needed': bool(state[4] > 0.5),
            'time_step': state[5]
        }
    
    def get_safe_actions(self, state: np.ndarray) -> List[int]:
        """
        Retourne la liste des actions sûres
        
        Args:
            state: État du système
            
        Returns:
            Liste des actions sûres
        """
        state_dict = self._state_to_dict(state)
        safe_actions = []
        
        # Température critique → seulement STOP
        if state_dict.get('temperature', 0) >= 850:
            return [4]
        
        # Maintenance → seulement STOP
        if state_dict.get('maintenance_needed', False):
            return [4]
        
        # Actions de base
        safe_actions = [0, 1, 3, 4]  # reduce, maintain, idle, stop
        
        # Vérifier si augmentation est permise
        temp_ok = state_dict.get('temperature', 0) <= 800
        pressure_ok = state_dict.get('pressure', 0) <= 9.0
        
        if temp_ok and pressure_ok:
            safe_actions.append(2)  # increase_speed autorisé
        
        return safe_actions
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques du shield"""
        return {
            'blocked_actions': self.blocked_action_count,
            'corrected_actions': self.corrected_action_count,
            'total_actions': self.blocked_action_count + self.corrected_action_count
        }


class ActionMasking:
    """
    Action Masking - Masquer les actions non sûres
    """
    
    def __init__(self):
        self.mask_history = []
    
    def compute_action_mask(self, state: np.ndarray, safe_actions: List[int],
                           total_actions: int = 5) -> np.ndarray:
        """
        Calcule le masque d'actions
        
        Args:
            state: État du système
            safe_actions: Liste des actions sûres
            total_actions: Nombre total d'actions
            
        Returns:
            Masque binaire
        """
        mask = np.zeros(total_actions, dtype=np.float32)
        
        for action in safe_actions:
            if 0 <= action < total_actions:
                mask[action] = 1.0
        
        self.mask_history.append(mask)
        
        return mask
    
    def apply_mask_to_probabilities(self, action_probs: np.ndarray,
                                   mask: np.ndarray) -> np.ndarray:
        """
        Applique le masque aux probabilités d'action
        
        Args:
            action_probs: Probabilités d'action
            mask: Masque binaire
            
        Returns:
            Probabilités masquées et renormalisées
        """
        masked_probs = action_probs * mask
        
        # Renormaliser
        sum_probs = np.sum(masked_probs)
        if sum_probs > 0:
            masked_probs = masked_probs / sum_probs
        else:
            # Si aucune action n'est sûre, permettre au moins une action
            masked_probs = mask / np.sum(mask)
        
        return masked_probs
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques du masking"""
        if not self.mask_history:
            return {}
        
        mask_array = np.array(self.mask_history)
        num_safe_actions = np.sum(mask_array, axis=1)
        
        return {
            'avg_safe_actions': np.mean(num_safe_actions),
            'min_safe_actions': np.min(num_safe_actions),
            'max_safe_actions': np.max(num_safe_actions),
            'num_timesteps': len(self.mask_history)
        }