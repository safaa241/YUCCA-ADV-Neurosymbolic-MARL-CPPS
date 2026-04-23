"""
LIVRABLE 3 - Agent MAPPO-NS (Neurosymbolique)
Agent MAPPO avec intégration du shield neurosymbolique

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
from pathlib import Path
import logging
from datetime import datetime

from neurosymbolic_shield import SymbolicShield, KnowledgeBase


class MAPPOAgentNS:
    """
    Agent MAPPO avec Shield Neurosymbolique (MAPPO-NS)
    
    Combine:
    - Apprentissage PPO pour la performance
    - Shield symbolique pour la sécurité garantie
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 entropy_coef: float = 0.01,
                 use_shield: bool = True):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.use_shield = use_shield
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Réseaux de neurones
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Shield neurosymbolique
        if use_shield:
            self.kb = KnowledgeBase()
            self.shield = SymbolicShield(self.kb)
        else:
            self.shield = None
        
        # Buffer de mémoire
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Métriques
        self.episode_rewards = []
        self.shield_stats_history = []
        
    def _build_actor(self) -> nn.Module:
        """Construit le réseau actor (politique)"""
        return nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def _build_critic(self) -> nn.Module:
        """Construit le réseau critic (valeur)"""
        return nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    # Sélection d'action avec filtrage neurosymbolique et explication des interventions du shield 
    def select_action(self, observation: np.ndarray, 
                      explore: bool = True) -> Tuple[int, torch.Tensor, torch.Tensor]:
       
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        # Forward pass
        action_logits = self.actor(obs_tensor)
        value = self.critic(obs_tensor)
        
        # Distribution de probabilité
        action_probs = torch.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        if explore:
            raw_action = action_dist.sample()
            log_prob = action_dist.log_prob(raw_action)
        else:
            raw_action = torch.argmax(action_probs)
            log_prob = torch.tensor(0.0)
        
        raw_action_int = raw_action.item()
        
        # Application du shield neurosymbolique
        if self.use_shield and self.shield:
            safe_action, was_modified, explanation, _ = self.shield.filter_action(
                raw_action_int, observation
            )
            final_action = safe_action
        else:
            final_action = raw_action_int
            was_modified = False
        
        # Si l'action a été modifiée, recalculer log_prob
        if was_modified and explore:
            # Recalculer log_prob pour l'action modifiée
            log_prob = action_dist.log_prob(torch.tensor(final_action).to(self.device))
        
        return final_action, log_prob.detach(), value.detach()
    
    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                        value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """Stocke une transition dans la mémoire"""
        self.memory['observations'].append(obs)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_returns_and_advantages(self, next_value: float = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule les retours et avantages avec GAE"""
        returns = []
        advantages = []
        gae = 0
        
        values = self.memory['values'] + [torch.tensor(next_value, dtype=torch.float32)]
        
        for t in reversed(range(len(self.memory['rewards']))):
            delta = (self.memory['rewards'][t] + 
                    self.gamma * values[t+1] * (1 - self.memory['dones'][t]) - 
                    values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self.memory['dones'][t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.stack([torch.tensor(a, dtype=torch.float32) for a in advantages])
        returns = torch.stack([torch.tensor(r, dtype=torch.float32) for r in returns])
        
        # Normalisation des avantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, returns: torch.Tensor, advantages: torch.Tensor,
               epochs: int = 3, batch_size: int = 32) -> Tuple[float, float]:
        """Met à jour les réseaux avec PPO"""
        
        obs_tensor = torch.stack([torch.FloatTensor(o) for o in self.memory['observations']]).to(self.device)
        actions_tensor = torch.LongTensor(self.memory['actions']).to(self.device)
        returns_tensor = returns.to(self.device)
        advantages_tensor = advantages.to(self.device)
        old_log_probs = torch.stack(self.memory['log_probs']).to(self.device)
        
        total_actor_loss = 0
        total_critic_loss = 0
        n_batches = 0
        
        for _ in range(epochs):
            indices = np.random.permutation(len(self.memory['observations']))
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                
                # Update critic
                values = self.critic(obs_tensor[batch_idx]).squeeze()
                critic_loss = nn.MSELoss()(values, returns_tensor[batch_idx])
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                # Update actor
                action_logits = self.actor(obs_tensor[batch_idx])
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(actions_tensor[batch_idx])
                
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                
                surrogate1 = ratio * advantages_tensor[batch_idx]
                surrogate2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_tensor[batch_idx]
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Entropy bonus
                entropy = action_dist.entropy().mean() * self.entropy_coef
                total_actor_loss_batch = actor_loss - entropy
                
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_batches += 1
        
        # Vider la mémoire
        self.memory = {k: [] for k in self.memory}
        
        return total_actor_loss / max(n_batches, 1), total_critic_loss / max(n_batches, 1)
    
    def get_shield_stats(self) -> Dict:
        """Retourne les statistiques du shield"""
        if self.use_shield and self.shield:
            return self.shield.get_stats()
        return {}
    
    def get_recent_explanations(self, n: int = 10) -> List[Dict]:
        """Retourne les dernières explications du shield"""
        if self.use_shield and self.shield:
            return self.shield.get_recent_explanations(n)
        return []
    
    def save(self, path: str):
        """Sauvegarde le modèle"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, save_path)
        
        # Sauvegarder aussi les explications si shield actif
        if self.use_shield and self.shield:
            self.shield.export_explanations(str(save_path.parent / "explanations.json"))
        
        print(f"✅ Modèle sauvegardé dans {save_path}")
    
    def load(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])
        print(f"✅ Modèle chargé depuis {path}")


class MultiAgentMAPPO_NS:
    """
    Système multi-agents MAPPO-NS
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int,
                 use_shield: bool = True, learning_rate: float = 3e-4):
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_shield = use_shield
        
        self.agents = {
            i: MAPPOAgentNS(obs_dim, action_dim, learning_rate, use_shield=use_shield)
            for i in range(num_agents)
        }
        
        self.name = "MAPPO-NS (Neurosymbolique)" if use_shield else "MAPPO (Standard)"
        
        # Historique global
        self.episode_history = []
    
    def select_actions(self, observations: Dict[int, np.ndarray], 
                       explore: bool = True) -> Dict[int, int]:
        """Sélectionne les actions pour tous les agents"""
        actions = {}
        
        for agent_id, obs in observations.items():
            action, _, _ = self.agents[agent_id].select_action(obs, explore)
            actions[agent_id] = action
        
        return actions
    
    def store_transitions(self, observations: Dict[int, np.ndarray],
                         actions: Dict[int, int],
                         rewards: Dict[int, float],
                         values: Dict[int, torch.Tensor],
                         log_probs: Dict[int, torch.Tensor],
                         dones: Dict[int, bool]):
        """Stocke les transitions pour tous les agents"""
        for agent_id in range(self.num_agents):
            self.agents[agent_id].store_transition(
                observations[agent_id],
                actions[agent_id],
                rewards[agent_id],
                values[agent_id],
                log_probs[agent_id],
                dones[agent_id]
            )
    
    def update_all(self, next_observations: Dict[int, np.ndarray]) -> Tuple[float, float]:
        """Met à jour tous les agents"""
        total_actor_loss = 0
        total_critic_loss = 0
        
        for agent_id in range(self.num_agents):
            # Calculer next_value
            _, _, next_value = self.agents[agent_id].select_action(
                next_observations[agent_id], explore=False
            )
            
            # Calculer returns et advantages
            returns, advantages = self.agents[agent_id].compute_returns_and_advantages(
                next_value.item()
            )
            
            # Update
            actor_loss, critic_loss = self.agents[agent_id].update(returns, advantages)
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        return total_actor_loss / self.num_agents, total_critic_loss / self.num_agents
    
    def get_shield_stats(self) -> Dict:
        """Retourne les statistiques globales du shield"""
        if not self.use_shield:
            return {}
        
        all_stats = [agent.get_shield_stats() for agent in self.agents.values()]
        
        # Aggréger les statistiques
        return {
            'total_checks': sum(s.get('total_checks', 0) for s in all_stats),
            'safe_actions': sum(s.get('safe_actions', 0) for s in all_stats),
            'corrected_actions': sum(s.get('corrected_actions', 0) for s in all_stats),
            'blocked_actions': sum(s.get('blocked_actions', 0) for s in all_stats),
            'intervention_rate': sum(s.get('intervention_rate', 0) for s in all_stats) / self.num_agents,
            'by_agent': {i: stats for i, stats in enumerate(all_stats)}
        }
    
    def get_all_explanations(self) -> List[Dict]:
        """Retourne toutes les explications de tous les agents"""
        explanations = []
        for agent_id, agent in self.agents.items():
            for exp in agent.get_recent_explanations(1000):
                exp['agent_id'] = agent_id
                explanations.append(exp)
        
        # Trier par timestamp
        explanations.sort(key=lambda x: x.get('timestamp', ''))
        
        return explanations