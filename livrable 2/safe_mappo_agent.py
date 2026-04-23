"""
LIVRABLE 2 - Safe Reinforcement Learning pour CPPS
Agent MAPPO avec contraintes de sécurité (Safe RL)

Méthodes implémentées:
- Pénalités Lagrangiennes
- Control Barrier Functions (CBF)
- Pénalités dynamiques adaptatives

Auteur: FEKNI Safaa
Projet: YUCCA-ADV PFE 2026
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

# Agent MAPPO avec contraintes de sécurité Safe RL 
class SafeMAPPOAgent:    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 cost_limit: float = 10.0,
                 lambda_lr: float = 0.01,
                 use_cbf: bool = True,
                 use_lagrangian: bool = True,
                 adaptive_penalty: bool = True):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.use_cbf = use_cbf
        self.use_lagrangian = use_lagrangian
        self.adaptive_penalty = adaptive_penalty
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Réseaux de neurones
        self.actor = self._build_network(obs_dim, action_dim).to(self.device)
        self.critic = self._build_network(obs_dim, 1).to(self.device)
        self.cost_critic = self._build_network(obs_dim, 1).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=learning_rate)
        
        # Multiplicateur Lagrangien
        self.lagrangian_lambda = 0.0
        
        # Historique
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'costs': []  # Coûts pour Safe RL
        }
        
        # Métriques
        self.violation_history = deque(maxlen=100)
        self.cost_history = deque(maxlen=100)
        
    def _build_network(self, input_dim: int, output_dim: int) -> nn.Module:
        """Construit un réseau de neurones"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    
    def cbf_check(self, observation: np.ndarray, action: int) -> bool:
        """
        Vérifie la condition Control Barrier Function (CBF)
        
        La CBF garantit que l'action ne mène pas à un état dangereux
        """
        # Dénormalisation
        temp = observation[0] * 850
        pressure = observation[1] * 10
        speed = observation[2] * 10
        
        # Condition CBF pour augmentation de vitesse
        if action == 2:  # increase_speed
            if temp > 800:  # La température ne doit pas dépasser 850°C
                return False
            if pressure > 9.0:  # La pression ne doit pas dépasser 10 bar
                return False
            if speed >= 9.5:  # La vitesse ne doit pas dépasser la limite
                return False
        
        return True
    
    def compute_cost(self, observation: np.ndarray, action: int, agent_id: int) -> float:
        """Calcule le coût (violation de sécurité)"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        maintenance = observation[4] > 0.5
        
        cost = 0.0
        
        # Coût température (échelle exponentielle)
        if temp >= 850:
            cost += 100.0
        elif temp > 800:
            cost += 50.0
        elif temp > 750:
            cost += 20.0
        elif temp > 700:
            cost += 10.0
        
        # Coût pression (agent peinture)
        if agent_id == 1:
            if pressure >= 10:
                cost += 100.0
            elif pressure > 9.0:
                cost += 50.0
            elif pressure > 8.5:
                cost += 20.0
        
        # Coût maintenance
        if maintenance:
            cost += 80.0
        
        # Coût pour actions dangereuses
        if action == 2 and temp > 750:
            cost += 30.0
        if action == 2 and pressure > 8.5:
            cost += 30.0
        
        return cost
    
    def compute_adaptive_penalty(self, cost: float) -> float:
        """Calcule une pénalité adaptative basée sur l'historique"""
        if not self.adaptive_penalty:
            return cost
        
        avg_cost = np.mean(self.cost_history) if self.cost_history else 0
        violation_rate = len([c for c in self.cost_history if c > 0]) / max(1, len(self.cost_history))
        
        # Pénalité adaptative
        beta = 1.0 + 5.0 * violation_rate
        adaptive_penalty = beta * cost * (1 + avg_cost / 100)
        
        return adaptive_penalty
    
    def select_action(self, observation: np.ndarray, explore: bool = True) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sélectionne une action avec vérification CBF"""
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        action_logits = self.actor(obs_tensor)
        value = self.critic(obs_tensor)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        if explore:
            action = action_dist.sample()
        else:
            action = torch.argmax(action_probs)
        
        raw_action = action.item()
        
        # Vérification CBF
        if self.use_cbf:
            if not self.cbf_check(observation, raw_action):
                # Trouver l'action sûre la plus proche
                safe_action = self._find_safest_action(observation)
                action = torch.tensor(safe_action)
        
        log_prob = action_dist.log_prob(action) if explore else torch.tensor(0.0)
        
        return action.item(), log_prob.detach(), value.detach()
    
    def _find_safest_action(self, observation: np.ndarray) -> int:
        """Trouve l'action la plus sûre pour l'état courant"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        
        if temp >= 850:
            return 4  # STOP d'urgence
        if temp > 800:
            return 0  # Réduction vitesse
        if pressure > 9.0:
            return 0
        if temp > 750:
            return 1  # Maintien
        
        return 1
    
    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                        value: torch.Tensor, log_prob: torch.Tensor, done: bool,
                        cost: float):
        """Stocke une transition avec son coût"""
        self.memory['observations'].append(obs)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
        self.memory['costs'].append(cost)
        
        self.cost_history.append(cost)
        if cost > 0:
            self.violation_history.append(1)
        else:
            self.violation_history.append(0)
    
    def compute_returns_and_advantages(self, next_value: float = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule les retours et avantages avec GAE"""
        returns = []
        advantages = []
        gae = 0
        lam = 0.95
        
        values = self.memory['values'] + [torch.tensor(next_value, dtype=torch.float32)]
        
        for t in reversed(range(len(self.memory['rewards']))):
            delta = self.memory['rewards'][t] + self.gamma * values[t+1] * (1 - self.memory['dones'][t]) - values[t]
            gae = delta + self.gamma * lam * (1 - self.memory['dones'][t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.stack([torch.tensor(a, dtype=torch.float32) for a in advantages])
        returns = torch.stack([torch.tensor(r, dtype=torch.float32) for r in returns])
        
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def compute_cost_returns(self, next_cost_value: float = 0) -> torch.Tensor:
        """Calcule les retours pour la fonction de coût"""
        cost_returns = []
        gamma = self.gamma
        
        cost_values = self.memory['costs'] + [next_cost_value]
        
        for t in reversed(range(len(self.memory['costs']))):
            ret = self.memory['costs'][t] + gamma * cost_values[t+1] * (1 - self.memory['dones'][t])
            cost_returns.insert(0, ret)
        
        return torch.tensor(cost_returns, dtype=torch.float32)
    
    def update(self, returns: torch.Tensor, advantages: torch.Tensor,
               cost_returns: torch.Tensor, epochs: int = 3, batch_size: int = 32) -> Tuple[float, float, float, float]:
        """
        Met à jour les réseaux avec contraintes de sécurité
        """
        obs_tensor = torch.stack([torch.FloatTensor(o) for o in self.memory['observations']]).to(self.device)
        actions_tensor = torch.LongTensor(self.memory['actions']).to(self.device)
        returns_tensor = returns.to(self.device)
        advantages_tensor = advantages.to(self.device)
        cost_returns_tensor = cost_returns.to(self.device)
        old_log_probs = torch.stack(self.memory['log_probs']).to(self.device)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_cost_critic_loss = 0
        n_batches = 0
        
        for _ in range(epochs):
            indices = np.random.permutation(len(self.memory['observations']))
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                
                # 1. Update critic (récompense)
                values = self.critic(obs_tensor[batch_idx]).squeeze()
                critic_loss = nn.MSELoss()(values, returns_tensor[batch_idx])
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                # 2. Update cost critic
                cost_values = self.cost_critic(obs_tensor[batch_idx]).squeeze()
                cost_critic_loss = nn.MSELoss()(cost_values, cost_returns_tensor[batch_idx])
                
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), 0.5)
                self.cost_critic_optimizer.step()
                
                # 3. Update actor avec contrainte Lagrangienne
                action_logits = self.actor(obs_tensor[batch_idx])
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                new_log_probs = action_dist.log_prob(actions_tensor[batch_idx])
                
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                clip_range = 0.2
                
                surrogate1 = ratio * advantages_tensor[batch_idx]
                surrogate2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages_tensor[batch_idx]
                
                # Perte PPO standard
                ppo_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Pénalité Lagrangienne
                if self.use_lagrangian:
                    estimated_costs = cost_values.detach()
                    lagrangian_penalty = self.lagrangian_lambda * estimated_costs.mean()
                else:
                    lagrangian_penalty = 0
                
                # Bonus d'entropie
                entropy = action_dist.entropy().mean() * 0.01
                
                # Perte totale de l'actor
                actor_loss = ppo_loss + lagrangian_penalty - entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_cost_critic_loss += cost_critic_loss.item()
                n_batches += 1
        
        # Mise à jour du multiplicateur Lagrangien
        if self.use_lagrangian:
            avg_cost = np.mean(self.memory['costs'])
            self.lagrangian_lambda = max(0, self.lagrangian_lambda + 
                                        self.lambda_lr * (avg_cost - self.cost_limit))
        
        # Vider la mémoire
        self.memory = {k: [] for k in self.memory}
        
        return (total_actor_loss / max(n_batches, 1),
                total_critic_loss / max(n_batches, 1),
                total_cost_critic_loss / max(n_batches, 1),
                self.lagrangian_lambda)
    
    def get_safety_stats(self) -> dict:
        """Retourne les statistiques de sécurité"""
        return {
            'lagrangian_lambda': self.lagrangian_lambda,
            'avg_cost': np.mean(self.cost_history) if self.cost_history else 0,
            'violation_rate': np.mean(self.violation_history) if self.violation_history else 0,
            'cost_limit': self.cost_limit,
            'use_cbf': self.use_cbf,
            'use_lagrangian': self.use_lagrangian,
            'adaptive_penalty': self.adaptive_penalty
        }


class SafeCPPSEnvironment:
    """Wrapper d'environnement CPPS avec calcul de coût intégré"""
    
    def __init__(self, base_env, cost_limit: float = 10.0):
        self.base_env = base_env
        self.cost_limit = cost_limit
        self.episode_costs = []
        
    def reset(self):
        self.episode_costs = []
        return self.base_env.reset()
    
    def step(self, actions):
        observations, rewards, dones, truncated, info = self.base_env.step(actions)
        
        # Calcul des coûts pour chaque agent
        costs = {}
        for agent_id, action in actions.items():
            obs = observations[agent_id]
            cost = self._compute_agent_cost(obs, action, agent_id)
            costs[agent_id] = cost
            self.episode_costs.append(cost)
        
        info['costs'] = costs
        info['total_cost'] = sum(costs.values())
        
        return observations, rewards, dones, truncated, info
    
    def _compute_agent_cost(self, observation, action, agent_id):
        """Calcule le coût pour un agent"""
        temp = observation[0] * 850
        pressure = observation[1] * 10
        
        cost = 0.0
        if temp >= 850:
            cost += 100.0
        elif temp > 800:
            cost += 50.0
        elif temp > 750:
            cost += 20.0
        
        if agent_id == 1:
            if pressure >= 10:
                cost += 100.0
            elif pressure > 9.0:
                cost += 50.0
        
        return cost
    
    def get_episode_cost(self):
        return sum(self.episode_costs)