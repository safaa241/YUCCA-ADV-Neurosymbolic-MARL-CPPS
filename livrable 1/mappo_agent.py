# Agent MAPPO pour apprentissage multi-agents ( réseaux, mémoire, PPO update, etc.)
# ce qu'il produit: Agent prêt à apprendre, avec des fonctions pour sélectionner des actions, stocker des transitions, calculer les avantages et mettre à jour les réseaux de neurones.

from altair import value
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List
from collections import deque

 
class MAPPOAgent:
    """
    Agent MAPPO pour apprentissage multi-agents
    """
    
    def __init__(self, obs_dim, action_dim, learning_rate=3e-4, gamma=0.99):
        """
        Initialisation de l'agent MAPPO
        
        Args:
            obs_dim: Dimension de l'observation
            action_dim: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage
            gamma: Facteur d'actualisation
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convertir le learning_rate en float (au cas où il viendrait d'une config en string)
        learning_rate = float(learning_rate)
        
        # Réseaux de neurones
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        
        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Buffer d'expérience
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def _build_actor(self):
        """Construit le réseau actor (politique)"""
        return nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def _build_critic(self):
        """Construit le réseau critic (évaluateur de valeur)"""
        return nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def select_action(self, observation, explore=True):
        obs_tensor = torch.FloatTensor(observation).to(self.device)
    
        action_logits = self.actor(obs_tensor)
        value = self.critic(obs_tensor)
    
        action_probs = torch.softmax(action_logits, dim=-1)
    
        if explore:
            # Échantillonner depuis la distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        else:
            # Choisir l'action avec la plus haute probabilité
            action = torch.argmax(action_probs)
    
        log_prob = action_dist.log_prob(action) if explore else torch.tensor(0.0)
    
        return action.item(), log_prob.detach(), value.detach()
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Mémorise une transition"""
        self.memory['observations'].append(obs)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_returns_and_advantages(self, next_value=0):
        """
        Calcule les retours et avantages (GAE)
        
        Returns:
            returns, advantages
        """
        returns = []
        advantages = []
        gae = 0
        next_value_tensor = torch.tensor(next_value, dtype=torch.float32)
        
        # Traiter en sens inverse
        values = self.memory['values'] + [next_value_tensor]
        
        for t in reversed(range(len(self.memory['rewards']))):
            delta = self.memory['rewards'][t] + \
                    self.gamma * values[t+1] * (1 - self.memory['dones'][t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - self.memory['dones'][t]) * gae
            
            advantage = gae
            ret = advantage + values[t]
            
            advantages.insert(0, advantage)
            returns.insert(0, ret)
        
        advantages = torch.stack([torch.tensor(a, dtype=torch.float32) 
                                 for a in advantages])
        returns = torch.stack([torch.tensor(r, dtype=torch.float32) 
                              for r in returns])
        
        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, returns, advantages, batch_size=32, num_epochs=3):
        """
        Met à jour les réseaux actor et critic
        
        Args:
            returns: Retours calculés
            advantages: Avantages calculés
            batch_size: Taille des batches
            num_epochs: Nombre d'epochs
            
        Returns:
            actor_loss, critic_loss
        """
        obs_tensor = torch.stack([torch.FloatTensor(o) for o in self.memory['observations']]).to(self.device)
        actions_tensor = torch.LongTensor(self.memory['actions']).to(self.device)
        returns_tensor = returns.to(self.device)
        advantages_tensor = advantages.to(self.device)
        
        actor_losses = []
        critic_losses = []
        
        for _ in range(num_epochs):
            # Créer des mini-batches
            indices = np.random.permutation(len(self.memory['observations']))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                obs_batch = obs_tensor[batch_indices]
                actions_batch = actions_tensor[batch_indices]
                returns_batch = returns_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                
                # Update critic
                values = self.critic(obs_batch).squeeze()
                critic_loss = nn.functional.mse_loss(values, returns_batch)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
                
                # Update actor
                action_logits = self.actor(obs_batch)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                log_probs = action_dist.log_prob(actions_batch)
                
                # PPO loss with clipping
                ratio = torch.exp(log_probs - torch.stack(self.memory['log_probs'])[batch_indices].to(self.device))
                clip_range = 0.2
                
                surrogate1 = ratio * advantages_batch
                surrogate2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages_batch
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Entropy bonus
                entropy_bonus = -action_dist.entropy().mean() * 0.01
                total_actor_loss = actor_loss + entropy_bonus
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
        
        # Vider la mémoire
        self.memory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        
        return avg_actor_loss, avg_critic_loss
    
    def save(self, path):
        """Sauvegarde le modèle"""
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict()
        }, path)
    
    def load(self, path):
        """Charge le modèle"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])
