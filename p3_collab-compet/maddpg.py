import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 6e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # Update every time step
NUM_UPDATE = 1          # Update once at each time step


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Agent():
    """Single agent as part of multi-agent setup"""
    def __init__(self, agent_number, state_size, action_size, random_seed):
        self.num = agent_number

        # Actor Networks (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_target = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        return


class MAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, num_agents=2):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        if num_agents != 2:
            raise Exception("Only tested to work with 2 agents")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.t_step = 0

        self.num_agents = num_agents
        self.agents = [ Agent(i, state_size, action_size, random_seed) for i in range(num_agents) ]

        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)

    def reset(self):
        self.noise.reset()

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy with noise"""
        states = torch.from_numpy(states).float().to(device)
        actions = []
        for i in range(self.num_agents):
            agent = self.agents[i]
            agent.actor_local.eval()
            with torch.no_grad():
                action = agent.actor_local(states[i].unsqueeze(0)).cpu().data.numpy().squeeze()
                #action = agent.actor_local(states[i]).cpu().data.numpy()
                actions.append(action)
            agent.actor_local.train()
        actions = np.array(actions)
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for i in range(self.num_agents):
            self.agents[i].memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if (len(self.agents[0].memory) > BATCH_SIZE) and self.t_step == 0:
            for _ in range(NUM_UPDATE):
                agent_experiences = [ agent.memory.sample() for agent in self.agents ]
            self.learn(agent_experiences, GAMMA)

    def learn(self, agent_experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
         Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
         where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            agent_experiences (List[Tuple[torch.Tensor]]): list (for num of agents) of tuple of (s, a, r, s', done) of the agent 0 tuples
            gamma (float): discount factor
        """

        # Expand out the experiences
        # Each output is a list of num_agents
        agent_states, agent_actions, agent_rewards, agent_next_states, agent_dones = zip(*agent_experiences)

        # Combine states, actions, next_states across both agents
        states = torch.cat(agent_states, dim=1).float().to(device)
        actions = torch.cat(agent_actions, dim=1).float().to(device)
        next_states = torch.cat(agent_next_states, dim=1).float().to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get the next actions for each agent
        next_actions = [ self.agents[i].actor_target(agent_next_states[i]) for i in range(self.num_agents) ]
        next_actions = torch.cat(next_actions, dim=1).float().to(device)
        for i in range(self.num_agents):
            agent = self.agents[i]
            rewards, dones = agent_rewards[i], agent_dones[i]
            # Get predicted next-state actions and Q values from target models
            Q_targets_next = agent.critic_target(next_states, next_actions)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = agent.critic_local(states, actions)
            huber_loss = torch.nn.SmoothL1Loss()
            critic_loss = huber_loss(Q_expected, Q_targets.detach())
            #critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            agent.critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic_optim.step()


        # ---------------------------- update actor  ---------------------------- #
        # This will get the actions for each state
        pred_actions = [ self.agents[i].actor_local(agent_states[i]) for i in range(self.num_agents) ]
        # Concatenated the predicted actions to give to the centralized critic
        pred_actions = torch.cat(pred_actions, dim=1).float().to(device)
        for agent in self.agents:
            # Compute actor loss
            actor_loss = -agent.critic_local(states, pred_actions).mean()
            # Minimize the loss
            agent.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor_optim.step()


        # ----------------------- update target networks ----------------------- #
        for agent in self.agents:
            self.soft_update(agent.critic_local, agent.critic_target, TAU)
            self.soft_update(agent.actor_local, agent.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
