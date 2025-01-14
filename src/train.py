from gymnasium.wrappers import TimeLimit
from fast_env_py import FastHIVPatient as HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate_HIV, evaluate_HIV_population
import os
import random

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.device = device

    def append(self, state, action, reward, next_state):
        if self.__len__() == self.capacity:
            self.data.pop(0)
        self.data.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)


class DQN(nn.Module):
    def __init__(self, env, hidden_size, depth):
        super(DQN, self).__init__()
        self.input = nn.Linear(env.observation_space.shape[0], hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.output = nn.Linear(hidden_size, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.input(x))
        for h in self.hidden:
            x = F.relu(h(x))
        return self.output(x)

class ProjectAgent:
    def __init__(self, config=None):

        self.max_episode = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.9
        self.batch_size = 128
        self.memory_length = 100000
        self.memory = ReplayBuffer(self.memory_length, self.device)
        self.eps0 = 1
        self.epsinf = 0.005
        self.epsdecay = 0.9999
        self.hidden_size = 128
        self.depth = 5
        self.policy_net = DQN(env, self.hidden_size, self.depth).to(self.device)
        self.target_net = DQN(env, self.hidden_size, self.depth).to(self.device)
        self.criterion = torch.nn.SmoothL1Loss()
        self.lr = 1E-3
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.gd_steps = 1
        self.update_target_every = 1
        self.update_tau = 1

        self.save_every=20

        self.eps=1
        
    def act(self, state, use_random=False):
        if use_random and np.random.rand() < self.eps:
            return env.action_space.sample()
        else: 
            with torch.no_grad():
                action_values = self.policy_net(torch.Tensor(state).unsqueeze(0).to(self.device))
                return torch.argmax(action_values).item() 

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            self.optimizer.zero_grad()

            states, actions, rewards, next_states = self.memory.sample(self.batch_size)
            next_state_values = self.target_net(next_states).max(1)[0].detach()
            update = rewards + self.gamma*next_state_values
            predicted_action_values = self.policy_net(states).gather(1, actions.to(torch.long).unsqueeze(1))
            loss = self.criterion(predicted_action_values, update.unsqueeze(1))
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
            
            self.optimizer.step() 
            loss = loss.detach().item()
        else: 
            loss = 'NA' 
            
        return loss

    def update_target(self):
        target_state_dict = self.target_net.state_dict()
        model_state_dict = self.policy_net.state_dict()
        for key in model_state_dict:
            target_state_dict[key] = self.update_tau * model_state_dict[key] + (1 - self.update_tau) * target_state_dict[key]
        self.target_net.load_state_dict(target_state_dict)

    
    def train(self, env):
        
        cumR_per_epoch = []
        self.eps = self.eps0
        step = 0
        best_score = 0
        
        for episode in range(self.max_episode):
            trunc = False
            cumR = 0
            state, _ = env.reset()
            step = 0
            loss = 'NA'

            episode_losses = []
            while not trunc :
                
                step+=1
                
                self.eps *= self.epsdecay
                self.eps = max(self.epsinf, self.eps)
    
                action = self.act(state, use_random=True)
                next_state, reward, _, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state)
                cumR += reward
                
                state = next_state

            for _ in range(self.gd_steps): 
                loss = self.gradient_step()
                
            if loss != 'NA':
                episode_losses.append(loss)
                    
            if episode%self.update_target_every == 0 :   
                self.update_target()
                
            episode += 1

            print(f"Episode {episode}, epsilon {self.eps:.4f}, cumulated reward {cumR:.3e}, prediction error {np.mean(episode_losses)}")

            if episode%50 == 0:
                score = evaluate_HIV(agent=self, nb_episode=1)
                print(f"Reward with greedy policy : {score:.3e}")

            if episode%self.save_every==0 :
                self.save(f"{os.getcwd()}/dqn.pth")
                print("saved model")

            print("------------------------------------------------------------------------------------------------------")
            
            cumR_per_epoch.append(cumR)
            
        return cumR
           

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load(f"{os.getcwd()}/" + 'dqn.pth'))
        self.model.eval()

    
if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(env)
    agent.save('dqn.pth')