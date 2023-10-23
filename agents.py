'''
This code is mainly based off https://github.com/RoyElkabetz/DQN_with_PyTorch_and_Gym
'''

import numpy as np
from networks import DQNetwork, DuelingDQNetwork
import torch as T
from replay_memory import ReplayBuffer

class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, image_input_dims,
                mem_size, batch_size, eps_min=0.01, eps_dec=1e-4,
                replace=1000, algo=None, env_name=None, fname=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.image_input_dims = image_input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_count = replace
        self.algo = algo
        self.env_name = env_name
        self.fname = fname
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(self.mem_size, self.image_input_dims, self.n_actions)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        image_state, xyz_state, action, reward, new_image_state, new_xyz_state, done = self.memory.sample_buffer(self.batch_size)

        image_states = T.tensor(image_state).to(self.q_eval.device)
        xyz_states = T.tensor(xyz_state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        image_states_ = T.tensor(new_image_state).to(self.q_eval.device)
        xyz_states_ = T.tensor(new_xyz_state).to(self.q_eval.device)
        dones = T.tensor(done, dtype=T.bool).to(self.q_eval.device)

        return image_states, xyz_states, actions, rewards, image_states_, xyz_states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def save_seperate_models(self, filename_end):
        self.q_eval.save_seperate_checkpoint(filename_end)
        self.q_next.save_seperate_checkpoint(filename_end)

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                name=self.fname + '_q_eval',
                                chkpt_dir=self.chkpt_dir)
        self.q_next = DQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                name=self.fname + '_q_next',
                                chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            image_state = T.tensor([observation['image']], dtype=T.float).to(self.q_eval.device)
            xyz_state = T.tensor([observation['xyz']], dtype=T.float).to(self.q_eval.device)            
            actions = self.q_eval.forward(image_state, xyz_state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.learn_step_counter < self.batch_size:
            self.learn_step_counter += 1
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        image_states, xyz_states, actions, rewards, image_states_, xyz_states_, dones = self.sample_memory()

        # Deep Q learning update rule
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(image_states, xyz_states)[indices, actions]
        q_next = self.q_next.forward(image_states_, xyz_states_).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                name=self.fname + '_q_eval',
                                chkpt_dir=self.chkpt_dir)
        self.q_next = DQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                name=self.fname + '_q_next',
                                chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            image_state = T.tensor([observation['image']], dtype=T.float).to(self.q_eval.device)
            xyz_state = T.tensor([observation['xyz']], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(image_state, xyz_state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.learn_step_counter < self.batch_size:
            self.learn_step_counter += 1
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        image_states, xyz_states, actions, rewards, image_states_, xyz_states_, dones = self.sample_memory()

        # Double Deep Q learning update rule
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(image_states, xyz_states)[indices, actions]
        max_actions = T.argmax(self.q_eval(image_states_, xyz_states_), dim=1)
        q_next = self.q_next.forward(image_states_, xyz_states_)[indices, max_actions]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                       name=self.fname + '_q_eval',
                                       chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                       name=self.fname + '_q_next',
                                       chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            image_state = T.tensor([observation['image']], dtype=T.float).to(self.q_eval.device)
            xyz_state = T.tensor([observation['xyz']], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(image_state, xyz_state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.learn_step_counter < self.batch_size:
            self.learn_step_counter += 1
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        image_states, xyz_states, actions, rewards, image_states_, xyz_states_, dones = self.sample_memory()

        # Dueling Deep Q learning update rule
        indices = np.arange(self.batch_size)
        V_s, A_s = self.q_eval.forward(image_states, xyz_states)
        V_s_, A_s_ = self.q_next.forward(image_states_, xyz_states_)
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DuelingDDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDDQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = DuelingDQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                       name=self.fname + '_q_eval',
                                       chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDQNetwork(self.image_input_dims, self.n_actions, self.lr,
                                       name=self.fname + '_q_next',
                                       chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            image_state = T.tensor([observation['image']], dtype=T.float).to(self.q_eval.device)
            xyz_state = T.tensor([observation['xyz']], dtype=T.float).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(image_state, xyz_state)
            action = T.argmax(advantages).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.learn_step_counter < self.batch_size:
            self.learn_step_counter += 1
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        image_states, xyz_states, actions, rewards, image_states_, xyz_states_, dones = self.sample_memory()

        # Dueling Double Deep Q learning update rule
        indices = np.arange(self.batch_size)
        V_s, A_s = self.q_eval.forward(image_states, xyz_states)
        V_s_, A_s_ = self.q_next.forward(image_states_, xyz_states_)
        V_s_eval, A_s_eval = self.q_eval.forward(image_states_, xyz_states_)
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
