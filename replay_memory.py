'''
This code is mainly based off https://github.com/RoyElkabetz/DQN_with_PyTorch_and_Gym
'''

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, image_input_shape, n_actions):
        self.mem_size = max_size
        self.mem_count = 0
        self.image_state_memory = np.zeros((self.mem_size, *image_input_shape), dtype=np.float32)
        self.xyz_state_memory = np.zeros((self.mem_size, 3), dtype=np.float32)
        self.new_image_state_memory = np.zeros((self.mem_size, *image_input_shape), dtype=np.float32)
        self.new_xyz_state_memory = np.zeros((self.mem_size, 3), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_count % self.mem_size
        self.image_state_memory[index] = state['image']
        self.xyz_state_memory[index] = state['xyz']
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_image_state_memory[index] = state_['image']
        self.new_xyz_state_memory[index] = state_['xyz']
        self.terminal_memory[index] = done
        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_count, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        image_states = self.image_state_memory[batch]
        xyz_states = self.xyz_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        image_states_ = self.new_image_state_memory[batch]
        xyz_states_ = self.new_xyz_state_memory[batch]
        dones = self.terminal_memory[batch]

        return image_states, xyz_states, actions, rewards, image_states_, xyz_states_, dones