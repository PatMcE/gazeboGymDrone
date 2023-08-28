'''
This code is mainly based off https://github.com/RoyElkabetz/DQN_with_PyTorch_and_Gym
'''

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import rospy

class DQNetwork(nn.Module):
	def __init__(self, image_input_dims, n_actions, lr, name, chkpt_dir):
		super().__init__()
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
		self.name = name

		self.conv1 = nn.Conv2d(image_input_dims[0], 32, (8, 8), stride=(4, 4))
		self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2))
		self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1))

		fc_input_dims = self.calculate_conv_output_dims(image_input_dims)
		self.fc1 = nn.Linear(fc_input_dims, 512)
		self.fc2 = nn.Linear(515, 16)
		self.fc3 = nn.Linear(16, n_actions)

		self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def calculate_conv_output_dims(self, image_input_dims):
		state = T.zeros(1, *image_input_dims)
		dims = self.conv1(state)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		return int(np.prod(dims.size()))

	def forward(self, image_state, xyz_state):
		x = F.relu(self.conv1(image_state))
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = x.view(x.size()[0], -1)
		x = F.relu(x)
		x = self.fc1(x)
		x = F.relu(x)
		###
		x = T.cat((x, xyz_state), dim=1)
		x = self.fc2(x)
		x = F.relu(x)
		###
		x = self.fc3(x)
		return x

	def save_checkpoint(self):
		print('... saving checkpoint ...')
		T.save(self.state_dict(), self.checkpoint_file)

	def save_seperate_checkpoint(self, filename_end):
		print('... saving seperate checkpoint ...')
		checkpoint_seperate_file = os.path.join(self.checkpoint_dir, self.name + filename_end)
		rospy.logwarn("seperate file = " + str(checkpoint_seperate_file))
		T.save(self.state_dict(), checkpoint_seperate_file)

	def load_checkpoint(self):
		print('... loading checkpoint ...')
		self.load_state_dict(T.load(self.checkpoint_file))


class DuelingDQNetwork(nn.Module):
	def __init__(self, image_input_dims, n_actions, lr, name, chkpt_dir):
		super().__init__()
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
		self.name = name

		self.conv1 = nn.Conv2d(image_input_dims[0], 32, (8, 8), stride=(4, 4))
		self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=(2, 2))
		self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1))

		fc_input_dims = self.calculate_conv_output_dims(image_input_dims)
		self.fc1 = nn.Linear(fc_input_dims, 512)
		self.fc2 = nn.Linear(515, 256)
		self.V = nn.Linear(256, 1)
		self.A = nn.Linear(256, n_actions)

		self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def calculate_conv_output_dims(self, image_input_dims):
		state = T.zeros(1, *image_input_dims)
		dims = self.conv1(state)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		return int(np.prod(dims.size()))

	def forward(self, image_state, xyz_state):
		x = F.relu(self.conv1(image_state))
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = x.view(x.size()[0], -1)
		x = F.relu(x)
		x = self.fc1(x)
		x = F.relu(x)
		###
		x = T.cat((x, xyz_state), dim=1)
		x = self.fc2(x)
		x = F.relu(x)
		###
		V = self.V(x)
		A = self.A(x)
		return V, A

	def save_checkpoint(self):
		print('... saving checkpoint ...')
		T.save(self.state_dict(), self.checkpoint_file)

	def save_seperate_checkpoint(self, filename_end):
		print('... saving seperate checkpoint ...')
		checkpoint_seperate_file = os.path.join(self.checkpoint_dir, self.name + filename_end)
		rospy.logwarn("seperate file = " + str(checkpoint_seperate_file))
		T.save(self.state_dict(), checkpoint_seperate_file)

	def load_checkpoint(self):
		print('... loading checkpoint ...')
		self.load_state_dict(T.load(self.checkpoint_file))
