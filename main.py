#! /usr/bin/env python3

'''
This code is mainly based off https://github.com/RoyElkabetz/DQN_with_PyTorch_and_Gym
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
import rospy
import argparse, os
import copy as cp
from utils import plot_learning_curve
import agents as Agents
import time
import drone_gym_gazebo_env
import torch

if __name__ == '__main__':
    # Allow the following arguments to be passed:
    parser = argparse.ArgumentParser(description = 'Drone StableBaselines')
    parser.add_argument('-train', type=int, default=1, help='1 = training, 0 = playing')
    parser.add_argument('-load_checkpoint', type=int, default=0, help='Load a model checkpoint, set to 1 when playing')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation.')    
    parser.add_argument('-epsilon', type=float, default=1, help='What epsilon starts at')    
    parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-max_mem', type=int, default=20000, help='Maximum size for memory replay buffer')    
    parser.add_argument('-bs', type=int, default=32, help='Batch size for replay memory sampling')    
    parser.add_argument('-eps_min', type=float, default=0.1, help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-eps_dec', type=float, default=0.5*1e-4, help='Linear factor for decreasing epsilon')
    parser.add_argument('-replace', type=int, default=1000, help='Interval for replacing target network')
    parser.add_argument('-algo', type=str, default='DuelingDDQNAgent', choices=['DQNAgent', 'DDQNAgent', 'DuelingDQNAgent', 'DuelingDDQNAgent'])  
    parser.add_argument('-n_games', type=int, default=2500)
    parser.add_argument('-render', type=bool, default=False)
    ############### Change the bellow 2 lines to match your system ###############
    parser.add_argument('-root', type=str, default='/home/patmc/catkin_ws/src/mavros-px4-vehicle/', help='root path for saving/loading')    
    parser.add_argument('-path', type=str, default='/home/patmc/catkin_ws/src/mavros-px4-vehicle/models/', help='path for model saving/loading')  
    ##############################################################################
    args = parser.parse_args()

    # Initialize/create ROS node and custom gym environment:
    rospy.init_node('train_node', anonymous=True)
    env_name = 'DroneGymGazeboEnv-v0'
    env = gym.make(env_name)

    # Setup files/arrays for saving/loading:
    fname = args.algo + '_env1A'
    if args.train:
        scores_file = args.root + 'scores/' + fname + '_scores.npy'
        steps_file = args.root + 'scores/' + fname + '_steps.npy'
        eps_history_file = args.root + 'scores/' + fname + '_eps_history.npy'
        figure_file_start = args.root + 'plots/' + fname
        figure_file = figure_file_start + '.png'    
        times_file = args.root + 'times/' + fname + '_times.txt'
        scores, eps_history, steps_array, times_array = [], [], [], []

    # Set up agent:
    agent_ = getattr(Agents, args.algo)
    if args.train:
        agent = agent_(gamma=args.gamma,
                       epsilon=args.epsilon,
                       lr=args.lr,
                       n_actions=env.action_space.n,
                       image_input_dims=(env.image_observation_space.shape),
                       mem_size=args.max_mem,
                       batch_size=args.bs,
                       eps_min=args.eps_min,
                       eps_dec=args.eps_dec,
                       replace=args.replace,
                       algo=args.algo,
                       env_name=env_name,
                       fname = fname,
                       chkpt_dir=args.path)
    else:
        agent = agent_(gamma=args.gamma,
                       epsilon=0.0,
                       lr=args.lr,
                       n_actions=env.action_space.n,
                       image_input_dims=(env.image_observation_space.shape),
                       mem_size=args.max_mem,
                       batch_size=args.bs,
                       eps_min=0.0,
                       eps_dec=0.0,
                       replace=args.replace,
                       algo=args.algo,
                       env_name=env_name,
                       fname = fname,
                       chkpt_dir=args.path)

    if args.load_checkpoint:
        agent.load_models() #load Q models

    if not args.train:
        num_exceeded_workspace = 0
        num_collided = 0
        num_reached_des_point = 0

    # Training/Playing
    best_score = -np.inf
    n_steps = 0
    start_time = time.time()
    for episode in range(args.n_games):
        episode += 1 #want first episode to be one not zero
        done = False
        score = 0
        observation = env.reset()
        rospy.logwarn("##############################")
        rospy.logwarn("Episode = " + str(episode))
        rospy.logwarn("##############################")
        while not done:
            rospy.loginfo("Episode = " + str(episode) + ", n_steps = " + str(n_steps))
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            score += reward
            if args.render:
                env.render()
            if args.train:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            
            observation = observation_
            n_steps += 1
        
        if args.train:
            scores.append(score)#scores equivalent to total_rewards
            steps_array.append(n_steps)
            eps_history.append(agent.epsilon)
            mean_score = np.mean(scores[-100:])

            print('Episode ', episode, 'score: ', score, 'average score %.1f best average score %.1f epsilon %.2f' %
                  (mean_score, best_score, agent.epsilon), 'steps ', n_steps)

            if mean_score > best_score:
                agent.save_models()
                best_score = mean_score
        else:
            if env.get_has_drone_exceeded_workspace():
                num_exceeded_workspace += 1
            if env.get_has_drone_collided():
                num_collided += 1
            if env.get_has_reached_des_point():
                num_reached_des_point += 1
            print('Episode', episode, ' Summary: exceeded workspace', num_exceeded_workspace, ', collided', num_collided, ', reached dest', num_reached_des_point)    
    
    #Save data and plot learning curve if in train mode:
    if args.train:
        # Save training data:
        times_array.append(round(time.time() - start_time, 2))
        np.savetxt(times_file, times_array, delimiter=',')  
        # Plot learning curve:
        plot_learning_curve(steps_array, scores, eps_history, figure_file)
    else:
        print('Average Steps = ' + str(n_steps/args.n_games))

    # Make the drone land
    env.land_disconnect_drone()
    rospy.logwarn("end")
