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
    parser.add_argument('-train', type=int, default=1, help='True = training, False = playing')
    parser.add_argument('-load_checkpoint', type=int, default=0, help='Load a model checkpoint')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation.')    
    parser.add_argument('-epsilon', type=float, default=1, help='What epsilon starts at')    
    parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-max_mem', type=int, default=20000, help='Maximum size for memory replay buffer')    
    parser.add_argument('-bs', type=int, default=128, help='Batch size for replay memory sampling')    
    parser.add_argument('-eps_min', type=float, default=0.1, help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-eps_dec', type=float, default=0.5*1e-4, help='Linear factor for decreasing epsilon')#1e-4=0.0001,1e-5, 1e-4=0.0001,600 games
    parser.add_argument('-replace', type=int, default=1000, help='Interval for replacing target network')
    parser.add_argument('-algo', type=str, default='DuelingDDQNAgent', choices=['DQNAgent', 'DDQNAgent', 'DuelingDQNAgent', 'DuelingDDQNAgent'])  
    parser.add_argument('-n_games', type=int, default=2500)#800
    parser.add_argument('-total_steps', type=int, default=60)#20000)    
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

    # Setup files for saving/loading:
    fname = args.algo + '_env1ABC_lr' + str(args.lr) + '_bs' + str(args.bs) + '_episodes' + str(args.n_games)
    log_path = args.root + 'logs/'
    scores_file = args.root + 'scores/' + fname + '_scores.npy'
    steps_file = args.root + 'scores/' + fname + '_steps.npy'
    eps_history_file = args.root + 'scores/' + fname + '_eps_history.npy'
    figure_file_start = args.root + 'plots/' + fname
    figure_file = figure_file_start + '.png'    
    times_file = args.root + 'times/' + fname + '_times.txt'

    # Various intializations:
    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma,
                   epsilon=args.epsilon,
                   lr=args.lr,
                   n_actions=env.action_space.n,
                   image_input_dims=(env.image_observation_space.shape),
                   mem_size=args.max_mem,
                   batch_size=args.bs,
                   eps_min=args.eps_min,
                   eps_dec=args.eps_dec,#1e-4,600 games
                   replace=args.replace,
                   total_steps=args.total_steps,
                   algo=args.algo,
                   env_name=env_name,
                   fname = fname,
                   chkpt_dir=args.path)
    n_steps = 0
    scores, eps_history, steps_array, times_array = [], [], [], []

    if args.load_checkpoint:
        agent.load_models() #load Q models

    # Training/Playing
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

        scores.append(score)#scores equivalent to total_rewards
        steps_array.append(n_steps)
        eps_history.append(agent.epsilon)
        mean_score = np.mean(scores[-100:])

        print('episode ', episode, 'score: ', score, 'average score %.1f best average score %.1f epsilon %.2f' %
              (mean_score, best_score, agent.epsilon), 'steps ', n_steps)

        if mean_score > best_score:
            if args.train:
                agent.save_models()
            best_score = mean_score
            
    #Save data and plot learning curve if in train mode:
    if args.train:
        # Save training data:
        times_array.append(round(time.time() - start_time, 2))
        np.savetxt(times_file, times_array, delimiter=',')  
        # Plot learning curve:
        plot_learning_curve(steps_array, scores, eps_history, figure_file)

    # Make the drone land
    env.land_disconnect_drone()
    rospy.logwarn("end")