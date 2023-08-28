# gazeboGymDrone
## Overview 
#### The code in this repository is connected to the (yet to be published) paper 'Towards Faster DRL Training: An Edge AI Approach for UAV Obstacle Avoidance by Splitting Complex Environments'.
#### Follow the bellow steps to get a Gazebo/PX4/Gym based SITL Drone (modeled on the UVify IFO-S drone: https://github.com/decargroup/ifo_gazebo) to employ a Pytorch D3QN based algorithm for obstacle avoidance/autonomous navigation.
## Installation
### Step 1 - Ensure you have a Ubuntu 20.04 Machine (can be a virtual machine such as Oracle VM Virtualbox or Google Cloud Platform)
### Step 2 - Install ROS Noetic by executing the following commands one at a time (based off http://wiki.ros.org/noetic/Installation/Ubuntu):
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop-full -y
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
### Step 3 - Execute the following commands one at a time:
```
sudo apt update
sudo apt-get install python3-catkin-tools -y
sudo apt install git
sudo snap install sublime-text --classic
```
### Step 4 - Create a catkin workspace by executing the following commands (based off noetic tab at http://wiki.ros.org/catkin/Tutorials/create_a_workspace):
```
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin build
source devel/setup.bash
echo $ROS_PACKAGE_PATH
```
### Step 5 - Get the Gazebo Model for the Uvify IFO-S (https://github.com/decargroup/ifo_gazebo):
#### Step 5a - Execute the following commands:
```
cd ~/catkin_ws/src
git clone https://github.com/decarsg/ifo_gazebo.git --recursive
cd ..
catkin config --blacklist px4
catkin build
catkin build
cd ..
bash ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot/Tools/setup/ubuntu.sh
```
#### Step 5b - Relogin or reboot and execute the following commands:
```
sudo apt install python3-pip
```
```
pip3 install pyulog
pip3 install future
sudo apt upgrade -y
```
```
cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make distclean

cd ~
pip3 install --user empy
pip3 install --user packaging
pip3 install --user toml
pip3 install --user numpy
pip3 install --user jinja2

cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make px4_sitl gazebo
#if gazebo black screen then cntrl+c and run make command again
```
```
#cntrl+c
cd ~/catkin_ws/src/ifo_gazebo
rm -r real*
git clone https://github.com/pal-robotics/realsense_gazebo_plugin.git
cd ~/catkin_ws
catkin build
#run catkin build again if previous catkin build returns with a warning
```
#### Step 5c - execute more commands:
```
cd ~
nano ubuntu_sim_ros_noetic.sh
#fill ubuntu_sim_ros_noetic.sh with the contents of https://gist.githubusercontent.com/ekaktusz/a1065a2a452567cb04b919b20fdb57c4/raw/8be54ed561db7e3a2ce61c9c7b1fb9fec72501f4/ubuntu_sim_ros_noetic.sh
#exit and save ubuntu_sim_ros_noetic.sh
bash ubuntu_sim_ros_noetic.sh
#answer 'y' for any prompts
```
#### Step 5d - execute more commands:
```
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/src/ifo_gazebo/setup_ifo_gazebo.bash suppress" >> ~/.bashrc
cd ~/catkin_ws
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
### Step 6 - Get the ROS package that allows a user to communicate with PX4 autopilot using MAVROS by executing the following commands (based off https://github.com/troiwill/mavros-px4-vehicle):
```
#cnrl+c
cd ~/catkin_ws/src
git clone https://github.com/troiwill/mavros-px4-vehicle.git
chmod +x mavros-px4-vehicle/scripts/*.py
chmod +x mavros-px4-vehicle/test/*.py
ln -s mavros-px4-vehicle mavros-px4-vehicle
cd ~/catkin_ws
catkin build
source devel/setup.bash
```
### Step 7 - Install scipy, gym and torch with bellow commands
```
pip3 install scipy
pip3 install gym==0.21
pip3 install torch
```
### Step 8 - Copy the files from this github repository into the appropriate places as outlined bellow
#### Step 8a - Open sublime text
```
cd ~/catkin_ws
subl .
```
#### Step 8b - Add 6 python scripts to '~/catkin_ws/src/mavros-px4-vehicle/scripts'
##### Step 8bi - Navigate to '~/catkin_ws/src/mavros-px4-vehicle/scripts' and create 6 empty files
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
touch {drone_gym_gazebo_env,main,agents,networks,replay_memory,utils}.py
```
##### Step 8bii - Copy and paste the contents of the 6 python files in this repo into their respective files on your computer with Sublime Text (and save before exiting files)
##### Step 8biii - Change 2 lines in 'main.py' that are specific to your computer (highlighted by comments)
##### Step 8biv - Ensure the python scripts are executable
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
chmod +x *.py
```
#### Step 8c - Replace 2 of your launch files
##### Step 8ci - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_empty_world_1.launch' and replace the contents with the file in this repo with the same name (and save). At line 9, change world1.world to world2.world if you want to launch world2 instead of world1.
##### Step 8cii - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_spawn_single.launch' and replace the contents with the file in this repo with the same name (and save)
#### Step 8d - Create a few folders and add two world files to one of them
##### Step 8di -  Create folders at '~/catkin_ws/src/mavros-px4-vehicle'
```
cd ~/catkin_ws/src/mavros-px4-vehicle
mkdir {models,scores,times,plots,worlds}
```
##### Step 8dii - Create 'world1.world' and 'world2.world' files at 'worlds' folder
```
cd ~/catkin_ws/src/mavros-px4-vehicle/worlds
touch {world1,world2}.world
```
##### Step 8diii - Copy and paste the contents of 'world1.world' and 'world2.world' in this repo into the files with the same name on your computer with Sublime Text (and save)
### Step 9 - Launch world and start training
#### Step 9a - Run the bellow commands
```
cd ~
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
#### Step 9b - Open another terminal tab and run the bellow commands
```
cd ~/catkin_ws
source ~/.bashrc
rosrun mavros_px4_vehicle main.py
```
