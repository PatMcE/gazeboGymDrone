# gazeboGymDrone
## Overview 
#### The code in this repository is connected to the (yet to be published) paper 'Scalable Deep Reinforcement Learning Based UAV Obstacle Avoidance using Edge AI'.
#### Follow the bellow steps to get a Gazebo/PX4/OpenAI Gym based SITL Drone (modeled on the UVify IFO-S drone: https://github.com/decargroup/ifo_gazebo) to employ a Pytorch D3QN based algorithm for obstacle avoidance/autonomous navigation.
## Installation
### Step 1 - Ensure you have Ubuntu 18.04 (e.g. through a virtual machine)
### Step 2 - Upgrade to python 3.7 by executing the following commands one at a time (based off: https://cloudbytes.dev/snippets/upgrade-python-to-latest-version-on-ubuntu-linux):
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
apt list | grep python3.7
#You should see python3.7 in red many times
sudo apt install python3.7 -y
sudo nano /usr/bin/gnome-terminal
#change #!/usr/bin/python3 to #!/usr/bin/python3.7 then cntrl+X, Y and press enter
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
sudo update-alternatives --config python3
#(select one of python 3.7 options)
sudo apt remove --purge python3-apt -y
sudo apt autoclean
sudo apt install python3-apt
sudo apt install python3.7-distutils -y
sudo apt install curl -y
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.7 get-pip.py
```
### Step 3 - Install ROS Melodic by executing the following commands one at a time (based off http://wiki.ros.org/melodic/Installation/Ubuntu):
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt-get update -y
sudo apt install ros-melodic-desktop-full -y
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
### Step 4 - Do various installs by executing the following commands one at a time:
```
sudo apt install python3-pip python3-all-dev python3-rospkg -y
sudo apt install ros-melodic-desktop-full --fix-missing -y
sudo apt update
sudo apt-get install python3-catkin-tools -y
sudo apt install git -y
sudo apt upgrade libignition-math2 -y
sudo snap install sublime-text --classic
```
### Step 5 - Create a catkin workspace by executing the following commands (based off http://wiki.ros.org/catkin/Tutorials/create_a_workspace):
```
source /opt/ros/melodic/setup.bash
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
#### Step 5b - Restart computer and execute the following commands:
```
sudo apt install python3-pip
```
```
pip3 install pyulog
pip3 install future
sudo apt upgrade -y

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

cd ~
wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.sh
bash ubuntu_sim_ros_melodic.sh
```
```
sudo rosdep init
rosdep update
bash ubuntu_sim_ros_melodic.sh
```
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

### Step 7 - Install opencv, scipy, gym and torch with bellow commands
```
pip3 install opencv-python
pip3 install scipy
pip3 install gym==0.15.7
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
##### Step 8ci - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_empty_world_1.launch' and replace the contents with the file in this repo with the same name (and save)
##### Step 8cii - Using Sublime Text, navigate to '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/ifo_spawn_single.launch' and replace the contents with the file in this repo with the same name (and save)
 
#### Step 8d - Create a few folders and add a world file to one of them
##### Step 8di -  Create folders at '~/catkin_ws/src/mavros-px4-vehicle'
```
cd ~/catkin_ws/src/mavros-px4-vehicle
mkdir {models,scores,times,plots,worlds}
```
##### Step 8dii - Create 'world1.world' file at 'worlds' folder
```
cd ~/catkin_ws/src/mavros-px4-vehicle/worlds
touch world1.world
```
##### Step 8diii - Copy and past the contents of 'world1.world' in this repo into the file with the same name on your computer with Sublime Text (and save)

### Step 9 - Launch world and start training
#### Step 9a - Run the bellow commands
```
cd ~/catkin_ws
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
#### Step 9b - Open another terminal tab and run the bellow command
```
source ~/.bashrc
rosrun mavros_px4_vehicle main.py
```