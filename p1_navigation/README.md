# Project 1: Navigation

This repository contains my solution to the Navigation problem from the Udacity Deep Reinforcement Learning Nanodegree. It is a deep Q-network with three enhancements: Double DQN, Dueling DQN, and Prioritized Experience Replay.

Please see the `Report.md` file for more details on results.

## Problem Overview

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.

3. I also needed to execute the following steps to get the code to work on my machine (a Mac).

```shell
conda create --name drlnd python=3.6 ipython
conda activate drlnd
conda install swig
pip install Box2D
pip install mlagents==0.4
pip install unityagents
pip install . # in the udacity course python folder
```

If you don't want to do `pip install .`, you can also put the following list in a requirements.txt file and use `pip install -r requirements.txt`

```
tensorflow==1.7.1
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==0.4.0
pandas
scipy
ipykernel
```

### Instructions

1. Run `jupyter lab` or `jupyter notebook`
2. View `Navigation_v2.ipynb` file and follow the instructions.

### Relevant Files

* `dqn_agent_v2.py`: Is the file with the Agent Class
* `model.py`: Is the PyTorch model file
* `Navigation_v2.ipynb`: The notebook that runs the different variations on the agent
* `prioritized_memory.py`: The model for running prioritized replay
* `Report.md`: A report on my findings from running different variants on the model as well as additional details explaining the model components.
* `SumTree.py`: The tree data structure used to hold the prioritized experience replay experiences, which allows faster log(n) time to access the data.
