# UniTreeGo1_ReinforceLearning_Controller

## Project Overview

This project implements a reinforcement learning controller for the [UniTree Go1](https://shop.unitree.com/products/unitreeyushutechnologydog-artificial-intelligence-companion-bionic-companion-intelligent-robot-go1-quadruped-robot-dog) quadruped robot. The system utilizes reinforce learning algorithms PPO to enable controlled locomotion for the Go1 robot platform.

## Features

* **Reinforcement Learning Integration**: PPO Implementation for Quadruped Robot Motion Control Using Custom Actor-Critic Network Hidden Layer Parameters

* **State Machine Gait Control**: Utilizing a state machine to guide the transition of robotic gaits, replacing traditional gait phase methods.

* **Multiple Gait Options**: The agent will autonomously select a gait based on the control speed.

* **Curriculum Learning**: Applying differently distributed commands and gait rewards at different stages ultimately achieves smooth gait transitions and high-speed forward movement.

* **Real-time Control Demonstration**: Capture keyboard control commands and publish to ROS topics to achieve real-time control.


## Demo

![demo](output/2025-12-21_22-43-26/demo_2025-12-21_22-43-26_20/mjc_2025-12-21_22-43-26_20%5B3%5D.gif)


## System Architecture

### Overall Architecture Diagram

<!-- Insight Agent Private Database Mining: AI agent for in-depth analysis of private public opinion databases

Media Agent Multimodal Content Analysis: AI agent with powerful multimodal capabilities

Query Agent Precise Information Search: AI agent with domestic and international web search capabilities

Report Agent Intelligent Report Generation: Multi-round report generation AI agent with built-in templates -->

### Project Code Structure Tree

```
├── assets/                     # Mujoco simulation environment assets
│   ├── mujoco_menagerie/       # Assets of google-deepmind for robot physics
|   └── model/                  # Assets customized based on google-deepmind resources
├── output/                     # Output directory for training results and logs
│   ├── checkpoint_tree.json    # Inheritance relationships and descriptions of checkpoints in JSON format
│   ├── checkpoint_tree.txt     # Inheritance relationships and descriptions of checkpoints in txt format
│   ├── YYYY-MM-DD_hh-mm-ss/    # Checkpoint files represented as YYYY-MM-DD_hh-mm-ss
│   │   ├── bkp_YYYY-MM-DD_hh-mm-ss_n.py    # Backup of training script unitree_go1.py of n-th checkpoint
│   │   ├── cfg_YYYY-MM-DD_hh-mm-ss_n.yaml  # Configuration file for this training run
│   │   ├── env_YYYY-MM-DD_hh-mm-ss_n.pkl   # Saved vec normalize env for reproducibility
│   │   ├── mdl_YYYY-MM-DD_hh-mm-ss_n.zip   # Trained PPO model weights archive
│   │   ├── log_YYYY-MM-DD_hh-mm-ss_n/      # Training tensorboard logs
│   │   ├── demo_YYYY-MM-DD_hh-mm-ss_n/     # Demo recordings from training session
│   │   │   ├── mjc_YYYY-MM-DD_hh-mm-ss_n[k].gif    # k-th Mujoco simulation demo animation
│   │   │   ├── plt_YYYY-MM-DD_hh-mm-ss_n[k].gif    # Training performance plots
│   │   │   └── ...
│   │   └── ...
|   └── ...
├── src/                                # Source code for the project
│   ├── callbacks/                      # Training callback functions
│   │   ├── calbacks.py                 # Aggregation of Callback Functions
│   │   ├── adaptive_learning_rate.py   # Dynamic learning rate adjustment
│   │   ├── custom_checkpoint.py        # Custom checkpoint saving
│   │   ├── custom_tensorboard.py       # Custom tensorboard logging
│   │   ├── progress_bar.py             # Training progress visualization
│   │   └── stage_schedule.py           # Curriculum learning stage management
│   ├── configs/            # Configuration file and methods
│   │   ├── config.py       # Configuration methods
│   │   └── config.yaml     # Configuration file
│   ├── env/                # Environment implementation
│   │   ├── control.py      # Robot control interface
│   │   ├── make_env.py     # Environment creation and setup
│   │   ├── rewards.py      # Reward function definitions
│   │   └── unitree_go1.py  # UniTree Go1 specific environment
│   ├── renders/    # Visualization and rendering utilities
│   ├── runners/    # Training and execution runners
│   └── utils/      # Utility functions and helpers
├── .gitignore      # Git ignore rules
├── .gitmodules     # Git submodules configuration
├── main.py         # Examples of main entry points for training/demonstration
└── README.md       # Project documentation
```

## Prerequisites

<!-- Python 3.x
Robot Operating System (ROS) - if applicable to your setup
NumPy, PyTorch/TensorFlow for ML components
UniTree Go1 SDK and drivers
Appropriate hardware dependencies -->

## Quick Start

<!-- Clone the repository:

bash
git clone https://github.com/your-username/UniTreeGo1_ReinforceLearning_Controller.git
Install required dependencies:

bash
pip install -r requirements.txt
Configure the UniTree Go1 SDK according to manufacturer documentation

Ensure the UniTree Go1 robot is properly connected and calibrated
Run the main controller script:
bash
python main_controller.py
The reinforcement learning model will begin training/operating based on the configured parameters -->


