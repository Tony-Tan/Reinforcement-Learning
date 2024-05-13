# Reinforcement Learning ![](https://img.shields.io/github/stars/Tony-Tan/Reinforcement-Learning?style=social)

## Introduction
Welcome to the Reinforcement Learning project! This project focuses on exploring and implementing different reinforcement learning algorithms to solve complex tasks. By studying and implementing state-of-the-art algorithms, we aim to gain insights into the theoretical foundations of reinforcement learning and apply them to real-world problems.

In this project, we provide step-by-step installation instructions to help you set up the project and its dependencies. Once installed, you can explore the code structure, which includes different algorithms implemented in separate Python files under the `algorithms/` directory.

Feel free to contribute to this project by adding new algorithms, improving existing code, or providing bug fixes. Check out the "Contributing" section for more information on how to contribute.

Now, let's get started with the installation and exploration of this Reinforcement Learning project!

## Table of Contents
- [Background and Theoretical Framework](#background-and-theoretical-framework)
- [Installation and Dependencies](#installation-and-dependencies)
- [Papers to Code](#papers-to-code)
- [Usage](#usage)
- [Code Structure and Explanation](#code-structure-and-explanation)
- [Contributing](#contributing)
- [Performance and Benchmarks](#performance-and-benchmarks)
- [License](#license)
- [Acknowledgments and References](#acknowledgments-and-references)

## Background and Theoretical Framework
Reinforcement learning is a subfield of machine learning that focuses on training agents to make sequential decisions in an environment to maximize a reward signal. It has gained significant attention in recent years due to its potential applications in various domains, including robotics, game playing, and autonomous systems.

This project aims to explore and implement different reinforcement learning algorithms to solve complex tasks. By studying and implementing state-of-the-art algorithms, we can gain insights into the theoretical foundations of reinforcement learning and apply them to real-world problems.

By understanding the background and theoretical framework of reinforcement learning, we can better appreciate the algorithms and techniques used in this project and their significance in the field.

## Step-by-step installation instructions
To install and run this project, follow these steps:

1. Clone the repository:
    ```
    git clone git@github.com:Tony-Tan/Reinforcement-Learning.git
    ```

2. Navigate to the project directory:
    ```
    cd Reinforcement-Learning
    ```

3. Create a new conda environment:
    ```
    conda create -n rl python=3.10
    ```

4. Activate the conda environment:
    ```
    conda activate rl
    ```

5. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

Now you have successfully installed the project and its dependencies. You can proceed with using and exploring the code.


## Papers to Code

| No | Year |      Status      | Name                                                                                                  |
|:--:|:----:|:----------------:|:------------------------------------------------------------------------------------------------------|
| 1  | 1951 |  🚧 Developing   | [A Stochastic Approximation Method]()                                                                 |
| 2  | 1986 |  🚧 Developing   | [Stochastic approximation for Monte Carlo optimization]()                                             | 
| 3  | 2001 |  🚧 Developing   | [A natural policy gradient]()                                                                         | 
| 4  | 2013 | 🧪 Experimenting | [Playing Atari with Deep Reinforcement Learning](./algorithms/dqn.py)                                 |
| 5  | 2015 | 🧪 Experimenting | [Human-level control through deep reinforcement learning](./algorithms/dqn.py)                        | 
| 6  | 2015 |  🚧 Developing   | [Trust Region Policy Optimization]()                                                                  | 
| 7  | 2015 |  🚧 Developing   | [Continuous control with deep reinforcement learning]()                                               |
| 8  | 2015 | 🧪 Experimenting | [Deep Reinforcement Learning with Double Q-Learning](./algorithms/double_dqn.py)                      | ![](https://img.shields.io/badge/dynamic/json?label=Citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F3b9732bb07dc99bde5e1f9f75251c6ea5039373e%3Ffields%3DcitationCount) |
| 8  | 2016 | 🧪 Experimenting | [Dueling Network Architectures for Deep Reinforcement Learning](./algorithm/dueling_dqn.py_)          | 
| 9  | 2016 | 🧪 Experimenting | [Prioritized Experience Replay](./algorithms/dqn_pp.py)                                               |
| 10 | 2017 |  🚧 Developing   | [Proximal Policy Optimization Algorithms]()                                                           | 
| 11 | 2018 |  🚧 Developing   | [Addressing Function Approximation Error in Actor-Critic Methods]()                                   | 
| 12 | 2018 |  🚧 Developing   | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor]() |


## Code Structure and Explanation

The main components of the project are:

- `algorithms/`: This directory contains the implementation of various algorithms. Each algorithm is implemented in its own Python file. For example, the DQN algorithm is implemented in `dqn.py`.
- `abc_rl`: This module contains the abstracted classes of reinforcement learning elements.
- `agnets`: 
- `configs`:
- `doc`: 
- `environment`:
- `experience_replay`:
- `exploration`
- `models`:
- `test`:
- `utils`: This directory contains utility functions and classes that are used by the algorithms. This could include functions for data preprocessing, model evaluation, etc.

## Acknowledgments and References
We would like to acknowledge the following resources and references that have been instrumental in the development of this project:

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Richard S. Sutton and Andrew G. Barto
- [OpenAI Gym](https://gym.openai.com/) - A toolkit for developing and comparing reinforcement learning algorithms
- [PyTorch](https://pytorch.org/) - A deep learning framework used for implementing neural networks in this project

We are grateful for the valuable insights and contributions from the open-source community and the authors of the above resources.

## Donations
Running this project involves significant computational resources. If you find this project helpful and would like to support its continued development, consider making a donation. Your support is greatly appreciated!

You can donate through the following platforms:

- WeChat: Please scan the QR code below to donate via WeChat.
  ![WeChat QR Code](./attachments/wechat.png)

- Alipay: Please scan the QR code below to donate via Alipay.
  ![Alipay QR Code](./attachments/alipay.png)

- PayPal: Please click [here](https://paypal.me/TonySTan?country.x=C2&locale.x=zh_XC) to donate via PayPal.

Thank you for your support!
