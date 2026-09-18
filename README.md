# Deep Reinforcement Learning on Game Environments

This project explores and compares different deep reinforcement learning algorithms across LunarLander, Assault, Super Mario Bros, and a custom Cubefield environment.

The main goal was to implement different RL approaches, modify them, and compare how those changes affect training performance. The project covers policy gradient methods, Deep Q-Learning, actor-critic methods, distributional RL, and attention-based networks.

## Environments

### LunarLander

LunarLander was used to experiment with Policy Gradient methods and investigate how changes to the basic algorithm affect training.

The following approaches were compared:

- Baseline Policy Gradient / REINFORCE
- Policy Gradient + Reward Normalization
- Reward Normalization + PPO

The baseline method showed high variance during training. Reward normalization was added to reduce the effect of large differences in returns, while PPO-style clipped updates were tested to prevent excessively large policy updates.

### Assault

Assault was used to compare different extensions of Deep Q-Learning.

The following algorithms were tested:

- DQN
- Double DQN
- Dueling DQN
- Double Dueling DQN

DQN was used as the baseline. Double DQN separates action selection from target evaluation to reduce Q-value overestimation. Dueling DQN separates the estimation of state value and action advantage. Double Dueling DQN combines both modifications.

The objective was to see how these changes affected learning and reward performance in a faster, image-based Atari environment.

### Super Mario Bros

Super Mario Bros was used to experiment with actor-critic reinforcement learning using A2C.

The actor determines the action policy while the critic estimates the value of the current state. Mario provided a more difficult environment because actions can have delayed consequences and the agent must learn both movement and exploration over longer sequences.

### Cubefield

Cubefield was used as a custom environment for testing more specialized RL architectures.

The objective of Cubefield is to control the player while moving through a field of incoming cubes without colliding with them. The agent is rewarded for surviving longer, so it must learn when and where to move as obstacles approach.

The following algorithms were compared:

- PPO-Dueling-DQN
- Quantile Regression DQN (QR-DQN)
- Attention DQN
- A3C

**PPO-Dueling-DQN** combines PPO-style clipped updates with a dueling architecture that separates state value from action advantage. This was used to improve training stability while allowing the network to distinguish between the quality of a state and the usefulness of individual actions.

**Quantile DQN** predicts a distribution of possible future returns rather than a single expected Q-value. Quantile regression was used to investigate whether modeling the return distribution could make the agent more robust to the uncertainty created by different obstacle configurations.

**Attention DQN** adds a multi-head attention mechanism to the DQN architecture. The goal was to allow the network to learn which parts of the state are most relevant when deciding how to avoid incoming obstacles.

**A3C** uses an actor-critic architecture where the actor learns the action policy and the critic estimates state values. The advantage calculated from the critic is then used to improve the policy updates.

## Results

Performance was evaluated primarily using episode reward curves during training.

For LunarLander, reward normalization produced a clear improvement over the baseline Policy Gradient implementation, while adding PPO-style updates further changed the stability and performance of the learned policy.

For Assault, the different DQN architectures were compared to determine the effect of Double Q-Learning and the Dueling architecture on training performance.

For Cubefield, PPO-Dueling-DQN, QR-DQN, Attention DQN, and A3C were trained and compared using the same environment. The results showed noticeable differences in learning speed, stability, and peak episode reward between the algorithms.

These experiments were used to understand how changes to the underlying RL algorithm affect learning rather than only trying to maximize the final game score.

## Tools and Libraries

- Python
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib
- PyGame
- Atari Learning Environment
- gym-super-mario-bros

## RL Concepts Used

- Deep Q-Learning
- Experience Replay
- Target Networks
- Epsilon-Greedy Exploration
- Double Q-Learning
- Dueling Networks
- Policy Gradients
- Reward Normalization
- PPO Clipped Updates
- Actor-Critic Learning
- Advantage Estimation
- Distributional Reinforcement Learning
- Quantile Regression
- Attention Mechanisms
- Gradient Clipping

## Running the Project

First install the required dependencies:

```bash
pip install -r requirements.txt
```

### Policy Gradient

To train a Policy Gradient agent:

```bash
python main.py --train_pg --pg_type=pg_nor --folder_name=[your_folder_name]
```

To test a trained Policy Gradient model:

```bash
python main.py --test_pg --model_path=./model/best/pg.cpt
```

### Deep Q-Learning

To train a DQN agent:

```bash
python main.py --train_dqn --dqn_type=DoubleDQN --folder_name=[your_folder_name]
```

To test a trained DQN model:

```bash
python main.py --test_dqn --dqn_type=DDDQN --model_path=./model/best/dqn.cpt
```

The `dqn_type` argument can be changed depending on the DQN variant being tested.

### Cubefield

The Cubefield experiments are contained in a separate Python script.

From the project directory, run:

```bash
python Cubefield.py
```

Alternatively, run it using the complete file path:

```bash
python [FILE_PATH]/Cubefield.py
```

The script trains the Cubefield agents and generates learning curves that can be used to compare their episode rewards over training.

## Project Structure

```text
Deep-Reinforcement-Learning-on-Atari-Games/
│
├── main.py
├── Cubefield.py
├── requirements.txt
├── model/
│   └── best/
├── README.md
└── ...
```

## Summary

This project gave me hands-on experience implementing and modifying reinforcement learning algorithms rather than only using existing RL libraries. I experimented with how changes such as reward normalization, Double Q-Learning, dueling architectures, PPO clipping, quantile regression, and attention affect the behavior and training stability of RL agents across different game environments.