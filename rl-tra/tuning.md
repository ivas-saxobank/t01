# Tuning PPO

To tune the PPO model, you can adjust several hyperparameters.
Here are some common ones you might consider:

- Learning Rate (learning_rate): Controls how much to change the model in response to the estimated error each time the model weights are updated.
- Batch Size (batch_size): Number of samples per gradient update.
- Number of Epochs (n_epochs): Number of epochs to update the policy.
- Gamma (gamma): Discount factor for future rewards.
- Clip Range (clip_range): Clipping parameter for PPO.
- GAE Lambda (gae_lambda): Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
- Policy Network Architecture (policy_kwargs): Custom architecture for the policy network.

```python
policy_kwargs = dict(
    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
)
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=verbose,
    learning_rate=3e-4,  # Adjust learning rate
    batch_size=64,       # Adjust batch size
    n_epochs=10,         # Adjust number of epochs
    gamma=0.99,          # Adjust gamma
    clip_range=0.2,      # Adjust clip range
    gae_lambda=0.95,     # Adjust GAE lambda
    policy_kwargs=policy_kwargs  # Custom policy network architecture
)
```

The net_arch parameter in the PPO model is used to define the architecture of the neural network used by the policy and value function.
Specifically, it allows you to customize the number and size of the layers in the policy network (pi) and the value function network (vf).

Here's a breakdown of how it works:

- pi (policy network): This part of the network is responsible for learning the policy, i.e., the mapping from states to actions.
- vf (value function network): This part of the network is responsible for estimating the value function, i.e., the expected return from a given state.

By customizing the net_arch parameter, you can control the complexity and capacity of these networks, which can help in better fitting the model to your specific problem.

For example:

```python
policy_kwargs = dict(
    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
)
```

In this example:

The policy network (pi) has two hidden layers with 64 neurons each.
The value function network (vf) also has two hidden layers with 64 neurons each.
This customization can help in improving the performance of the model by allowing it to better capture the underlying patterns in the data.

The optimal size of the network architecture (net_arch) depends on various factors, including the complexity of your problem, the amount of data, and the specific characteristics of your observations.
However, a common approach is to start with a simple architecture and gradually increase its complexity if the model's performance is not satisfactory.

For N inputs in the observation, a good starting point could be a network with two to three hidden layers, each with a number of neurons that is a multiple of the input size.
For example, if you have N inputs, you might start with hidden layers of size 2N, N, and N/2.

Here's how you can modify your code to include a custom net_arch:

```python
policy_kwargs = dict(
    net_arch=[dict(pi=[2*N, N, N//2], vf=[2*N, N, N//2])]

    # N = 180 * 5 = 1080
    # [2160, 1080, 540]
)
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=verbose,
    learning_rate=3e-4,  # Adjust learning rate
    batch_size=64,       # Adjust batch size
    n_epochs=10,         # Adjust number of epochs
    gamma=0.99,          # Adjust gamma
    clip_range=0.2,      # Adjust clip range
    gae_lambda=0.95,     # Adjust GAE lambda
    policy_kwargs=policy_kwargs  # Custom policy network architecture
)
```

In this example:

- The policy network (pi) and the value function network (vf) each have three hidden layers.
- The sizes of the hidden layers are 2*N, N, and N//2.

You can adjust these sizes based on the performance of your model.
If the model is underfitting, you might increase the number of neurons or add more layers.
If the model is overfitting, you might reduce the number of neurons or layers.

The clip_range parameter in the PPO (Proximal Policy Optimization) model is used to control the clipping of the policy update.
This is a crucial part of the PPO algorithm, which helps to ensure that the new policy does not deviate too much from the old policy during training.

- Purpose: The clip_range parameter helps to stabilize training by limiting the extent to which the policy can change in a single update. This prevents large updates that could destabilize the learning process.
- Mechanism: PPO uses a clipped objective function to penalize changes to the policy that are too large. The clip_range parameter defines the threshold for this clipping. If the probability ratio between the new and old policy is outside the range [1 - clip_range, 1 + clip_range], the objective function is clipped to this range.
- Effect: By clipping the policy updates, PPO ensures that the updates are more conservative, which can lead to more stable and reliable training.

The gamma parameter in the PPO (Proximal Policy Optimization) model is the discount factor for future rewards.
It is a crucial parameter in reinforcement learning algorithms that determines how much importance the agent places on future rewards compared to immediate rewards.

- Purpose: The gamma parameter controls the trade-off between short-term and long-term rewards. A lower gamma value makes the agent focus more on immediate rewards, while a higher gamma value makes the agent consider future rewards more heavily.
- Range: The gamma parameter typically ranges between 0 and 1. A value of 0 means the agent only considers immediate rewards, while a value of 1 means the agent considers rewards far into the future without discounting them.
- Effect: The choice of gamma can significantly impact the learning process and the resulting policy. A well-chosen gamma can help the agent learn a more balanced policy that optimizes long-term returns.
  
The gamma parameter is set to 0.99, meaning the agent will consider future rewards with a discount factor of 0.99. This encourages the agent to optimize for long-term rewards while still considering immediate rewards.

The gae_lambda parameter in the PPO (Proximal Policy Optimization) model is used to control the trade-off between bias and variance in the Generalized Advantage Estimation (GAE) method.
GAE is a technique used to compute the advantage function, which helps in reducing the variance of the policy gradient estimates while maintaining a low bias.

- Purpose: The gae_lambda parameter helps to balance the bias-variance trade-off in the advantage estimation. It determines how much weight to give to the future rewards when estimating the advantage.
- Range: The gae_lambda parameter typically ranges between 0 and 1. A value of 0 means that only the immediate reward is considered (high bias, low variance), while a value of 1 means that all future rewards are considered (low bias, high variance).
- Effect: By adjusting gae_lambda, you can control the smoothness of the advantage estimates. A well-chosen gae_lambda can lead to more stable and efficient training.
  
The gae_lambda parameter is set to 0.95, which is a common choice that provides a good balance between bias and variance. This means that the advantage estimates will consider future rewards with a slight discount, leading to more stable training.

The batch_size parameter in the PPO (Proximal Policy Optimization) model specifies the number of samples used in each gradient update during the training process.
It is an important hyperparameter that can affect the stability and efficiency of the training.

- Purpose: The batch_size parameter determines how many samples (or experiences) are used to compute each gradient update. This affects how the model's weights are adjusted during training.
- Effect on Training:
  - Smaller Batch Size: Leads to more frequent updates with higher variance in the gradient estimates. This can help the model escape local minima but may also lead to less stable training.
  - Larger Batch Size: Leads to less frequent updates with lower variance in the gradient estimates. This can result in more stable training but may require more memory and computational resources.
- Typical Values: Common values for batch_size range from 32 to 256, but the optimal value can depend on the specific problem and the amount of available data.

The batch_size parameter is set to 64, meaning that each gradient update will be computed using 64 samples.
You can adjust this value based on your specific needs and the performance of the model.

The n_epochs parameter in the PPO (Proximal Policy Optimization) model specifies the number of times the entire batch of data is passed through the neural network during each update.
This parameter is crucial for controlling the amount of learning that happens in each update cycle.

- Purpose: The n_epochs parameter determines how many times the collected batch of experiences is used to update the policy and value networks. Each epoch involves a full pass through the batch, performing gradient descent updates.
- Effect on Training:
  - Higher n_epochs: More epochs mean that the model will perform more updates on the same batch of data, which can lead to better convergence but also increases the risk of overfitting to the batch.
  - Lower n_epochs: Fewer epochs mean that the model will perform fewer updates on the same batch of data, which can lead to faster training but might result in underfitting.
- Typical Values: Common values for n_epochs range from 3 to 10, but the optimal value can depend on the specific problem and the amount of available data.

The n_epochs parameter is set to 10, meaning that the model will perform 10 passes through the batch of data during each update. You can adjust this value based on your specific needs and the performance of the model.

The learning_rate parameter in the PPO (Proximal Policy Optimization) model controls the step size at each iteration while moving towards a minimum of the loss function.
It is a crucial hyperparameter that affects the speed and quality of the learning process.

- Purpose: The learning_rate parameter determines how much to adjust the model's weights in response to the estimated error each time the model weights are updated. It essentially controls the magnitude of the updates to the model parameters.
- Effect on Training:
  - Higher Learning Rate: Leads to larger updates and can speed up the training process. However, if the learning rate is too high, it can cause the training to become unstable and the model may fail to converge.
  - Lower Learning Rate: Leads to smaller updates and can result in more stable training. However, if the learning rate is too low, the training process can become very slow and may get stuck in local minima.
- Typical Values: Common values for learning_rate range from 1e-5 to 1e-3, but the optimal value can depend on the specific problem and the architecture of the model.

The learning_rate parameter is set to 3e-4, meaning that the model's weights will be updated with a step size of 0.0003 during each iteration.
You can adjust this value based on your specific needs and the performance of the model.

in code in active tab, I'm training the PPO model using stable_baselines3 implementation.
My observation data is a dictionary of Open, High, Low, Close prices and Volume, which are arrays of length N, so 5*N floating point numbers per observation.
On each step, actions are Buy, Sell or Hold. The reward is the profit or loss with regards to initial amount of money.
But reward on each step doesn't matter that much, what matters is that after, say, 100 steps we have a profit, which we want to optimize.
Can you suggest PPO parameter values for this learning? Please consider (1) net_arch (2) batch_size (3) n_epochs (4) clip_range (5) gae_lambda (6) gamma (7) learning_rate

Given your problem setup, where the observation data consists of financial time series data and the goal is to optimize profit over a longer horizon (e.g., 100 steps), here are some suggested PPO parameter values:

net_arch: Since your observation data is relatively complex (5*N floating point numbers), a deeper network might be beneficial. You can start with a network architecture that has three hidden layers with sizes proportional to the input size.

batch_size: A larger batch size can help stabilize training, especially with financial data. Start with a batch size of 128.

n_epochs: Since you want to optimize over a longer horizon, more epochs can help the model learn better. Start with 10 epochs.

clip_range: A typical value for the clip range is 0.2, which works well in many scenarios.

gae_lambda: A value of 0.95 is commonly used and provides a good balance between bias and variance.

gamma: Since you are interested in long-term rewards, a higher gamma value like 0.99 is appropriate.

learning_rate: A learning rate of 3e-4 is a good starting point for PPO.

```python
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
)

batch_size=128
n_epochs=10
clip_range=0.2
gae_lambda=0.95
gamma=0.99
learning_rate=3e-4
```

These parameter values should provide a good starting point for training your PPO model on financial time series data.
You may need to further tune these parameters based on the performance of your model.

The total_timesteps parameter in the model.learn() method specifies the total number of timesteps to train the model.
Given that your profit-optimizing horizon is 100 steps, you want to ensure that the model has enough data to learn effectively.

Here are some considerations for setting total_timesteps:

- Sufficient Training Data: The model needs enough data to learn the patterns in the financial time series. Training for too few timesteps might result in underfitting, while training for too many timesteps might lead to overfitting or unnecessary computational expense.
- Exploration and Exploitation: The model needs to explore different actions (Buy, Sell, Hold) to learn an optimal policy. More timesteps allow for better exploration.
- Computational Resources: Training for a very large number of timesteps can be computationally expensive. You need to balance the training time with the available computational resources.

A good starting point could be to set total_timesteps to a value that allows the model to experience a wide variety of market conditions.
For example, you might start with total_timesteps set to 100 (your horizon) times 10000 repetitions, which equals 1,000,000 timesteps.

Yes, it can make sense to use RecurrentPPO from sb3_contrib instead of plain PPO for your task, especially given the nature of financial time series data. Here's why:

- Sequential Data: Financial time series data (Open, High, Low, Close prices, and Volume) is inherently sequential. The current state often depends on previous states. Recurrent neural networks (RNNs), such as LSTMs (Long Short-Term Memory networks), are designed to handle such sequential dependencies.
- Memory: RecurrentPPO uses LSTM layers, which can maintain a memory of previous observations. This can be particularly useful in trading tasks where the decision to buy, sell, or hold might depend on patterns observed over several time steps.
- Long-Term Dependencies: LSTMs can capture long-term dependencies in the data, which is beneficial for optimizing profit over a longer horizon (e.g., 100 steps).

Here's how you can modify your code to use RecurrentPPO from sb3_contrib.
In this example:

- RecurrentPPO is used instead of plain PPO.
- The policy is set to 'MlpLstmPolicy', which includes LSTM layers.
- policy_kwargs includes lstm_hidden_size to configure the size of the LSTM hidden layers.

Using RecurrentPPO can help your model better capture the sequential dependencies in financial time series data, potentially leading to better performance in your trading task.

```python
policy_kwargs = dict(
                net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])],
                lstm_hidden_size=128  # Adjust LSTM hidden size as needed
            )
            model = RecurrentPPO(
                'MlpLstmPolicy', 
                env, 
                verbose=verbose,
                learning_rate=3e-4,  # Adjust learning rate
                batch_size=128,      # Adjust batch size
                n_epochs=10,         # Adjust number of epochs
                gamma=0.99,          # Adjust gamma
                clip_range=0.2,      # Adjust clip range
                gae_lambda=0.95,     # Adjust GAE lambda
                policy_kwargs=policy_kwargs  # Custom policy network architecture
            )
```
