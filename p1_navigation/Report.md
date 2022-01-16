# Report on Navigation Project

In this project, the Unity Banana Navigation environment was solved using the DQN learning algorithm. Additional tweaks using Double DQN, Prioritized Experience Replay DQN, and Dueling DQN were also added and experimented with.

## Final Algorithm

This approach was able to solve the problem in 250 episodes. (Please note that when re-running this approach to get the model file, it was run in 331 episodes.)

### Hyperparameters

* Maximum steps per episode: 600
* Starting epsilion: 1.0
* Ending epsilion: 0.01
* Epsilion decay rate: 0.98
* Replay buffer size: 1e5
* Batch Size: 64
* Gamma (discount factor): 0.99
* Tau (or soft update of target parameters): 1e-3
* Learning Rate: 5e-4
* Update Frequency (how often to update the network): 4
* Used a double DQN
* Used SmoothL1Loss (note: in code it's labeled as Huber since I believe it's identical with the chosen parameters)

The use of the SmoothL1Loss means that you have a standard MSE for errors below 1 and then above that it's an L1 loss, which helps reduce the effect of outliers.

### Model Architecture

1. The input to the neural network consists of a 37 x 1 vector including the agent's velocities and the ray based perception in the agent's forward direction.
2. The first hidden layer is fully-connected and consists of 64 rectifier units.
3. The second hidden layer is fully-connected and also consists of 64 rectifier units.
4. The output layer is a fully-connected linear layer with a single output for each of the 4 actions.

## Hyperparameters Tried

I tried a few different approaches that were variants to the above chosen algorithm. Each was less optimal.

* No Double DQN and No Huber: Solved in 292 episodes.
* No Huber: 265 episodes
* +Gradient-Clipping and No Huber: Solved in 303 episodes.
* No Double DQN: Solved in 278 episodes
* +Dueling: Solved in 346 episodes
* Higher Tau: Solved in 384 episodes
* Smaller Buffer Size: Not solves in 600 episodes
* Larger first hidden layer (128): 329 episodes

## Components of the algorithm

The algorithm uses trial-and-error to figure out the optimal policy (set of actions) that it can take to maximize its rewards.

### Q Function

To discover this optimal policy, we use a Q function, which calculates the expected reward for all possible actions in all possible states. The optimal policy at each state is the action with the maximal Q value. By following the action with the maximal Q value in each state, the agent can then maximize the total expected reward.

### Epsilon Greedy Algorithm

It may not be ideal to always take the action with the maximal Q value. For instance, early in training the information can be noisy. Our agent might find an action in a state to provide the maximal Q value accidently due to noise, and commit to this action too early in training. In this case, it would be ideal to every so often choose the actions in a state at random allowing for more exploration of the state-action-value space. Over time as we explore the space, we can reduce the amount of exploration. This is exactly what the epsilon-greedy algorithm allows.

This algorithm allows the agent to manage the exploration-exploitation trade-off. The agent "explores" by picking a random action with a uniform probability, epsilon. The agent continues to "exploit" its knowledge of the environment by choosing actions based on the policy with probability one minus epsilon. The epsilon value is decayed over time so that the agent favors exploration early on but over time favors exploitation as it gains more experience.

### Deep Q-Network (DQN)

A deep neural network is used to approximate the Q function (function approximater). Given a network `F`, finding an optimal policy is a matter of finding the best weights `w` such that `F(s,a,w) ≈ Q(s,a)`. Basically, we take our input state and then use the neural network to transform the state to the Q values for each action. The Q-values are then used for deciding on the best action as described with the Epsilon Greedy Algorithm.

### Experience Replay

There are issues that arise if we are only learning online from each successive new experience.

A primary issue arises from assumptions in stochastic gradient descent, which works best with independent and identically distributed samples. But in reinforcement learning, samples are received sequential via interactions from the environment. This causes the network to see too many samples of one kind and forget the others, catastrophic forgetting. For example, the agent might have a different starting or ending location in a game that involves moving in a grid. The agent may forget the optimal moves based on one starting location if they are later different from the optimal moves based on another starting location.

Experience replay solves this problem. It involves storing each new experience in a replay buffer. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically due to correlations in the sequential data. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.


### Prioritized Experience Replay (PER)

The vanilla experience replay can increase learning time since it treats all experience samples the same. In addition, based on the PER paper, they note that standard experience replay often might ignore certain experiences all together due to the uniform sampling over a large set of experiences.

To solve these problems, Schaul et al., set a priority to each experience based on the TD Error and sample experiences based on this priority. Hence experiences with greater TD error will be more likely be sampled, presumably because there is more to learn from these experiences. The priority is updated based on each replay. This means that recent experiences are more likely to be sampled since they tend to have the highest priority (largest TD error) and so an experience is likely to be sampled at least once. Then when the experience is replayed, the updated priority (TD error) is often smaller, making it less likely to be recalled again.

Sampling based on priority can take a longer time than uniform and so in this implementation I used a SumTree that makes access into a log(n) problem. In addition, due to the non-uniform sampling, this requires changes in our update rule, which now includes an importance sampling weight.

NOTE: I wasn't able to get the prioritized experience replay working. So while much of the code is implemented it right now gives an error at some point in training with NaNs in the tree.

### Double DQN

DQNs have an issue where they often overestimate Q-values. The accuracy of the Q-values depends on which actions have been tried and which states have been explored. Early on, the agent will not have gathered enough experiences so the Q-function will end up selecting the maximum value from a noisy set of reward estimates. This will be inaccurate and the agent will get stuck in a non optimal policy over the long run.

We can address this issue using Double Q-Learning, where one neural network is used to select the best action (local network) and another neural network is used to evaluate that action (target network). The local network is the one that you are actively training online while the target network is one that is updated more slowly with weights that are some steps behind the local network. By doing this, it means that if there are a noisy set of reward estimates, then it less likely that extreme Q-values from the local network will shift learning. This is because if the local network produces a high Q-value, then this will be evaluated by the target network. If it's a noisy estimate, then the target network will give a smaller value. If the estimate is accurate, then the target network will likely agree with the local network. This seems to be similar to regularization.

Also a note that while the target network was updated frequently (every 4 steps), the amount of the update was small (1e-3).

### Dueling agents

Dueling networks utilize two streams: one that estimates the state value function V(s), and another that estimates the advantage for each action A(s,a). These two values are then combined to obtain the desired Q-values.

The reasoning behind this approach is that state values don't change  across actions, so we can estimate them directly. However, we still want to measure the impact that individual actions have in each state, hence the need for the advantage function, which is . Estimating the state values separately is especially advantageous if we have many actions and hence we may rarely encounter a particular action. If we encounter some actions, then we will likely have a good estimate of the state value. We can use this state value estimate to help in selecting or updating the Q-value for an action that hasn't been encountered before.


## Future Improvements

* Try more hyper-parameter tuning
* Fix the errors with prioritized experience replay and see how it might benefit performance.
* I made the additional layers for the Dueling DQN simple. The state and advantage layers do not have separate hidden layers beforehand. The shared hidden layers goes directly to the state and advantage layers. In the future, I might try to add a separate hidden layer right before the state and advantage layers.
* Try noisy DQN, which uses noisy linear layers for exploration instead of epsilon-gready exploration. The parameters of the noise are learned with gradient descent along with the remaining network weights. In the Noisy Networks for Exploration paper, they find this approach yields higher scores for Atari games. It would be interested to compare this approach in this Banana problem when compared to epsilon-greedy.
* For the prioritized replay, I have a constant Beta used in the importance sampling. In a future iteration, I could try gradually changing the beta over the course of learning.
