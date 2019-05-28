# Stable-oid

Implemented different RL algorithms to solve the infamous [CartPole](https://github.com/openai/gym/wiki/CartPole-v0) problem.

## Algorithms:
- #### Q-Learning:
     "Bucket-ised" the continous state space to construct a lookup table, a Q-table, which is used 
     to perform updates as governed by the Bellman Optimality Equation. Check out `q_learning_results.txt` and 
     the **q_learning_plots** folder for the write-up (on the complete training process) and plots, for consecutive runs, respectively.  
     
     Insights from 
     [Ferdinand](https://ferdinand-muetsch.de/cartpole-with-qlearning-first-experiences-with-openai-gym.html) 
     and [Matthew](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947). 
     
- #### Deep Q-Networks (DQN):
     (*Coming soon*)
     
## To Do:
- Tuning and update `q_learning_results.txt`
- Implement DQN (with experience replay?)
