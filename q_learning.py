import gym
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

'''

Formulating the problem: (See https://github.com/openai/gym/wiki/CartPole-v0)

    Env: Cartpole-v0
    State (conti): Cart position, Cart velocity, Pole angle, Pole velocity at top
    Actions (discrete): Left (-1), Right (+1)
    Reward: +1 for every time step the cart-pole is "alive" (remains upright)

'''

EXPLORATION_RATE_MIN = 0.01
LEARNING_RATE_MIN = 0.001

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def get_env_bounds(env):
    print("\nPrinting env limits...\n")
    print("Min cart posi: {}\tMax cart posi: {}".format(env.observation_space.low[0], env.observation_space.high[0]))
    print("Min cart vel: {}\tMax cart vel: {}".format(env.observation_space.low[1], env.observation_space.high[1]))
    print("Min pole angle: {}\tMax pole angle: {}".format(env.observation_space.low[2], env.observation_space.high[2]))
    print("Min pole vel: {}\tMax pole vel: {}\n".format(env.observation_space.low[3], env.observation_space.high[3]))

def init_Qtab(nActions=2, nBuckets=(1, 5, 10, 5)):
    return np.zeros(nBuckets + (nActions, ))

def into_buckets(observ, nStateVar=4, nBuckets=(1, 5, 10, 5)):
    observ[1] = 5 * sigmoid(observ[1]) 
    observ[3] = 5 * sigmoid(observ[3])
    
    upper_lim = [env.observation_space.high[0], 5, env.observation_space.high[2], 5]
    lower_lim = [env.observation_space.low[0], 0, env.observation_space.low[2], 0]
    buckets = [np.linspace(lower_lim[i], upper_lim[i], nBuckets[i]+1) for i in range(nStateVar)]
    buckets = np.asarray(buckets)

    encodedStates = [j for i in range(nStateVar) for j in range(nBuckets[i]) if (observ[i] > buckets[i][j]) & (observ[i] < buckets[i][j+1])]
    return tuple(encodedStates)

def pull_action(Qtab, curState, expRate):
    if (np.random.random() <= expRate): # explore
        return env.action_space.sample()
    else: # act greedy (exploit)
        return np.argmax(Qtab[curState])

def update_Qtab(Qtab, oldState, newState, action, learningRate, discountFactor, reward):
    Qtab[oldState][action] += learningRate * (reward + (discountFactor * np.max(Qtab[newState])) - Qtab[oldState][action]) 
    return Qtab

def get_learning_rate(learningRate, learningRateDecay=0.98):
    learningRate *= learningRateDecay
    return max(LEARNING_RATE_MIN, learningRate)

def get_exploration_rate(expRate, expDecay=0.99):
    expRate *= expDecay
    return max(EXPLORATION_RATE_MIN, expRate)

if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    nActions = env.action_space.n
    nStateVar = 4
    nEpisodes = 1000
    scores = []

    # print env observ limits
    get_env_bounds(env)

    for epiNum in range(nEpisodes):
        print("\nEpisode number {}:".format(epiNum))

        # init episode  
        discountFactor = 0.9 # gamma    
        learningRate = 1 # alpha  
        expRate = 1 # epsilon  
        tState = False
        Qtab = init_Qtab(nActions)
        curObserv = env.reset()

        t = 0
        while not tState:
            env.render()

            action = pull_action(Qtab, into_buckets(curObserv), expRate)
            newObserv, reward, tState, _ = env.step(action)
            Qtab = update_Qtab(Qtab, into_buckets(curObserv), into_buckets(newObserv), action, learningRate, discountFactor, reward)
            
            curObserv = newObserv
            expRate = get_exploration_rate(expRate)
            learningRate = get_learning_rate(learningRate)
            t += 1

            # print("\tCurrent encoded state: {}".format(into_buckets(curObserv)))
            # print("\tLearning Rate: {}, Exploration rate: {}".format(learningRate, expRate))
        
        print("\t\tReward won: {}".format(t))
        scores.append(t)
    
    scores = np.asarray(scores)
    print("\n---End---\n")
    print("Avg score: {}".format(np.mean(scores)))
    print("Max score: {} at episode number {}".format(np.max(scores), np.argmax(scores)))

    env.close()

    plt.plot(range(nEpisodes), scores)
    plt.title("Agent's Performance")
    plt.xlabel("Episode Number")
    plt.ylabel("Total Reward")
    plt.show()
    

