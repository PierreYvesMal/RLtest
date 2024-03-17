import gym
import pygame
import numpy as np
WIDTH = 640
HEIGHT = 480

class Fighter:
    def __init__(self, x) -> None:
        self.x = int(x)
        self.health = int(100)

    @property
    def binx(self):
        return int(self.x/64)

class GameEnv(gym.Env):
    def __init__(self) -> None:
        super(GameEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(6) #punch, crouch, 4 dirs
        self.observation_space = gym.spaces.MultiDiscrete([100+1, 100+1, 10+1, 10+1], dtype=int) #health p1, health p2, p1.x, p2.x, 
        self.q_table = np.zeros((self.observation_space.nvec.prod(), self.action_space.n))
        print("Q-Table shape:", self.q_table.shape)
        self.reset()

    def step(self, action):
        reward = 0
        reward -= 10 # To optimise the number of action
        if action == 2: # Stupid example where a specific action (punch) would simply always be the better option
            reward += 10
            self.player.health -= 3
        if self.player.health <= 0:
            done = True
        else:
            done = False
        info = {}
        return self.get_obs(), reward, done, info
    
    def reset(self):
        self.player = Fighter(WIDTH/4)
        self.bot = Fighter(WIDTH*3/4)
        info = {}
        return self.get_obs(), info
    
    def get_obs(self):
        return np.array([self.player.health, self.bot.health, self.player.binx, self.bot.binx])

    def close(self):
        pygame.quit()

env = GameEnv()
state = env.reset()

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

for episode in range(1000):
    state = env.reset()[0]
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(env.q_table[np.ravel_multi_index(state, env.observation_space.nvec)])  # Exploit learned values

        next_state, reward, done, _ = env.step(action)

        if done: 
            break

        # Update Q-table
        print(next_state)
        best_next_action = np.argmax(env.q_table[np.ravel_multi_index(next_state, env.observation_space.nvec)])
        td_target = reward + gamma * env.q_table[np.ravel_multi_index(next_state, env.observation_space.nvec)][best_next_action]
        td_error = td_target - env.q_table[np.ravel_multi_index(state, env.observation_space.nvec)][action]
        env.q_table[np.ravel_multi_index(state, env.observation_space.nvec)][action] += alpha * td_error

        state = next_state

print(env.q_table.shape)
for i in range(env.q_table.shape[0]):
    if np.sum(env.q_table[i,:]):
        print(env.q_table[i,:])

for i in range(env.q_table.shape[1]):
    print("action", i, ":",np.sum(env.q_table[:,i]))