import torch
import torch.nn as nn

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from sympy.physics.control.control_plots import plt
from torch.distributions import Categorical
from torchvision import transforms
import gym
import numpy as np
import os


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)
env.seed(42)
env.action_space.seed(42)
torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

stat = env.reset()

print('Complex: ', JoypadSpace(env, COMPLEX_MOVEMENT).action_space)
print("Complex movement: ", COMPLEX_MOVEMENT)
print('\nSimple: ', JoypadSpace(env, SIMPLE_MOVEMENT).action_space)
print("Simple movement: ", SIMPLE_MOVEMENT)

print('State space: ', env.observation_space)
print('Initial state: ', stat)
print('\nAction space: ', env.action_space)
print('Random action: ', env.action_space.sample())


class MarioSolver:
    def __init__(self, learning_rate):
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim=-1)
        ).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-4)
        self.reset()

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.episode_actions = torch.tensor([], requires_grad=True).cuda()
        self.episode_rewards = []

    def save_checkpoint(self, directory, episode):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, 'checkpoint_{}.pth'.format(episode))
        torch.save(self.model.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def load_checkpoint(self, directory, filename):
        self.model.load_state_dict(torch.load(os.path.join(directory, filename)))
        print('Resuming training from checkpoint \'{}\'.'.format(filename))
        return int(filename[11:-4])

    def backward(self):
        future_reward = 0
        rewards = []
        # Calculate future rewards for each step in the episode
        for r in self.episode_rewards[::-1]:
            future_reward = r + gamma * future_reward
            rewards.append(future_reward)
        # Reverse the rewards list and convert it to a PyTorch tensor
        rewards = torch.tensor(rewards[::-1], dtype=torch.float32).cuda()

        # Standardize the rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate the policy gradient loss
        loss = torch.sum(torch.mul(self.episode_actions, rewards).mul(-1)) # the gradient of the expected reward is equal to the expectation of the log probability of the current policy multiplied by the reward.

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset episode-specific variables
        self.reset()


batch_size = 10
gamma = 0.95
load_filename = 'checkpoint_18200.pth'
save_directory = "./mario_pg"
batch_rewards = []
episode = 0
checkpoint_filename = f'checkpoint_{episode}.pth'
checkpoint_path = os.path.join(save_directory, checkpoint_filename)

model = MarioSolver(learning_rate=0.00025)
if load_filename is not None and os.path.exists(os.path.join(save_directory, load_filename)):
    episode = model.load_checkpoint(save_directory, load_filename)
else:
    # The file does not exist, so you may want to create it or take appropriate action
    # For example, you can save the initial model state as the checkpoint
    model.save_checkpoint(save_directory, episode)

all_episode_rewards = []
all_mean_rewards = []
while True:
    observation = env.reset()
    done = False
    while not done:
        env.render()
        observation = torch.tensor(observation.__array__()).cuda().unsqueeze(0)
        # print("observation: ", observation)
        distribution = Categorical(model.forward(observation))
        action = distribution.sample()
        observation, reward, done, _ = env.step(action.item())
        model.episode_actions = torch.cat([model.episode_actions, distribution.log_prob(action).reshape(1)])
        model.episode_rewards.append(reward)
        if done:
            print(episode)
            all_episode_rewards.append(np.sum(model.episode_rewards))
            batch_rewards.append(np.sum(model.episode_rewards))
            model.backward()
            episode += 1
            if episode % batch_size == 0:
                print('Batch: {}, average reward: {}'.format(episode // batch_size,
                                                             np.array(batch_rewards).mean()))
                batch_rewards = []
                all_mean_rewards.append(np.mean(all_episode_rewards[-batch_size:]))
                plt.plot(all_mean_rewards)
                plt.savefig("{}/mean_reward_{}.png".format(save_directory, episode))
                plt.clf()
            if episode % 50 == 0 and save_directory is not None:
                model.save_checkpoint(save_directory, episode)
