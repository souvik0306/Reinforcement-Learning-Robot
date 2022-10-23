import gym
import pybullet
import pybullet_envs
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("AntBulletEnv-v0")

# env.render(mode='human')

MAX_AVG_SCORE = 1500

policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,net_arch=[512,512])

model = PPO('MlpPolicy', env, learning_rate=0.003, policy_kwargs=policy_kwargs, verbose=1)

for i in range(8000):
  print("Traning iteration ", i)
  model.learn(total_timesteps=10000)
  model.save("ppo_Ant_saved_model_5")
  mean_reward, std_reward = evaluate_policy(model,model.get_env(),n_eval_episodes=15)
  print("mean reward ", mean_reward)
  if mean_reward >= MAX_AVG_SCORE:
    break

