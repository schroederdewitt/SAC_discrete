

import gym
#import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import SAC

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--log_beta", type=float, default=None, help="log-beta")
    parser.add_argument("--autotune", type=int, default=1, help="whether to use log_beta")
    parser.add_argument("--save_policy_every_n_steps", type=int, default=1, nargs="?", const=True,
        help="save policy model")
    
    args = parser.parse_args()
    return args

import os

class PolicySaver():

    def __init__(self, path):
        self.path = path
        self._last_filepath = None
        self._last_r = None

    def save(self, network, r, filename):
        filepath = os.path.join(self.path, filename)
        save = False
        if self._last_r is not None:
            if self._last_r <= self._last_r:
                save = True
        else:
            save = True
        self._last_r = r
        if save:
            torch.save(network.state_dict(), filepath)
        if self._last_filepath is not None:
            os.remove(self._last_filepath)
        self._last_filepath = filepath

def train(config):

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make(config.env)
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    path = "./store"
    if not os.path.exists(path):
        os.makedirs(path)
    policy_saver = PolicySaver(path)

    old_global_step_save = 0
    last_return = None
    with wandb.init(project="SAC_Discrete", name=config.run_name, config=config):
        
        agent = SAC(state_size=env.observation_space.shape[0],
                         action_size=env.action_space.n,
                         device=device,
                         log_beta=config.log_beta if not config.autotune else None)

        wandb.watch(agent, log="gradients", log_freq=10)

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        
        collect_random(env=env, dataset=buffer, num_samples=10000)
        
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        for i in range(1, config.episodes+1):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

            average10.append(rewards)
            current_return = rewards
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Policy Loss: {} | Steps: {} | alpha: {}".format(i, rewards, policy_loss, steps, current_alpha.item()))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Bellmann error 1": bellmann_error1,
                       "Bellmann error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Steps": steps,
                       "Episode": i,
                       "Buffer size": buffer.__len__()})

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="SAC_discrete", model=agent.actor_local, wandb=wandb, ep=0)

            global_step = i
            if (global_step - old_global_step_save) >= config.save_policy_every_n_steps:
                if last_return is None or current_return > last_return:
                    logb = config.log_beta if not config.autotune else "AUTO"
                    print("Saving model...")
                    policy_saver.save(agent.actor_local,
                                      current_return,
                                      "trace-{}-logbeta{}-step{}-perf{}-actor.pth".format(config.env,
                                                                                         logb,
                                                                                         global_step,
                                                                                         current_return))

                    last_return = current_return
                old_global_step_save = global_step

if __name__ == "__main__":
    config = get_config()
    train(config)
