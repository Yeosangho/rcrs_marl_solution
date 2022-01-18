import os
import sys

from MADDPG import MADDPG
import numpy as np
import torch
from params import scale_reward
from RCRS_Env import RCRS_Env
import subprocess
from subprocess import Popen, PIPE
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from PIL import Image
import argparse
from copy import deepcopy

def make_ckp(itr, agent_model):
    model_list = []
    opt_list = []

    model_list.append(agent_model.actor.state_dict())
    model_list.append(agent_model.actor_target.state_dict())
    model_list.append(agent_model.critic.state_dict())
    model_list.append(agent_model.critic_target.state_dict())

    opt_list.append(agent_model.actor_optimizer.state_dict())
    opt_list.append(agent_model.critic_optimizer.state_dict())
    ckp = {
        'itr':itr+1,
        'state_dict' : model_list,
        'optimizer':opt_list
    }
    return ckp

def run(rank, size, ex_args):
    gpu_num = 2
    torch.cuda.set_device(int(rank%gpu_num))

    # do not render the scene
    e_render = False

    food_reward = 10.
    poison_reward = -1.
    encounter_reward = 0.01
    n_coop = 2

    reward_record = []

    np.random.seed(rank)
    torch.manual_seed(rank)
    n_agents = 18
    in_channels = 1
    if(ex_args.action_space == 0 ):
        n_actions = 6 
    else :
        n_actions = 6 * 2
    capacity = 1000000
    batch_size = 32

    iteration_num = 10000000
    max_itr_per_ep = 400
    episodes_before_train = 1
    itr_before_train = ex_args.itr_before_train

    win = None
    param = None
    state = torch.Tensor(84,84)
    recent_observations = torch.Tensor(4, 84, 84)
    state_ = torch.zeros(84,84)
    non_state = torch.zeros(84,84)
    rewards = torch.zeros(n_agents)
    env = None
    n_exploration_eps = ex_args.exploration_eps #original 25000
    init_noise_scale = ex_args.init_noise_scale #original 0.3
    final_noise_scale = 0.0 

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=size)
    FNULL = open(os.devnull, 'w')
    if(rank ==0):
        #subprocess.call(["bash", "./start.sh", "--map", "../suwon_mod/map", "-c", "../suwon_mod/config"])
        child_proc = subprocess.Popen(["bash", "./start.sh", "--map", "../suwon_mod/map", "-c", "../suwon_mod/config", "-l", "logs"],  stdout=FNULL, stderr=FNULL)
        time.sleep(6)  

        env = RCRS_Env(ex_args)
        state = env.getState()
        #print(state.size())
    dist.broadcast(state, 0)
    maddpg = MADDPG(18, in_channels, n_actions, batch_size, capacity,
                    episodes_before_train, itr_before_train, rank, size, ex_args)
    
    explr_pct_remaining = max(0, n_exploration_eps - maddpg.episode_done) / n_exploration_eps
    maddpg.scale_noise(final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
    maddpg.reset_noise()  
    if(rank == 0):
        beta_remaining =  min(ex_args.beta_eps, maddpg.episode_done) / ex_args.beta_eps
        beta_remaining = (1 - beta_remaining)**2.2
        env.beta = ex_args.final_beta - (ex_args.final_beta-ex_args.init_beta) * beta_remaining

    FloatTensor = torch.cuda.FloatTensor 
    c_loss_ep = []
    a_loss_ep = []
    for i in range(iteration_num):
        itrStart = time.time()
        total_reward = 0.0
        rr = np.zeros((n_agents,)) 

        #if(i < 400 and rank == 0):
        #    #print(state.size())
        #    state1 = state.cpu()
        #    state1 = state.numpy().astype(np.uint8)
###
        #    image = Image.fromarray(state1, "L")
        #    image.save(str(i)+ ".png")        

        if((i%max_itr_per_ep >= (max_itr_per_ep-1)) and (i > 0)):
            if(rank ==0):
                c_loss_mean = 0
                a_loss_mean = 0
                if(i > itr_before_train):
                    c_loss_mean = np.array(c_loss_ep).mean()
                    a_loss_mean = np.array(a_loss_ep).mean()
                env.reset(c_loss_mean, a_loss_mean) 
                state = env.getState()
                beta_remaining =  min(ex_args.beta_eps, maddpg.episode_done) / ex_args.beta_eps
                beta_remaining = (1 - beta_remaining)**2.2
                env.beta = ex_args.final_beta - (ex_args.final_beta-ex_args.init_beta) * beta_remaining                
            c_loss_ep = []
            a_loss_ep = []
            maddpg.episode_done += 1
            explr_pct_remaining = max(0, n_exploration_eps - maddpg.episode_done) / n_exploration_eps
            maddpg.scale_noise(final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
            maddpg.reset_noise()  

            dist.broadcast(state, 0)
            if(int((i)/400) % 100 == 99): 
                ckp = make_ckp(i, maddpg)
                torch.save(ckp, f"/scratch/x2026a02/new_rcrs/rcrs-server/boot/{ex_args.file_prefix}_{i}_{rank}.pth")
    
    
        else :
            #action =  random.randint(0, len(env.actionspace)-1)
            if(rank == 0):
                last_idx = maddpg.memory.store_frame(state.numpy())
                #state = state.type(FloatTensor)
                recent_observations_list = maddpg.memory.encode_recent_observation()
                recent_observations = torch.Tensor(recent_observations_list)
            dist.broadcast(recent_observations, 0)
            recent_observations = recent_observations.type(torch.cuda.FloatTensor)
            recent_observations = torch.where(recent_observations==(-(rank+1)), torch.tensor(255.0).cuda(), recent_observations)
            recent_observations = torch.where(recent_observations<0, torch.tensor(20.0).cuda(), recent_observations)
            
            recent_observations = recent_observations.type(torch.cuda.FloatTensor) / 255.0
            
            action = maddpg.select_action(recent_observations.cuda(), rank).data.cpu()
            if(rank ==0):
                action_env = deepcopy(action).detach()
                if(ex_args.out_activation == 0):
                    
                    if(ex_args.action_space  == 0):
                        action_env = torch.clamp(action_env, -1.0, 1.0)
                    else :
                        feature_dim = int(n_actions /2)
                        action_env[:, :feature_dim] = torch.clamp(action_env[:, :feature_dim], -1.0, 1.0)
                        action_env[:, feature_dim:] = torch.clamp(action_env[:, feature_dim:], 0.0, 1.0)
                else :    
                    if(ex_args.action_space  == 0):
                        action_env = torch.tanh(action_env)
                    else :
                        #act = torch.sigmoid(act)
                        action_env = (torch.tanh(action_env)+1.)/2.
                #feature_dim = int(self.n_actions /2)
                #act[:feature_dim] = torch.tanh(act[:feature_dim])
                #act[feature_dim:] = torch.sigmoid(act[feature_dim:])

                state_, rewards = env.step(action_env) 
                #print(state_.size())

                rewards = torch.FloatTensor(rewards).type(FloatTensor).cpu()

                #print(rewards.size())
            dist.broadcast(rewards, 0)
            dist.broadcast(state_, 0)
    

            if i != (max_itr_per_ep - 2):
                next_state = state_
            else:
                next_state = non_state
    

            if(rank ==0):
                rewards = rewards.tolist()
                maddpg.memory.store_effect(last_idx, action.tolist(), rewards, 0)
            state = next_state
            
            #updateStart =time.time()
            c_loss, a_loss = maddpg.update_policy()
            if c_loss is not None :
                c_loss_ep.append(c_loss)
                a_loss_ep.append(a_loss)
                
            #updateEnd = time.time()
            #print(f'update time {updateEnd-updateStart}')
            maddpg.itr += 1
            itrEnd = time.time()
            #print(f'itr time {itrEnd-itrStart}')

    if(rank == 0):
        env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_space", type=int, help='0 is weight vector, 1 is feature + weight vector')
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--aux_reward", type=float, help="weight for aux reward")
    parser.add_argument("--scale_reward", type=float, help="weight for aux reward")
    parser.add_argument("--file_prefix", type=str)
    parser.add_argument("--itr_before_train", type=int)
    parser.add_argument("--exploration_eps", default=500, type=int)
    parser.add_argument("--init_noise_scale",default=0., type=float)
    parser.add_argument("--lr_scale",default=1.0, type=float)
    parser.add_argument("--out_activation",default=0.0, type=float)
    parser.add_argument("--init_beta",default=0.0001, type=float)
    parser.add_argument("--final_beta",default=0.01, type=float)
    parser.add_argument("--beta_eps", default=1000, type=float)





    ex_args = parser.parse_args()
    n_agents = 18
    size = n_agents    
    #Requester is defined as "centralized actor-critic"
    mp.set_start_method("spawn")
    processes = []
    for rank in range(n_agents):
        p = mp.Process(target=run, args=(rank, size, ex_args))    
        processes.append(p)    

    for p in processes:
        p.start()

    for p in processes:
        p.join()

