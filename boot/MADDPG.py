from model import Critic, Actor
import torch
from copy import deepcopy
from memory import ReplayMemory, Experience

from utils.replay_buffer import ReplayBuffer

from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
import time
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from OUNoise import OUNoise

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, in_channels, dim_act, batch_size,
                 capacity, episodes_before_train, itr_before_train, rank, size, ex_args):
        self.ex_args = ex_args
        self.action_space = self.ex_args.action_space
        self.scale_reward = self.ex_args.scale_reward
        self.out_activation = self.ex_args.out_activation

        self.frame_history_len = 4         
        self.actor = Actor(in_channels*self.frame_history_len, dim_act) 
        self.critic = Critic(n_agents, in_channels*self.frame_history_len, dim_act) 
        #self.critic = Critic(n_agents, in_channels, dim_act )

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.n_agents = n_agents
        self.n_states = in_channels
        self.n_actions = dim_act
        self.memory = None
        if(rank ==0):
            self.memory = ReplayBuffer(capacity, self.frame_history_len)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        #self.critic_optimizer = Adam(self.critic.parameters(), lr=0.001) 
        self.critic_optimizer = Adam(self.critic.parameters(), lr=0.001 * ex_args.lr_scale ) 
        self.actor_optimizer = Adam(self.actor.parameters(), lr=0.0001* ex_args.lr_scale) 
        self.itr = 0
        self.itr_before_train = itr_before_train

        self.rank = rank
        self.size = size
        self.act_list = [torch.zeros(1, dim_act) for i in range(n_agents)]
        self.act_batch_list = [torch.zeros(self.batch_size, 1, dim_act) for i in range(n_agents)]

        self.state_batch = torch.zeros(self.batch_size, in_channels* self.frame_history_len, 84, 84).cpu()
        self.next_state_batch =  torch.zeros(self.batch_size, in_channels *  self.frame_history_len, 84, 84).cpu()
        self.action_batch = torch.zeros(self.batch_size, self.n_agents, self.n_actions).cpu()
        self.reward_batch = torch.zeros(self.batch_size, self.n_agents).cpu()

        self.exploration = OUNoise(self.n_actions)

        if self.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()
        self.steps_done = 0
        self.episode_done = 0


    def update_policy(self):
        # do not train until exploration is enough
        #if self.episode_done <= self.episodes_before_train:
        #    return None, None

        if self.itr <= self.itr_before_train :
            return None, None

        ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        wholeUpdateStart = time.time()
        processtime = 0
        dataloadtime = 0
        #for agent in range(self.n_agents):
        dataloadstart = time.time()
        state_batch_list = []
        next_state_batch_list = []
        action_batch_list = []
        reward_batch_list = []
     
        if(self.rank == 0):
            #print(f'#####################{self.rank}###################')
            for i in range(self.n_agents):
                state_batch, act_batch, rew_batch, next_state_batch, done_mask = self.memory.sample(self.batch_size)
                
                next_states = torch.from_numpy(next_state_batch).type(torch.FloatTensor)
                states = torch.from_numpy(state_batch).type(torch.FloatTensor) 
                actions = torch.from_numpy(act_batch)
                rewards = torch.from_numpy(rew_batch)
                #print(f'next_states[0] {torch.sum(next_states[0])} next_states[16] {torch.sum(next_states[16])}')
                #print(f'states[0] {torch.sum(states[0])} states[16] {torch.sum(states[16])}')
                #print(f'actions[0] {torch.sum(actions[0])} actions[16] {torch.sum(actions[16])}')
                #print(f'rewards[0] {torch.sum(rewards[0])} rewards[16] {torch.sum(rewards[16])}')

                next_state_batch_list.append(next_states)
                state_batch_list.append(states)
                action_batch_list.append(actions)
                reward_batch_list.append(rewards)
                # : (batch_size_non_final) x n_agents x dim_obs
            dist.scatter(self.state_batch, scatter_list=state_batch_list, src=0)
            dist.scatter(self.action_batch, scatter_list=action_batch_list, src=0)
            dist.scatter(self.reward_batch, scatter_list=reward_batch_list, src=0)
            dist.scatter(self.next_state_batch, scatter_list=next_state_batch_list, src=0)
        else:            
            dist.scatter(self.state_batch, src=0)
            dist.scatter(self.action_batch, src=0)
            dist.scatter(self.reward_batch, src=0)
            dist.scatter(self.next_state_batch, src=0)

        state_batch = self.state_batch.cuda()
        state_batch = torch.where(state_batch==(-(self.rank+1)), torch.tensor(255.0).cuda(), state_batch)
        state_batch = torch.where(state_batch<0, torch.tensor(20.0).cuda(), state_batch)    

        next_state_batch = self.next_state_batch.cuda()
        next_state_batch = torch.where(next_state_batch==(-(self.rank+1)), torch.tensor(255.0).cuda(), next_state_batch)
        next_state_batch = torch.where(next_state_batch<0, torch.tensor(20.0).cuda(), next_state_batch)    

        state_batch = self.state_batch.type(torch.cuda.FloatTensor) / 255.0
        action_batch = self.action_batch.type(torch.cuda.FloatTensor) 
        reward_batch = self.reward_batch.type(torch.cuda.FloatTensor)
        next_state_batch = self.next_state_batch.type(torch.cuda.FloatTensor) / 255.0

    
        #non_final_mask = ByteTensor(list(map(lambda s:torch.count_nonzero(s).item() >0,
        #                                     next_state_batch)))
        #non_final_next_states = torch.stack(
        #    [s for s in next_state_batch
        #     if torch.count_nonzero(s).item() > 0]).type(FloatTensor)                                                 

        dataloadend = time.time()
        dataloadtime += dataloadend - dataloadstart

        processstart = time.time()


        # for current agent
        whole_state = state_batch
        whole_action = action_batch
        self.critic_optimizer.zero_grad()
        current_Q = self.critic(whole_state, whole_action)

        non_final_next_action = self.actor_target(next_state_batch).unsqueeze(1).cpu() 
        #print(non_final_next_action.size())
        dist.all_gather(self.act_batch_list, non_final_next_action)
        non_final_next_actions = torch.cat(self.act_batch_list, dim=1).cuda()
        #non_final_next_actions = th.stack(non_final_next_actions)
        #non_final_next_actions = (
        #    non_final_next_actions.transpose(0,
        #                                     1).contiguous())

        target_Q = torch.zeros(
            self.batch_size).type(FloatTensor)
        target_Q = self.critic_target(
            next_state_batch,
            non_final_next_actions
        ).squeeze()
        # scale_reward: to scale reward in Q functions

        target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
            reward_batch[:, self.rank].unsqueeze(1) * self.scale_reward)
        #print(reward_batch[:, self.rank].unsqueeze(1) * self.scale_reward)
        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        #print(loss_Q)
        loss_Q.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        obs_i = state_batch
        action_i = self.actor(obs_i)
        #if(self.out_activation == 0):
        #    if(self.action_space  == 0):
        #        action_i = torch.clamp(action_i, -1.0, 1.0)
        #    else :
        #        feature_dim = int(self.n_actions /2)
        #        action1 = action_i[:,:feature_dim].clone()
        #        action2 = action_i[:,feature_dim:].clone()
        #        action1 = torch.clamp(action1, -1.0, 1.0)
        #        action2 = torch.clamp(action2, 0.0, 1.0)
        #        action_i = torch.cat([action1, action2], dim=1)            
        #else :
        #    if(self.action_space  == 0):
        #        action_i = torch.tanh(action_i)
        #    else :
        #        action_i = (torch.tanh(action_i)+1.)/2.
        #        #feature_dim = int(self.n_actions /2)
        #        #action1 = action_i[:,:feature_dim].clone()
        #        #action2 = action_i[:,feature_dim:].clone()
        #        #action1 = torch.tanh(action1)
        #        #action2 = torch.sigmoid(action2)
        #        #action_i = torch.cat([action1, action2], dim=1)

        
        ac = self.action_batch.clone()

        ac[:, self.rank, :] = action_i
        whole_action = ac.view(self.batch_size,self.n_agents, self.n_actions).cuda()


        actor_loss = -self.critic(whole_state, whole_action)
        actor_loss = actor_loss.mean()
        actor_loss += (action_i**2).mean() * 1e-3
        #actor_loss = torch.abs(actor_loss)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        c_loss.append(loss_Q.item())
        a_loss.append(actor_loss.item())


        processend = time.time()
        processtime += processend - processstart

        wholeUpdateEnd = time.time()
        #print(f'datalodtime {dataloadtime}')
        #print(f'processtime {processtime}')
        #print(f'wholeUpdateTime {wholeUpdateEnd-wholeUpdateStart}')
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)


        return c_loss, a_loss

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def select_action(self, state_batch, rank):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor
        sb = state_batch.detach()
        act = self.actor(sb.unsqueeze(0))


        #act += torch.from_numpy( np.random.randn(self.n_actions) * self.var[rank]).type(FloatTensor)
        
        if self.itr > self.itr_before_train :
            act += Variable(torch.Tensor(self.exploration.noise()).type(FloatTensor), requires_grad=False)
        #if(self.out_activation == 0):
        #    if(self.action_space  == 0):
        #        act = torch.clamp(act, -1.0, 1.0)
        #    else :
        #        feature_dim = int(self.n_actions /2)
        #        act[:feature_dim] = torch.clamp(act[:feature_dim], -1.0, 1.0)
        #        act[feature_dim:] = torch.clamp(act[feature_dim:], 0.0, 1.0)
        #else :    
        #    if(self.action_space  == 0):
        #        act = torch.tanh(act)
        #    else :
        #        #act = torch.sigmoid(act)
        #        act = (torch.tanh(act)+1.)/2.
        #        #feature_dim = int(self.n_actions /2)
        #        #act[:feature_dim] = torch.tanh(act[:feature_dim])
        #        #act[feature_dim:] = torch.sigmoid(act[feature_dim:])


        dist.all_gather(self.act_list ,act.cpu())
        acts = torch.cat(self.act_list, dim=0)
        #print(acts)
        #print(act.size())
        #print(acts.size())
        self.steps_done += 1

        return acts
