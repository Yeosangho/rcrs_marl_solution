import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, n_agent, in_channels=3, dim_action=9):

        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.in_channels = in_channels
        #self.dim_action = dim_action * n_agent

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.conv4 = nn.Conv1d(self.n_agent, 16, kernel_size=2)
        self.fc2 = nn.Linear(512+(16*(dim_action-1)), 300)
        self.fc3 = nn.Linear(300, 1)


        #nn.init.xavier_uniform(self.conv1.weight)
        #nn.init.xavier_uniform(self.conv2.weight)
        #nn.init.xavier_uniform(self.conv3.weight)
        #nn.init.xavier_uniform(self.conv4.weight)
        #nn.init.xavier_uniform(self.fc1.weight)
        #nn.init.xavier_uniform(self.fc2.weight)
        #nn.init.xavier_uniform(self.fc3.weight)

      
    def forward(self, state, acts):
        #print(state.size())
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        a = F.relu(self.conv4(acts))
        a = a.view(a.size(0), -1)
        c = th.cat([x, a], 1)
        x = F.relu(self.fc2(c))
        return self.fc3(x)        

class Actor(nn.Module):
    def __init__(self, in_channels=3, dim_action=9):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, dim_action)

        #nn.init.xavier_uniform(self.conv1.weight)
        #nn.init.xavier_uniform(self.conv2.weight)
        #nn.init.xavier_uniform(self.conv3.weight)
        #nn.init.xavier_uniform(self.fc1.weight)
        #nn.init.xavier_uniform(self.fc2.weight)


    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)