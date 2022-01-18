from socket import *
from select import select
import sys
import json
import random
from rcrsJsonStateStringParser import RcrsJsonStateStringParser
import csv
from imagifier import Imagifier
import numpy as np
import ast
import numpy as np
import copy
from PIL import Image
from imagifier import Imagifier
import subprocess
from subprocess import Popen, PIPE
import time
import torch
import traceback
import torch.nn.functional as F
import argparse

HOST = 'localhost'
PORT = 9999
BUFSIZE = 2048
class RCRS_Env:
    def __init__(self, ex_args):
        self.clientSocket = socket(AF_INET, SOCK_STREAM)
        self.parser = RcrsJsonStateStringParser()
        self.actionspace = []
        self.previousSqrtBuildingDamage = 1.0
        self.previousBuildingDamage = 1.0
        self.previousDiffSqrtBuildingDamage = 0.0
        self.previousDiffBuildingDamage = 0.0
        self.previousSqrtReward = 0.0
        self.previousReward = 0.0
        self.states = []
        #현재 시뮬레이션의 모든 정보를 기록하는 개체. 
        #self.world = dict()
        self.states_for_pca = []
        self.zeroToOneList = []
        self.ex_args = ex_args
        self.action_space = self.ex_args.action_space
        self.alpha = ex_args.alpha
        self.beta = ex_args.beta
        self.init_beta = ex_args.init_beta
        self.gamma = ex_args.aux_reward
        self.rewardPerAgent = {} 
        self.buildingdamage = ""
        self.imageTranslator = Imagifier(84, 84, 3, '/scratch/x2026a02/new_rcrs/rcrs-server/boot/states_sample_policy_short_feature.json')

        self.burningBuildingDict = {}
        self.jsonStringList = []
        self.n_agents = 18
        self.agent_score = torch.zeros(self.n_agents)
        self.agent_score_sum = 0
        self.global_score = 0
        self.currentagentNormFeature = None
        try :
            self.clientSocket.connect((HOST, PORT))
            self.jsonStringList = copy.deepcopy(self.recvMultiJsonStr())

            print("object number : " + str(len(self.jsonStringList))+ ".")
            #print("building")
            
            self.parser.addBuildings(self.jsonStringList[0], self.jsonStringList[2], "building")
            self.parser.addFirebrigades(self.jsonStringList[1], "firebridage")
            for eId in self.parser.entityDictPerType['firebridage']:
                self.rewardPerAgent[eId] = 0


            self.actionspace = self.parser.entityDictPerType["building"]
            print(len(self.parser.entityDictPerType["building"]))
            print(len(self.parser.entityDictPerType["firebridage"]))
        except Exception as e :
            print("############")
            traceback.print_exc()
            print(e)
    def reset(self, c_loss=0, a_loss=0):
        #state = self.minMaxScaler.transform(self.parser.getCumulativeChangeSetVectorList())
        try : 
            action = [-1] * 18
            action = json.dumps(action)
            self.sendJson(action.encode())            

            jsonStringList = self.recvMultiJsonStr()
            print(jsonStringList)
            print("object number : " + str(len(jsonStringList))+ ".")
            self.parser.reset()  
            self.parser.addBuildings(self.jsonStringList[0], self.jsonStringList[2], "building")
            self.parser.addFirebrigades(self.jsonStringList[1], "firebridage")
            self.buildingdamages  = json.loads(jsonStringList[0])
            self.buildingdamages.append(self.agent_score_sum)
            self.buildingdamages.append(self.global_score)
            self.buildingdamages.append(c_loss)
            self.buildingdamages.append(a_loss)
            print(len(self.parser.entityDictPerType["building"]))
            print(len(self.parser.entityDictPerType["firebridage"]))

            with open(f'/scratch/x2026a02/new_rcrs/rcrs-server/boot/{self.ex_args.file_prefix}_scores_itr_before_train_{self.ex_args.itr_before_train}_scale_{self.ex_args.scale_reward}_alpha_{self.alpha}_beta_{self.init_beta}_gamma_{self.gamma}_a_space_{self.action_space}_explore_eps_{self.ex_args.exploration_eps}_noise_scale_{self.ex_args.init_noise_scale}_lr_scale_{self.ex_args.lr_scale}_out_active_{self.ex_args.out_activation}.csv','a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.buildingdamages)                
            for eId in self.parser.entityDictPerType['firebridage']:
                self.rewardPerAgent[eId] = 0
            self.agent_score_sum = 0
            self.global_score = 0
        except Exception as e :
            print("############")
            traceback.print_exc()
            print(e)              
        #return self.stateBuilder.kPcaImagify(state, 83)

    def recvJsonStr(self):
        recv = ''
        while True :
            data = self.clientSocket.recv(BUFSIZE)
            data_str = data.decode()
            #print(data_str)
            recv += data_str
            if "$" in data_str :
                break
        recv = recv.split("$")[0]        
        return recv



    def recvMultiJsonStr(self):
        recv = ''
        while True :
            data = self.clientSocket.recv(BUFSIZE)
            data_str = data.decode()
            recv += data_str
            if "@" in data_str :
                break
        recv = recv.split("@")[0]
        recvList = recv.split("#")
        resultList = []
        for i in range(0, len(recvList)):
            resultList.append(recvList[i])


        return resultList

    def sendJson(self, action):
        #print('sendJson')
        length = len(action)
        # 데이터 사이즈를 little 엔디언 형식으로 byte로 변환한 다음 전송한다.
        self.clientSocket.sendall(length.to_bytes(4, byteorder='little'))
        # 데이터를 클라이언트로 전송한다.
        self.clientSocket.sendall(action)

    def addRandomAgentoOnVectorList(self, vectorDict) :
        vectorList = []
        agentNum = len(self.parser.entityDictPerType["firebridage"])
        idx = random.randint(0, agentNum-1)
        agentId = self.parser.entityDictPerType["firebridage"][idx]
        for eId in vectorDict.keys() :
            if(eId == agentId):
                vectorDict[eId][8] = 2
            vectorList.append(vectorDict[eId])
        #print(vectorDict)
        #print(vectorList)    
        return vectorList



    def step(self, weightvector):
        #print(action)
        #action = [{"building" : self.actionspace[action]}] 
        #
        stateandscoreinfo = ""
        #distance_error_reward = torch.zeros(18).cuda()
        distance_error_reward = []
        try:
            burningBuildingCnt = len(self.burningBuildingDict.values())
            if(burningBuildingCnt > 0 and self.action_space == 0):
                action = []
                action_per_agent = None
                for i in range(18):
                    buildingFeatures = self.imageTranslator.normalizeFeatures(list(self.burningBuildingDict.values()), 0)
#
                    buildingFeatures[:,:1] = torch.abs( self.currentagentNormFeature[i,:1] - buildingFeatures[:,:1])


                    buildingFeatures = torch.unsqueeze(buildingFeatures, 1)
                    weightvector_per_agent = weightvector[i, :].cuda()

                    buildingValues = torch.tensordot(weightvector_per_agent.unsqueeze(1), buildingFeatures, dims=((0,1),(2,1)))
                    
                    buildingFeatures = buildingFeatures.squeeze(1).sum(dim=1)
                    #distance_error_reward.append(buildingValues)
                    minbuildingIdx = torch.argmin(buildingValues, dim=0)
                    distance_error_reward.append(0)
                    burningBuildingKeys = torch.Tensor(list(self.burningBuildingDict.keys())).cuda()
                    #action_per_agent = torch.zeros(18).cuda().long()
#
                    #action_per_agent = burningBuildingKeys[maxbuildingIdx[:]]
                    #action = action_per_agent.detach().cpu().tolist()
                    action_per_agent = burningBuildingKeys[minbuildingIdx]
                    #print(action_per_agent.cpu().item())
                    action.append(int(action_per_agent.cpu().item()))
     
            elif(burningBuildingCnt > 0 and self.action_space == 1):
                action = []
                action_per_agent = None
                for i in range(18):
                    buildingFeatures = self.imageTranslator.normalizeFeatures(list(self.burningBuildingDict.values()), 0)
#
                    #buildingFeatures[:,:1] = torch.abs( self.currentagentNormFeature[i,:1] - buildingFeatures[:,:1])
                    #buildingPredicted = F.softmax( weightvector[i, 9:].cuda())
                    buildingFeatures = torch.abs(buildingFeatures - weightvector[i, 6:].cuda())
                    #buildingFeatures = torch.abs(buildingFeatures - weightvector[i, :].cuda())

                    buildingFeatures = torch.unsqueeze(buildingFeatures, 1)
                    weightvector_per_agent = weightvector[i, :6].cuda()
                    #weightvector_per_agent = weightvector[i, :].cuda()

                    #weightvector_per_agent -= weightvector_per_agent.min()
                    #weightvector_per_agent /= weightvector_per_agent.max() 
                    buildingValues = torch.tensordot(weightvector_per_agent.unsqueeze(1), buildingFeatures, dims=((0,1),(2,1)))
                    
                    buildingFeatures = buildingFeatures.squeeze(1).sum(dim=1)
                    #distance_error_reward.append(buildingValues)
                    minbuildingIdx = torch.argmin(buildingValues, dim=0)
                    distance_error_reward.append(buildingFeatures[minbuildingIdx].item())
                    burningBuildingKeys = torch.Tensor(list(self.burningBuildingDict.keys())).cuda()
                    #action_per_agent = torch.zeros(18).cuda().long()
#
                    #action_per_agent = burningBuildingKeys[maxbuildingIdx[:]]
                    #action = action_per_agent.detach().cpu().tolist()
                    action_per_agent = burningBuildingKeys[minbuildingIdx]
                    #print(action_per_agent.cpu().item())
                    action.append(int(action_per_agent.cpu().item()))                
            elif(burningBuildingCnt > 0 and self.action_space == 2):
                action = []
                action_per_agent = None
                for i in range(18):
                    buildingFeatures = self.imageTranslator.normalizeFeatures(list(self.burningBuildingDict.values()), 0)
#
                    buildingFeatures = torch.abs(buildingFeatures - weightvector[i, :].cuda())

                    buildingFeatures = torch.unsqueeze(buildingFeatures, 1)

                    buildingFeatures = buildingFeatures.squeeze(1).sum(dim=1)
                    minbuildingIdx = torch.argmin(buildingFeatures, dim=0)
                    distance_error_reward.append(buildingFeatures[minbuildingIdx].item())
                    burningBuildingKeys = torch.Tensor(list(self.burningBuildingDict.keys())).cuda()

                    action_per_agent = burningBuildingKeys[minbuildingIdx]
                    action.append(int(action_per_agent.cpu().item()))                    
            else:
                action = [-1] * 18
                distance_error_reward = [0] * 18

            #print(action)
            action = json.dumps(action)
            self.sendJson(action.encode())
            #dict = self.recvJson()
            stateandscoreinfo = self.recvJsonStr()
            stateandscorelist = stateandscoreinfo.split("@")
            #print(stateandscorelist)

            changeSetStr = stateandscorelist[0]

            #make reward set 
            scoreMapStr = stateandscorelist[1]
            #dispatch set
            dispatchMapStr = stateandscorelist[2]
            #print(f"scoreMapStr {scoreMapStr}")
            #print(f"dispatchMapStr {dispatchMapStr}")
            scoreMap = json.loads(scoreMapStr)

            globalReward = scoreMap['rewards']['-1']
            #print(globalReward)
            #print(distance_error_reward)
            #print(scoreMap['rewards'])
            i =0
            #with open("reward.csv", "a+") as f:
            #    f.write(f"{globalReward}\n")             
            for key in self.rewardPerAgent.keys() :
                if(str(key) in scoreMap['rewards']):
                    self.rewardPerAgent[key] = self.alpha * scoreMap['rewards'][str(key)] + self.beta * globalReward -self.gamma *distance_error_reward[i] 
                else :
                    self.rewardPerAgent[key] = self.beta * globalReward -self.gamma *distance_error_reward[i] 
                i += 1

            agentList, buildingList, self.burningBuildingDict, agentDict = self.parser.cumulativeParse(changeSetStr, dispatchMapStr)
            #state = self.addRandomAgentoOnVectorList(copy.deepcopy(vectorDict))
            #state = state.float() / 255
            #state = state.repeat(18, 1,1)
            agentDict = sorted(agentDict.items())
            agentDict = dict(agentDict)
            sortedAgentId = agentDict.keys()
            sortedAgentFeature = list(agentDict.values())
            self.currentagentNormFeature = self.imageTranslator.normalizeFeatures(sortedAgentFeature, 1)
            state = self.imageTranslator.transformToImage(sortedAgentFeature, buildingList)
            state_ = copy.deepcopy(state).cpu().type(torch.IntTensor)

            #sortedAgentPCA = self.imageTranslator.transformToPCA(sortedAgentFeature)
            #agentObsPos = torch.range(0, 17).long().cuda().unsqueeze(1)
            #sortedAgentPCA = torch.cat([agentObsPos, sortedAgentPCA], dim=-1) 
#
            #state[sortedAgentPCA[:,0], sortedAgentPCA[:,1], sortedAgentPCA[:,2]] = sortedAgentPCA[:,3]
            rewardSorted = sorted(self.rewardPerAgent.items())
            rewardSorted = dict(rewardSorted)
            rewardSorted = list(rewardSorted.values())
            rewardSorted = torch.Tensor(rewardSorted)
            observations = state.cpu()
            #observations = torch.cat((state_, observations), 0).cpu()
            #print(len(state)-18)
            torch_rewards = torch.FloatTensor(rewardSorted).cpu()
            #total_reward += torch_rewards.sum()
            self.agent_score_sum += torch_rewards.sum().cpu().numpy()
            self.global_score += globalReward
            #print(state)
            #state = self.stateBuilder.kPcaImagify(state, 83)
            #self.states.extend(state)
            #print(self.parser.getEntitys())
            return state_, rewardSorted 
        except Exception as e :
            print("############")
            traceback.print_exc()
            print(e)
            #print(stateandscoreinfo)  


    def getState(self):
        agentList, buildingList, self.burningBuildingDict, agentDict = self.parser.getCumulativeChangeSetVectorList()
        agentDict = sorted(agentDict.items())
        agentDict = dict(agentDict)
        sortedAgentFeature = list(agentDict.values())
        self.currentagentNormFeature = self.imageTranslator.normalizeFeatures(sortedAgentFeature, 1)        
        state = self.imageTranslator.transformToImage(sortedAgentFeature, buildingList)
        #state = state.float() / 255
        state_ = copy.deepcopy(state).cpu().type(torch.IntTensor)
        return state_

    def close(self):
        self.clientSocket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_space", type=int, default=0, help='0 is weight vector, 1 is feature + weight vector')
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument("--aux_reward", default=1, type=float, help="weight for aux reward")
    parser.add_argument("--scale_reward", default=1, type=float, help="weight for aux reward")
    parser.add_argument("--file_prefix", default="", type=str)
    parser.add_argument("--itr_before_train", default=33,type=int)
    parser.add_argument("--exploration_eps", default=500, type=int)
    parser.add_argument("--init_noise_scale",default=0., type=float)
    parser.add_argument("--lr_scale",default=1.0, type=float)
    parser.add_argument("--out_activation",default=0.0, type=float)
    parser.add_argument("--init_beta",default=0.0001, type=float)
    parser.add_argument("--final_beta",default=0.01, type=float)
    parser.add_argument("--beta_eps", default=1000, type=float)
    ex_args = parser.parse_args()

    child_proc = subprocess.Popen(["bash", "./start.sh", "--map", "../suwon_mod/map", "-c", "../suwon_mod/config"], stdout=PIPE, stderr=PIPE)
    #subprocess.call(["bash", "./start.sh", "--map", "../maps/gml/test/map", "-c", "../maps/gml/test/config"])
    time.sleep(6)     
    env = RCRS_Env(ex_args)
    #env.action_space
    for i in range(800):
        if((i%400 >= (400-1)) and (i > 0)):
            env.reset(0, 0) 
        else :   
            print(i) 
            #print(f"env.actionspace {env.actionspace}")
            #randomactionsample =  random.randint(0, len(env.actionspace)-1)
            randomactionsample = torch.zeros(18, 6).cpu()
            #randomactionsample = 1

            state, rewards = env.step(randomactionsample)
            state = state.type(torch.cuda.FloatTensor)
            state2 = torch.where(state==(-1), torch.tensor(255.0).cuda(), state.cuda())
            print((state-state2).sum())
            state2 = torch.where(state<0, torch.tensor(0.0).cuda(), state.cuda())
            print((state-state2).sum())

            #if(i < 200):
            #    state = state.cpu()
            #    state = state.numpy().astype(np.uint8)
#
            #    state = Image.fromarray(state)
            #    state.save(str(i)+ ".png")
            #if(i==100):
            #    state = Image.fromarray(state, "L")
            #    state.save("state.png")
            #if(i==150):
            #    state = Image.fromarray(state, "L")
            #    state.save("state2.png")    
    #with open(f'/scratch/x2026a02/new_rcrs/rcrs-server/boot/states_sample_policy.json', 'w') as f:
    #    json.dump(env.states, f)    
    env.close()
