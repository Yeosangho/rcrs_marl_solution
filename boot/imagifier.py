import torch
import json

class Imagifier:
    def __init__(self, h, w, c, state_data_path):
        torch.cuda.set_device(1)

        featureList = []
        with open(state_data_path) as f:
            featureList.extend(json.load(f))
        self.buildingTensor = []
        self.agentTensor = []
        for i in range(len(featureList)):
            if(featureList[i][7] == 0):
                vector = featureList[i][:7]
                vector.remove(vector[3])
                self.buildingTensor.append(vector)
            elif(featureList[i][7]== 1):
                vector = featureList[i][:7]
                vector.remove(vector[3])                
                self.agentTensor.append(vector)
        #Divide tensor as two cases "agent" "building"
        #buildingTensor = torch.Tensor(buildingTensor).cuda()
        #agentTensor = torch.Tensor(agentTensor).cuda()

        self.featureMax_list = []
        self.featureMin_list = []

        self.featureGamma = 1
        self.featureBeta = 0
        self.featureNum_list = []

        self.U_list = []
        self.S_list = []
        self.V_list = []

        self.pcaMax_list = []
        self.pcaMin_list = []

        self.color = 256
        self.h = h
        self.w = w
        self.c = c
        self.boundMin = torch.Tensor([0,0,0,0]).cuda()
        self.boundMax = torch.Tensor([self.h-1, self.w-1, self.color-1, self.c -1]).cuda()

        self.pcaGamma = torch.Tensor([float(self.h-1), float(self.w-1), float(self.color-1), float(self.c-1)]).cuda()
        self.pcaBeta = torch.Tensor([0,0,0,0]).cuda()

        self.initialize(self.buildingTensor, 3)
        self.initialize_agent(self.agentTensor, 2)
    def initialize(self, featureList, featureNum):
        featureTensor = torch.Tensor(featureList).cuda()
        featureMin = torch.min(featureTensor, 0)[0]
        t1 = featureTensor - featureMin
        featureMax = torch.max(t1, 0)[0]
        stdFeature = t1 / (featureMax+1e-6)
        self.featureNum_list.append(featureNum)
        self.featureMax_list.append(featureMax)
        self.featureMin_list.append(featureMin)


        normFeature = stdFeature * self.featureGamma + self.featureBeta

        U, S, V = torch.pca_lowrank(normFeature, q=6, center=True, niter=10)

        self.U_list.append(U)
        self.S_list.append(S)
        self.V_list.append(V)
        #if(featureNum == 2):
        #    pcaFeature = torch.matmul(normFeature, V[:, : featureNum])
        #else:
        pcaFeature = torch.matmul(normFeature, V[:, : featureNum])
        pcaMin = torch.min(pcaFeature, 0)[0]
        t1 = pcaFeature - pcaMin
        pcaMax = torch.max(t1, 0)[0]

        #t2 = torch.sqrt(self.pcaVar + 1e-6) 
        normPCA = t1 / (pcaMax + 1e-6)

        self.pcaMax_list.append(pcaMax)
        self.pcaMin_list.append(pcaMin)

        stdPCA =  normPCA * self.pcaGamma[:featureNum] + self.pcaBeta[:featureNum]
        clippedPCA = torch.max(torch.min(stdPCA, self.boundMax[:featureNum]), self.boundMin[:featureNum] )  
        clippedPCA = clippedPCA.long()

        imageTensor = torch.zeros(self.h,self.w).long().cuda()

        imageTensor[clippedPCA[:,0], clippedPCA[:,1]] = clippedPCA[:,2]

    def initialize_agent(self, featureList, featureNum):
        featureTensor = torch.Tensor(featureList).cuda()
        featureMin = torch.min(featureTensor, 0)[0]
        t1 = featureTensor - featureMin
        featureMax = torch.max(t1, 0)[0]
        stdFeature = t1 / (featureMax+1e-6)
        normFeature = stdFeature * self.featureGamma + self.featureBeta
        normFeature = normFeature[:,:2]
        self.featureNum_list.append(featureNum)
        self.featureMax_list.append(featureMax)
        self.featureMin_list.append(featureMin)        
        #print(normFeature.shape)
        #if(featureNum == 2):
        #    U, S, V = torch.pca_lowrank(normFeature[:,:2], q=2, center=True, niter=10)
        #else:
        #    U, S, V = torch.pca_lowrank(normFeature, q=6, center=True, niter=10)
#
        #self.U_list.append(U)
        #self.S_list.append(S)
        #self.V_list.append(V)
        #pcaFeature = torch.matmul(normFeature[:,:2], V[:, : featureNum])
#
        #pcaMax = torch.max(pcaFeature, 0)[0]
        #pcaMin = torch.min(pcaFeature, 0)[0]
        #t1 = pcaFeature - pcaMin
        ##t2 = torch.sqrt(self.pcaVar + 1e-6) 
        #normPCA = t1 / (pcaMax + 1e-6)
#
        #self.pcaMax_list.append(pcaMax)
        #self.pcaMin_list.append(pcaMin)

        normFeature =  normFeature * self.pcaGamma[:featureNum] + self.pcaBeta[:featureNum]
        clippedPCA = torch.max(torch.min(normFeature, self.boundMax[:featureNum]), self.boundMin[:featureNum] )  
        clippedPCA = clippedPCA.long()
        imageTensor = torch.zeros(self.h,self.w).long().cuda()
   
        imageTensor[clippedPCA[:,0], clippedPCA[:,1]] = 25


    def transformToImage(self, agentList, buildingList):
        imageTensor1 = self._transformToImage_agent(agentList, 1)
        if(len(buildingList)>0):
            imageTensor2 = self._transformToImage(buildingList, 0)
            imageTensor = torch.where(imageTensor1 != 0, imageTensor1, imageTensor2)
            return imageTensor
        else:
            return  imageTensor1 

    def _transformToImage_agent(self, featureList, idx):
        featureTensor = torch.Tensor(featureList).cuda()
        featureTensor = torch.cat([featureTensor[:,:3], featureTensor[:,4:]], dim=1)

        t1 = featureTensor - self.featureMin_list[idx]
        #t2 = torch.sqrt(self.featureVariance + 1e-6)
        stdFeature = t1 / (self.featureMax_list[idx] + 1e-6)

        normFeature = stdFeature * self.featureGamma + self.featureBeta
        normFeature = normFeature[:,:2]
        #pcaFeature = torch.matmul(normFeature, self.V_list[idx][:, : self.featureNum_list[idx]])
        #t1 = pcaFeature - self.pcaMin_list[idx]
        ##t2 = torch.sqrt(self.pcaVar + 1e-6) 
        #normPCA = t1 / (self.pcaMax_list[idx] + 1e-6)

        normFeature =  normFeature * self.pcaGamma[:self.featureNum_list[idx]] + self.pcaBeta[:self.featureNum_list[idx]]
        clippedPCA = torch.max(torch.min(normFeature, self.boundMax[:self.featureNum_list[idx]]), self.boundMin[:self.featureNum_list[idx]] )  
        clippedPCA = clippedPCA.long()
        #imageTensor = torch.zeros(self.c, self.h,self.w).long().cuda()
        #imageTensor[clippedPCA[:,3], clippedPCA[:,0], clippedPCA[:,1]] = clippedPCA[:,2]
        imageTensor = torch.zeros(self.h,self.w).long().cuda()
        for i in range(18):
            imageTensor[clippedPCA[i,0], clippedPCA[i,1]] = -(i+1)
        return imageTensor

    def _transformToImage(self, featureList, idx):
        featureTensor = torch.Tensor(featureList).cuda()
        featureTensor = torch.cat([featureTensor[:,:3], featureTensor[:,4:]], dim=1)
        t1 = featureTensor - self.featureMin_list[idx]
        #t2 = torch.sqrt(self.featureVariance + 1e-6)
        stdFeature = t1 / (self.featureMax_list[idx] + 1e-6)

        normFeature = stdFeature * self.featureGamma + self.featureBeta
        pcaFeature = torch.matmul(normFeature, self.V_list[idx][:, : self.featureNum_list[idx]])
        t1 = pcaFeature - self.pcaMin_list[idx]
        #t2 = torch.sqrt(self.pcaVar + 1e-6) 
        normPCA = t1 / (self.pcaMax_list[idx] + 1e-6)

        stdPCA =  normPCA * self.pcaGamma[:self.featureNum_list[idx]] + self.pcaBeta[:self.featureNum_list[idx]]
        clippedPCA = torch.max(torch.min(stdPCA, self.boundMax[:self.featureNum_list[idx]]), self.boundMin[:self.featureNum_list[idx]] )  
        clippedPCA = clippedPCA.long()
        #imageTensor = torch.zeros(self.c, self.h,self.w).long().cuda()
        #imageTensor[clippedPCA[:,3], clippedPCA[:,0], clippedPCA[:,1]] = clippedPCA[:,2]
        imageTensor = torch.zeros(self.h,self.w).long().cuda()
        imageTensor[clippedPCA[:,0], clippedPCA[:,1]] = clippedPCA[:,2]

        return imageTensor

    
    def transformToPCA(self, featureList, idx):
        featureTensor = torch.Tensor(featureList).cuda()
        featureTensor = torch.cat([featureTensor[:,:3], featureTensor[:,4:]], dim=1)
        t1 = featureTensor - self.featureMin_list[idx]
        #t2 = torch.sqrt(self.featureVariance + 1e-6)
        stdFeature = t1 / (self.featureMax_list[idx] + 1e-6)

        normFeature = stdFeature * self.featureGamma + self.featureBeta
        pcaFeature = torch.matmul(normFeature, self.V_list[idx][:, : self.featureNum_list[idx]])
        t1 = pcaFeature - self.pcaMin_list[idx]
        #t2 = torch.sqrt(self.pcaVar + 1e-6) 
        normPCA = t1 / (self.pcaMax_list[idx] + 1e-6) 
        stdPCA =  normPCA * self.pcaGamma[:self.featureNum_list[idx]] + self.pcaBeta[:self.featureNum_list[idx]]
        clippedPCA = torch.max(torch.min(stdPCA, self.boundMax[:self.featureNum_list[idx]]), self.boundMin[:self.featureNum_list[idx]] )  
        clippedPCA = clippedPCA.long()

        return clippedPCA

    def normalizeFeatures(self, featureList, idx):
        featureTensor = torch.Tensor(featureList).cuda()
        featureTensor = torch.cat([featureTensor[:,:3], featureTensor[:,4:]], dim=1)
        t1 = featureTensor - self.featureMin_list[idx]
        #t2 = torch.sqrt(self.featureMax + 1e-6)
        stdFeature = t1 / (self.featureMax_list[idx] + 1e-6)
        normFeature = stdFeature * self.featureGamma + self.featureBeta

        return normFeature
