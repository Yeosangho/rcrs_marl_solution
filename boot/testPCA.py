from imagifier import Imagifier
import ast
import numpy as np
from PIL import Image
import json
import torch
from sklearn.preprocessing import MinMaxScaler
#from matplotlib import pyplot as plt

stateList = []
strList = ""
with open('/scratch/x2026a02/new_rcrs/rcrs-server/boot/states_sample_policy.json') as f:
    stateList.extend(json.load(f))
torch.cuda.set_device(1)
stateTensor = torch.Tensor(stateList).cuda()
meanTensor = torch.mean(stateTensor, 0)
varTensor = torch.var(stateTensor, 0)
#stateArray = np.array(stateList)
#print(meanTensor.size())


#print(varTensor.size())

standardTensor1 = stateTensor - meanTensor
standardTensor2 = torch.sqrt(varTensor + 1e-6)
standardTensor = standardTensor1 / standardTensor2
#print(standardTensor.size())

#apply 3 sigma rule to fit 99.7% of value in (+-)0.5
gamma = 1
#moving center to 0.5
#beta = 0.5
beta = 0
normalizedTensor = standardTensor * gamma + beta

U,S,V = torch.pca_lowrank(normalizedTensor, q=9, center=True, niter=10)
print(normalizedTensor.size())
print(V.T.size())
print(U.size())
print(S.size())
resultTensor=torch.matmul(normalizedTensor, V[:, :4])   

#apply different batch normalization on each dimension 
h = 84
w = 84
color = 256
frame = 3
boundMin = torch.Tensor([0, 0, 0, 0]).cuda()
boundMax = torch.Tensor([h-1,w-1,color-1, frame-1]).cuda()
resultMean = torch.mean(resultTensor, 0)
resultVar = torch.var(resultTensor, 0)
standardTensor1 = resultTensor - resultMean
standardTensor2 = torch.sqrt(resultVar + 1e-6)
resultTensor = standardTensor1 / standardTensor2
gammaTensor = torch.Tensor([float(h)/6, float(w)/6, float(color)/6, float(frame)/6]).cuda()
betaTensor = torch.Tensor([float(h)/2,float(w)/2, float(color)/2, float(frame)/2 ]).cuda()
#[number of data, 3]
resultTensor = resultTensor * gammaTensor + betaTensor
resultTensor = torch.max(torch.min(resultTensor, boundMax), boundMin)
resultTensor = resultTensor.long()
imageTensor = torch.zeros(h,w, frame).long().cuda()
imageTensor[resultTensor[:,0], resultTensor[:,1], resultTensor[:,3]] = resultTensor[:,2]
a = [0] * 256
b = [0] * 84
imageTensor = imageTensor.cpu()
imageTensor = imageTensor.numpy().astype(np.uint8)

image1 = Image.fromarray(imageTensor)
image1.save("aaa_nominmax.png")
#print(111)
#print()
 

##minMaxScaler = MinMaxScaler().fit(stateList)
##zeroToOneList = minMaxScaler.transform(stateList)
#stateBuilder = Imagifier()
#stateBuilder.fitPcaModel(stateList)
#zeroToOneList =np.asarray(stateList)
#zeroToOneList =zeroToOneList.reshape(-1, 38, 10);
#
#image1 = stateBuilder.pcaImagify(zeroToOneList[100], 32)
#image2 = stateBuilder.pcaImagify(zeroToOneList[110], 32)
#image3 = image1-image2
#with open(f"resultList.csv", "w") as f:
#	for i in range(len(zeroToOneList[110])):
#		print(zeroToOneList[110][i] - zeroToOneList[120][i], file=f)
#with open(f"resultimage.csv", "w") as f:
#	for i in range(len(image3)):
#		print(image3[i], file=f)
#	
#image1 = Image.fromarray(image1, "L")
#image2 = Image.fromarray(image2, "L")
#image1.save("aaa_nominmax.png")
#image2.save("bbb_nominmax.png")