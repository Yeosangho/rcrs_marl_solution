# 의존성
# mongoDB
# pip install pymongo
import json
import copy

class RcrsJsonStateStringParser:
    def __init__(self):
        self.entityDict = {} #id-type piar
        self.cumulateChangeSet = {}
        self.entityDictPerType = {}
        self.buildings = {}
        self.fbs = {}
    def reset(self):
        self.entityDict = {} #id-type piar
        self.cumulateChangeSet = {}
        self.buildings = {}
        self.agents = {}
        self.entityDictPerType = {}
    def addBuildings(self, jsonString, nearBuildingJsonString, type):
        jsonArray = json.loads(jsonString)
        nearBuildingJsonArray = json.loads(nearBuildingJsonString)
        entityType = 0
        for ob in jsonArray:
            entityId = ob["id"]["id"]
            self.entityDict[entityId] = type
            self.cumulateChangeSet[entityId] = initInfoParser(ob, -1);
            
            #"nearBuildings", "dispatchedTeams", "entityType"
            #print(entityId)
            self.cumulateChangeSet[entityId].append(nearBuildingJsonArray[str(entityId)])
            self.cumulateChangeSet[entityId].append(0)
            self.cumulateChangeSet[entityId].append(0)
            if( type in self.entityDictPerType):
                self.entityDictPerType[type].append(entityId)
            else :
                self.entityDictPerType[type] = [entityId]

            #print(str(entityId) + type + "added")                
    def addFirebrigades(self, jsonString, type):
        jsonArray = json.loads(jsonString)
        entityType = 1
        for ob in jsonArray:
            entityId = ob["id"]["id"]
            self.entityDict[entityId] = type
            self.cumulateChangeSet[entityId] = initInfoParser(ob, -1);
            #"nearBuildings", "dispatchedTeams", "entityType"
            self.cumulateChangeSet[entityId].append(-1)
            self.cumulateChangeSet[entityId].append(-1)
            self.cumulateChangeSet[entityId].append(1)

            if( type in self.entityDictPerType):
                self.entityDictPerType[type].append(entityId)
            else :
                self.entityDictPerType[type] = [entityId]
            self.fbs[entityId] = self.cumulateChangeSet[entityId]    
        self.fbs = sorted(self.fbs.items())
        self.fbs = dict(self.fbs)
            #print(str(entityId) + type + "added")


    def getEntitys(self):
        return self.entityDict

    def parse(self, jsonString, fallback):
        jsonObject = json.loads(jsonString)
        vectorList = []

        for id, type in sorted(self.entityDict.items()):
            # print(type+" "+str(id)+"차례")
            object = None
            try:
                object = jsonObject["changes"][str(id)]
                # print("object 찾았음")
            except:
                object = None

            vector = infoParser(object, fallback);

            vectorList.append(vector)

        return vectorList

    def linearParse(self, jsonString, fallback):
        vectorList = self.parse(jsonString, fallback)

        linearArray = []

        for vector in vectorList:
            for value in vector:
                linearArray.append(value)

        return linearArray

    def getFieldNameList(self):
        #entityType(0(building) 1(Other Team), 2(Me))
        #return ["x", "y", "floors", "groundArea","fieryness", "temperature", "nearBuildings", "dispatchedTeams", "entityType"]
        return ["x", "y", "totalArea", "temperature", "nearBuildings", "dispatchedTeams", "entityType"]

    def cumulativeParse(self, jsonString, dispatchJsonString):
        jsonObject = json.loads(jsonString)
        dispatchJson = json.loads(dispatchJsonString)
        for id, type in self.entityDict.items():
            # print(type+" "+str(id)+"차례")
            object = None
            try:
                object = jsonObject["changes"][str(id)]
                # print("object 찾았음")
            except:
                #이번 step에서 변경되지 않음
                continue

            vector = infoParser(object, None)

            # cumulate
            for i in range(0, len(vector)):
                value = vector[i]
                if(value != None):
                    self.cumulateChangeSet[id][i] = value

        return self.getCumulativeChangeSetVectorList(dispatchJson)
    
    def cumulativeLinearParse(self, jsonString):
        self.cumulativeParse(jsonString)
        return self.getCumulativeChangeSetLinearVector()


    def getCumulativeChangeSetVectorList(self, dispatchJson=None):
        vectorList = []
        buildingList = []
        agentList = []
        agentDict = {}
        burningBuildingDict = {}
        burningBuilding_cnt = 0
        for id, vector in sorted(self.cumulateChangeSet.items()):
            if((dispatchJson != None) and (str(id) in dispatchJson.keys())):
                vector[6] = dispatchJson[str(id)]
            elif(int(vector[7]) == 0):
                vector[6]  = 0        
                       
            if(int(vector[3]) < 4 and int(vector[3]) > 0 and ( vector[7] == 0 )):
                #vector = copy.deepcopy(vector)
                #vector.remove(vector[3])
                buildingList.append(vector[:7])
                burningBuildingDict[id] = vector[:7]
                burningBuilding_cnt += 1
            elif(int(vector[7]) != 0):
                #vector = copy.deepcopy(vector)
                #vector.remove(vector[3])
                agentList.append(vector[:7])
                copyVector = copy.deepcopy(vector[:7])
                #copyVector[8] = 2
                agentDict[id] = copyVector
            elif((dispatchJson != None) and (str(id) in dispatchJson.keys())):
                #vector = copy.deepcopy(vector)
                #vector.remove(vector[3])
                buildingList.append(vector[:7])               
        return agentList, buildingList, burningBuildingDict, agentDict

    def getCumulativeChangeSetLinearVector(self):
        vectorList = self.getCumulativeChangeSetVectorList()

        linearArray = []

        for vector in vectorList:
            for value in vector:
                linearArray.append(value)

        return linearArray

    def getCumulativeChangeSetVectorListByType(type):
        vectorList = []

        for id, vector in sorted(self.cumulateChangeSet.items()):
            if self.entityDict[id] == type:
                vectorList.append(vector)

        return vectorList;

    def getCumulativeChangeSetLinearVectorByType(type):
        vectorList = self.getCumulativeChangeSetVectorListByType(type)

        linearArray = []

        for vector in vectorList:
            for value in vector:
                linearArray.append(value)

        return linearArray

    def checkBurningBuilding(self, firelevel):
        firelevel = float(firelevel)
        if(float(firelevel) < 4 and float(firelevel) > 0 ):
            return True
        else :
            return False
    


# ----------------------------

#범용 함수들

def infoParser(object, fallback):
    vector = []


    vector.append(tryGetValue(object, "x", "value", fallback))
    vector.append(tryGetValue(object, "y", "value", fallback))
    vector.append(tryGetValue(object, "totalArea", "value", fallback))
    vector.append(tryGetValue(object, "fieryness", "value", fallback))
    vector.append(tryGetValue(object, "temperature", "value", fallback))

    #blockadeList = tryGetValue(object, "blockades", "ids", fallback)
    #if(isinstance(blockadeList, list) == False):
    #    vector.append(fallback)
    #else:
    #    vector.append(len(blockadeList)) #길에 있는 blockade의 개수

    return vector

# 첫 entity 정보 파싱용 함수
def initInfoParser(initObject, fallback):
    vector = []

    vector.append(tryGetInitValue(initObject, "x", "value", fallback))
    vector.append(tryGetInitValue(initObject, "y", "value", fallback))
    vector.append(tryGetInitValue(initObject, "totalArea", "value", fallback))
    vector.append(tryGetInitValue(initObject, "fieryness", "value", fallback))
    vector.append(tryGetInitValue(initObject, "temperature", "value", fallback))

    #print(vector)

    #blockadeList = tryGetInitValue(object, "blockades", "ids", fallback)
    #if(isinstance(blockadeList, list) == False):
    #    vector.append(fallback)
    #else:
    #    vector.append(len(blockadeList)) #길에 있는 blockade의 개수

    return vector

def tryGetValue(object, propertyName, propertyAttribute, fallback):
    value = None
    try:
        value = object["urn:rescuecore2.standard:property:" + propertyName][propertyAttribute]
    except:
        value = fallback

    return value

def tryGetInitValue(object, propertyName, propertyAttribute, fallback):
    value = None
    try:
        value = object[propertyName][propertyAttribute]
    except:
        value = fallback

    return value

def getAveragePoint(pointList):
    # 벡터 평균 내기
    xSum = 0
    ySum = 0
    count = len(pointList)/2
    for i in range(0, len(pointList), 2):
        xSum = xSum + pointList[i]
        ySum = ySum + pointList[i+1]

    return {"x": xSum/count, "y": ySum/count}