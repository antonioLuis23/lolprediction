import pandas as pd
import numpy as np
import math

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def strToNpArray(strArray):

    if strArray == '[]':
        return []
    newArray = strArray.replace("[","").replace("]","")
    newArray = newArray.split(',')
    newArray = [float(i) for i in newArray if hasNumbers(i)]
    return np.array(newArray)

def convertToMin(array, gameLength):
    if array == []:
        return np.zeros(gameLength)
    array= array.astype(int)
    array = np.sort(array)
    arrayMin = np.zeros(gameLength)
    for i in range(len(array)):
        arrayMin[array[i]]=i+1
        if i+1<len(array):
            arrayMin[array[i]:array[i+1]]=i+1
    arrayMin[array[i]:] = i+1
    return arrayMin

def convertToMin2(array, gameLength):
    if array == []:
        return np.zeros(gameLength)
    array= array.astype(int)
    array = np.sort(array)
    arrayMin = np.zeros(gameLength)
    for i in range(len(array)):
        arrayMin[array[i]]=1

    return arrayMin


def descobreArrayObj(df, obj):
    gameLength = df.loc[i, ['gamelength']].values[0]
    arrayObj= strToNpArray(df.loc[i, [obj]].values[0])
    arrayObjMin=convertToMin(arrayObj, gameLength)
    return arrayObjMin

def descobreArrayObj2(df, obj):
    gameLength = df.loc[i, ['gamelength']].values[0]
    arrayObj= strToNpArray(df.loc[i, [obj]].values[0])
    arrayObjMin=convertToMin2(arrayObj, gameLength)
    return arrayObjMin

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
df = pd.read_csv('LeagueofLegends.csv')
N = len(df)

teams = np.unique(df[['blueTeamTag', 'redTeamTag']].astype(str))
teams = np.delete(teams, np.where(teams=='nan'))
champString = ['blueTopChamp', 'blueJungleChamp','blueMiddleChamp','blueADCChamp', 'blueSupportChamp',
'redTopChamp', 'redJungleChamp','redMiddleChamp','redADCChamp', 'redSupportChamp']
champions = np.unique(df[champString].astype(str))
print(champions)
print(teams)
matrizDeMinutos = np.array([])
INICIO = 0
for i in range(INICIO, N, 1):
    print(i)
    blueTeam = df.loc[i, ['blueTeamTag']].values[0]
    redTeam = df.loc[i, ['redTeamTag']].values[0]
    if not isinstance(blueTeam, str) or not isinstance(redTeam, str):
        continue
    gameLength = df.loc[i, ['gamelength']].values[0]

    objs = ['bInhibs', 'bDragons', 'bBarons', 'bHeralds', 'rTowers', 'rInhibs', 'rDragons', 'rBarons', 'rHeralds']

    bTowers = strToNpArray(df.loc[i, ['bTowers']].values[0])

    #print('bTowers',bTowers)
    arrayObjsMin=convertToMin(bTowers, gameLength)
    #print('arrayObjsMin', arrayObjsMin)
    for obj in objs:
        objArray = descobreArrayObj(df, obj)
        arrayObjsMin=np.vstack((arrayObjsMin, objArray))

    golds = ['goldblue', 'goldred', 'goldblueTop', 'goldredTop', 'goldblueJungle', 'goldredJungle', 'goldblueMiddle',
    'goldredMiddle','goldblueADC', 'goldredADC', 'goldblueSupport','goldredSupport']

    j=0
    while j<len(golds):
        goldBlue = strToNpArray(df.loc[i, [golds[j]]].values[0])
        goldRed = strToNpArray(df.loc[i, [golds[j+1]]].values[0])
        golddiff = goldBlue - goldRed

        arrayObjsMin=np.vstack((arrayObjsMin, goldBlue))
        arrayObjsMin=np.vstack((arrayObjsMin, goldRed))
        arrayObjsMin=np.vstack((arrayObjsMin, golddiff))
        j+=2

    teamBlue = ['blueTop', 'blueJungle', 'blueMiddle', 'blueADC', 'blueSupport']
    teamRed  = ['redTop', 'redJungle', 'redMiddle', 'redADC', 'redSupport']

    bluePlayers = []
    for pos in teamBlue:
        name = df.loc[i, [pos]].values[0]
        bluePlayers.append(name)

    redPlayers = []
    for pos in teamRed:
        name = df.loc[i, [pos]].values[0]
        redPlayers.append(name)

    bKills = eval(df.loc[i, ['bKills']].values[0])
    rKills = eval(df.loc[i, ['rKills']].values[0])

    blueTeam = df.loc[i, ['blueTeamTag']].values[0]
    redTeam = df.loc[i, ['redTeamTag']].values[0]
    rMinDeath = np.zeros((5, gameLength))
    rMinKill = np.zeros((5, gameLength))
    rQuantDeath = np.zeros(5)
    rQuantKill = np.zeros(5)
    bMinDeath = np.zeros((5, gameLength))
    bMinKill = np.zeros((5, gameLength))
    bQuantDeath = np.zeros(5)
    bQuantKill = np.zeros(5)


    killBlueMin = []
    if not bKills:
        killBluePerMin = np.zeros(gameLength)

    else:
        for k in range(len(bKills)):
            kill = bKills[k]
            killBlueMin.append(kill[0])

            #descobre kills e deaths dos players red e blue
            namePlayerRed = kill[1].replace(str(redTeam)+" ", "")
            if namePlayerRed in redPlayers:
                indice = redPlayers.index(namePlayerRed)
                minute = int(kill[0])
                rQuantDeath[indice]+=1
                rMinDeath[indice,minute:] = rQuantDeath[indice]

            if blueTeam == "nan":
                namePlayerBlue = kill[2]
            else:
                namePlayerBlue = kill[2].replace(str(blueTeam)+" ", "")
            if namePlayerBlue in bluePlayers:
                indice = bluePlayers.index(namePlayerBlue)
                minute = int(kill[0])
                bQuantKill[indice]+=1
                bMinKill[indice, minute:] = bQuantKill[indice]


        killBlueMin = np.array(killBlueMin)
        killBluePerMin = convertToMin(killBlueMin, gameLength)

    arrayObjsMin=np.vstack((arrayObjsMin, killBluePerMin))

    killRedMin = []
    #print(rKills)
    if not rKills:
        killRedPerMin = np.zeros(gameLength)
    else:
        for p in range(len(rKills)):
            kill = rKills[p]
            killRedMin.append(kill[0])

            #descobre kills e deaths dos players red e blue
            if blueTeam == "nan":
                namePlayerBlue = kill[1]
            else:
                namePlayerBlue = kill[1].replace(str(blueTeam)+" ", "")
            if namePlayerBlue in bluePlayers:
                indice = bluePlayers.index(namePlayerBlue)
                minute = int(kill[0])
                bQuantDeath[indice]+=1
                bMinDeath[indice,minute:] = bQuantDeath[indice]
            #print('redTeam', redTeam)
            #print('kill', kill)
            namePlayerRed = kill[2].replace(redTeam+" ", "")
            if kill[2] in redPlayers:
                indice = redPlayers.index(namePlayerRed)
                minute = int(kill[0])
                rQuantKill[indice]+=1
                rMinKill[indice, minute:] = rQuantKill[indice]

        killRedMin = np.array(killRedMin)
        killRedPerMin = convertToMin(killRedMin, gameLength)

    arrayObjsMin=np.vstack((arrayObjsMin, killRedPerMin))

    arrayObjsMin = np.vstack((arrayObjsMin, bMinKill))
    arrayObjsMin = np.vstack((arrayObjsMin, rMinKill))
    arrayObjsMin = np.vstack((arrayObjsMin, bMinDeath))
    arrayObjsMin = np.vstack((arrayObjsMin, rMinDeath))

    minutes = list(range(gameLength))
    minutes = np.array(minutes)+1
    arrayObjsMin=np.vstack((arrayObjsMin, minutes))
    bResult = df.loc[i, ['bResult']].values[0]
    result = np.zeros(gameLength)
    result[:] = bResult
    arrayObjsMin=np.vstack((arrayObjsMin, result))

    rResult = df.loc[i, ['rResult']].values[0]
    resultZeros = np.zeros(gameLength)
    resultZeros[:] = rResult
    arrayObjsMin=np.vstack((arrayObjsMin, resultZeros))

    if type(blueTeam) == float:
        if  math.isnan(blueTeam):
            blueTeam = 'yoeFW'
    blueTeamIndex = np.where(teams==blueTeam)
    blueTeamMin = np.zeros(gameLength)
    blueTeamMin[:] = blueTeamIndex
    arrayObjsMin=np.vstack((arrayObjsMin, blueTeamMin))

    redTeamIndex = np.where(teams== redTeam)
    redTeamMin = np.zeros(gameLength)
    redTeamMin[:] = redTeamIndex
    arrayObjsMin=np.vstack((arrayObjsMin, redTeamMin))

    for championPos in champString:
        nameChampion = df.loc[i, [championPos]].values[0]
        index = np.where(champions==nameChampion)
        champMin = np.zeros(gameLength)
        champMin[:] = index
        arrayObjsMin=np.vstack((arrayObjsMin, champMin))
    arrayObjsMin=np.transpose(arrayObjsMin)

    if i==INICIO:
        matrizDeMinutos = arrayObjsMin
    else:
        matrizDeMinutos = np.vstack((matrizDeMinutos, arrayObjsMin))

columnsName=['bTowers','bInhibs', 'bDragons', 'bBarons', 'bHeralds', 'rTowers', 'rInhibs', 'rDragons', 'rBarons', 'rHeralds',
'goldblue', 'goldred','golddiff', 'goldblueTop', 'goldredTop','golddiffTop', 'goldblueJungle', 'goldredJungle', 'golddiffJungle', 'goldblueMiddle',
    'goldredMiddle', 'golddiffMiddle','goldblueADC', 'goldredADC', 'golddiffADC', 'goldblueSupport','goldredSupport', 'golddiffSupport', 'numKillBlue','numKillRed','blueKillTop',
     'blueKillJungle', 'blueKillMiddle','blueKillADC', 'blueKillSupport','redKillTop', 'redKillJungle',
    'redKillMiddle','redKillADC','redKillSupport', 'blueDeathTop','blueDeathJungle', 'blueDeathMiddle', 'blueDeathADC', 'blueDeathSupport', 'redDeathTop', 'redDeathJungle',
    'redDeathMiddle','redDeathADC','redDeathSupport', 'minute', 'resultBlue','resultRed', 'blueTeam', 'redTeam'] + champString
print('matrizDeMinutosShape',matrizDeMinutos.shape)
print('len', len(columnsName))
newdf = pd.DataFrame(data = matrizDeMinutos, columns = columnsName)
#print(newdf)
newdf.to_csv('newLeague.csv', index=False)
