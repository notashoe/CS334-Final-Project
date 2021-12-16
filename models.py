from matplotlib.pyplot import disconnect
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    pastSeasonsX = pd.read_csv("pastSeasonsX.csv")
    pastSeasonsY = pd.read_csv("pastSeasonsY.csv").fillna(0)
    currSeasonX = pd.read_csv("currSeasonX.csv").fillna(0)
    pastSeasonsY = pastSeasonsY.to_numpy().ravel()

    NNacc = 0
    LRacc = 0
    RFacc = 0
    NNaccMVPS = 0
    LRaccMVPS = 0
    RFaccMVPS = 0
    NNaccNON = 0
    LRaccNON = 0
    RFaccNON = 0

    ksplits = 5

    kNumNeigh = 1
    Kweight = 'uniform'

    #LRpenalty = 'none'
    LRpenalty = 'l2'
    LRc = 7


    kf = KFold(n_splits=ksplits, shuffle = True)
    for train_index, test_index in kf.split(pastSeasonsX):
        xTrain, xTest = pastSeasonsX.iloc[train_index], pastSeasonsX.iloc[test_index]
        yTrain, yTest = pastSeasonsY[train_index], pastSeasonsY[test_index]

        neigh = KNeighborsClassifier(n_neighbors=kNumNeigh,weights=Kweight)
        neigh.fit(xTrain[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']], yTrain)
        yHatNeigh = neigh.predict(xTest[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
        NNacc += accuracy_score(yTest, yHatNeigh)
        NNaccMVPS += accuracy_score(yTest[yTest.nonzero()],yHatNeigh[yTest.nonzero()])
        NNaccNON += accuracy_score(yTest[np.argwhere(yTest == 0)],yHatNeigh[np.argwhere(yTest == 0)])

        # print("# mvps predicted", np.count_nonzero(yHatNeigh))
        # print("# actual mvps",np.count_nonzero(yTest))

        # yHatNeigh = neigh.predict(currSeasonX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
        # print('NN: ')
        # print(currSeasonX[["player_name"]].iloc[yHatNeigh.nonzero()])

        logreg = LogisticRegression(penalty=LRpenalty,C = LRc, max_iter=1500)
        logreg.fit(xTrain[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']], yTrain)
        yHatLog = logreg.predict(xTest[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
        LRacc += accuracy_score(yTest, yHatLog)
        LRaccMVPS += accuracy_score(yTest[yTest.nonzero()],yHatLog[yTest.nonzero()])
        LRaccNON += accuracy_score(yTest[np.argwhere(yTest == 0)],yHatLog[np.argwhere(yTest == 0)])

        # print("# mvps predicted", np.count_nonzero(yHatLog))
        # print("# actual mvps",np.count_nonzero(yTest))

        # yHatLog = logreg.predict(currSeasonX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
        # print('LR: ')
        # print(currSeasonX[["player_name"]].iloc[yHatLog.nonzero()])


        clf = RandomForestClassifier(max_depth=7, max_features = 7, min_samples_leaf = 2, random_state=0) #,bootstrap = False
        clf.fit(xTrain[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']], yTrain)
        yHatRF = clf.predict(xTest[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
        RFacc += accuracy_score(yTest, yHatRF)
        RFaccMVPS += accuracy_score(yTest[yTest.nonzero()],yHatRF[yTest.nonzero()])
        RFaccNON += accuracy_score(yTest[np.argwhere(yTest == 0)],yHatRF[np.argwhere(yTest == 0)])

        # print("# mvps predicted", np.count_nonzero(yHatRF))
        # print("# actual mvps",np.count_nonzero(yTest))

        # yHatRF = clf.predict(currSeasonX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
        # print('RF: ')
        # print(currSeasonX[["player_name"]].iloc[yHatRF.nonzero()])
    
    print('Nearest Neighbors Acc: ', NNacc / ksplits)
    print('NN Acc (MVP): ', NNaccMVPS / ksplits)
    #print('NN Acc (non-MVP): ', NNaccNON / ksplits)
    #print('# Neigh: ', kNumNeigh)
    print("\n")

    print('logistic regression Acc: ', LRacc / ksplits)
    print('logistic regression Acc (MVP): ', LRaccMVPS / ksplits)
    #print('LR Acc (non-MVP): ', LRaccNON / ksplits)
    print("\n")

    print("Random Forest Acc: ", RFacc / ksplits)
    print('RF Acc (MVP): ', RFaccMVPS / ksplits)
    #print('RF Acc (non-MVP): ', RFaccNON / ksplits)
    print("\n")



    

    neigh = KNeighborsClassifier(n_neighbors=kNumNeigh,weights=Kweight)
    neigh.fit(pastSeasonsX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']], pastSeasonsY)
    yHatNeigh = neigh.predict(currSeasonX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
    print('Nearest Neighbors: ')
    print(currSeasonX[["player_name"]].iloc[yHatNeigh.nonzero()])
    print("\n")
    

    logreg = LogisticRegression(penalty = LRpenalty,C = LRc, max_iter=1000)
    logreg.fit(pastSeasonsX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']], pastSeasonsY)
    yHatLog = logreg.predict(currSeasonX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
    yHatLogP = logreg.predict_proba(currSeasonX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
    print('logistic regression: (p>50%)')
    print(currSeasonX[["player_name"]].iloc[yHatLog.nonzero()])
    print("\n")

    print('logistic regression: (top 5)')
    MVPprobs = pd.DataFrame(index=range(len(yHatLogP)),columns=['MVPp'])
    for i in range(len(yHatLogP)):
        MVPprobs.loc[i,'MVPp'] = yHatLogP[i][1]
    MVPprobs['player_name'] = currSeasonX['player_name']
    print(MVPprobs[["player_name",'MVPp']].loc[MVPprobs.sort_values(by=['MVPp'],ascending=False).head(5).index])
    print("\n")


    clf = RandomForestClassifier(max_depth=7, max_features = 7, min_samples_leaf = 2, random_state=0) #,bootstrap = False
    clf.fit(pastSeasonsX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']], pastSeasonsY)
    yHatRF = clf.predict(currSeasonX[['pts','reb','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct','gp_pct','win_pct']])
    print("random forest: ")
    print(currSeasonX[["player_name"]].iloc[yHatRF.nonzero()])
    print("\n")


if __name__ == "__main__":
    main()