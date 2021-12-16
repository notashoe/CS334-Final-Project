import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

k = [1,2,3,5,7,9]

overallAcc = [0.9972649572649572,0.997948717948718,0.9976923076923077,0.9978632478632479,0.997948717948718,0.9978632478632479]

mvpAcc = [0.2833333333333333,0.09523809523809523,0.09047619047619046,0.08214285714285714,0.16666666666666666,0.0]

#nonAcc = [0.9988867661133586,0.9999142734676383,0.9995718443494033,0.9997429671321101,0.9998287671232877,1.0]

kW = [1,2,3,5,7,9,10]

overallAccW = [0.9976068376068377,0.9976068376068377,0.9974358974358974,0.9976923076923077,0.9977777777777778,0.9977777777777778,0.9978632478632479]

mvpAccW = [0.38,0.32047619047619047,0.12857142857142856,0.12380952380952381,0.12380952380952381,0.2,0.025]

#nonAccW = [0.9989721621758321,0.9990584016006878,0.9991440904930373,0.9995717342553977,0.9996574607878402,0.9998288402683114,0.9999142367066896]

df = pd.DataFrame(
    {'# of Nearest Neighbors': k,
     'Overall Accuracy Unweighted': overallAcc,
     'MVP Accuracy Unweighted': mvpAcc,
     #'non-MVP Accuracy' : nonAcc
    })

dfW = pd.DataFrame(
    {'# of Nearest Neighbors': kW,
     'Overall Accuracy Weighted': overallAccW,
     'MVP Accuracy Weighted': mvpAccW,
     #'non-MVP Accuracy' : nonAccW
    })

sns.lineplot(x='# of Nearest Neighbors',y='Overall Accuracy Unweighted',data = df,color = 'blue')
sns.lineplot(x='# of Nearest Neighbors',y='Overall Accuracy Weighted',data = dfW,color = 'red')
plt.title('KNN Overall Accuracy')
plt.legend(('Overall Accuracy (Distances Unweighted)','Overall Accuracy (Distances Weighted)',))

plt.show()

sns.lineplot(x='# of Nearest Neighbors',y='MVP Accuracy Unweighted',data = df,color = 'blue')
sns.lineplot(x='# of Nearest Neighbors',y='MVP Accuracy Weighted',data = dfW,color = 'red')
plt.title('KNN MVP Accuracy')
plt.legend(('MVP Accuracy (Distances Unweighted)','MVP Accuracy (Distances Weighted)',))
plt.show()
