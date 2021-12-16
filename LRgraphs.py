import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

c = [0,1,2,5,7,10,25,50,100]

overallAcc = [0.997948717948718,0.9977777777777778,0.997948717948718,0.997948717948718,0.9981196581196581,0.9981196581196581,0.9983760683760684,0.9983760683760684,0.9982905982905983]

mvpAcc = [0.4485714285714285,0.14523809523809522,0.15555555555555553,0.12380952380952381,0.2976190476190476,0.4038095238095238,0.399047619047619,0.445,0.43571428571428567]


df = pd.DataFrame(
    {'Inverse of Regularization Strength': c,
     'Overall Accuracy': overallAcc,
     'MVP Accuracy': mvpAcc,
     #'non-MVP Accuracy' : nonAcc
    })



sns.lineplot(x='Inverse of Regularization Strength',y='Overall Accuracy',data = df,color = 'blue')
plt.title('Logistic Regression Overall Accuracy')
plt.legend(('Overall Accuracy',))

plt.show()

sns.lineplot(x='Inverse of Regularization Strength',y='MVP Accuracy',data = df,color = 'blue')
plt.title('Logistic Regression MVP Accuracy')
plt.legend(('MVP Accuracy',))
plt.show()
