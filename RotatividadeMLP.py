import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn import preprocessing
import numpy as np
import time

data=pandas.read_csv('HR_comma_sep.csv')
#print(data.head())

le = preprocessing.LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data['Department']=le.fit_transform(data['Department'])

data.info()
dropout=data.groupby('left').count()
plt.bar(dropout.index.values, dropout['satisfaction_level'])
plt.xlabel('Employees Left Company')
plt.ylabel('Number of Employees')
#plt.show()

xset=data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Department', 'salary']]
yset=data['left']

#divisão do dataset entre treinamento e testes
xset_train, xset_test, yset_train, yset_test = train_test_split(xset, yset, test_size=0.3, random_state=42)

modelo = MLPClassifier(hidden_layer_sizes=(9,9), #tupla com núimero de camadas e neurons por camada
                    activation="relu", #função de ativação
                    max_iter=2000,
                    verbose=True,
                    learning_rate_init=0.01) #taxa de aprendizado

# Fit ou treinamento do modelo, com métrica de tempo
start_time = time.time()
modelo.fit(xset_train,yset_train)
print("Tempo de execução do treinamento: %s seconds" % round((time.time() - start_time),5))
ypred=modelo.predict(xset_test)

#Cálculo e exibição das métricas de performance de classificação
print("Acurácia: "+str(round(accuracy_score(yset_test,ypred),5)))
print("Precisão: "+str(round(precision_score(yset_test,ypred,average='macro'),5)))
print("Recall: "+str(round(recall_score(yset_test,ypred,average='macro'),5)))
print("F1 Score: " +str(round(f1_score(yset_test,ypred, average='macro'),5)))
print("Jaccard: " +str(round(jaccard_score(yset_test,ypred,average='macro'),5)))