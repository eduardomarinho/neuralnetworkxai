import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

data=pandas.read_csv("data.csv", sep=';')
data.info()

xset=data[["Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance	", "Previous qualification", "Previous qualification (grade)", "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", "Admission grade", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", "Unemployment rate", "Inflation rate", "GDP"]]
yset=data['Target']

#divisão do dataset entre treinamento e testes
xset_train, xset_test, yset_train, yset_test = train_test_split(xset, yset, test_size=0.3, random_state=42)

modelo = MLPClassifier(hidden_layer_sizes=(35,35), #tupla com número de camadas e neurons por camada
                    activation='relu', #função de ativação
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