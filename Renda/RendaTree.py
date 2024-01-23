import pandas
import matplotlib.pyplot as plt
import sklearn
import time
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.tree import export_text

data=pandas.read_csv("adult.csv", sep=";")
le = preprocessing.LabelEncoder()
data['workclass']=le.fit_transform(data['workclass'])
data['education']=le.fit_transform(data['education'])
data['marital-status']=le.fit_transform(data['marital-status'])
data['occupation']=le.fit_transform(data['occupation'])
data['relationship']=le.fit_transform(data['relationship'])
data['race']=le.fit_transform(data['race'])
data['sex']=le.fit_transform(data['sex'])
data['native-country']=le.fit_transform(data['native-country'])
data.info()

xset=data[["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]]
yset=data["income"]

#divisão do dataset entre treinamento e testes
xset_train, xset_test, yset_train, yset_test = train_test_split(xset, yset, test_size=0.3, random_state=42)

modeloarvore = tree.DecisionTreeClassifier()
# Fit ou treinamento do modelo, com métrica de tempo
start_time = time.time()
modeloarvore = modeloarvore.fit(xset_train,yset_train)
print("Tempo de execução do treinamento: %s seconds" % round((time.time() - start_time),5))
tree.plot_tree(modeloarvore)
#plt.show() #para plotagem da árvore em tempo de execução, é somente necessário descomentar esta linha
ypred=modeloarvore.predict(xset_test)

#descomentar as linhas abaixo para salvar a árvore no formato vetorial eps
#plt.figure()
#tree.plot_tree(modeloarvore,filled=True)  
#plt.savefig('treestudents.eps',format='eps',bbox_inches = "tight")

#descomentar as linhas abaixo para imprimir a árvore na linha de comando
#r = export_text(modeloarvore)
#print(r)

#Cálculo e exibição das métricas de performance de classificação
print("Acurácia: "+str(round(accuracy_score(yset_test,ypred),5)))
print("Precisão: "+str(round(precision_score(yset_test,ypred,average='macro'),5)))
print("Recall: "+str(round(recall_score(yset_test,ypred,average='macro'),5)))
print("F1 Score: " +str(round(f1_score(yset_test,ypred, average='macro'),5)))
print("Jaccard: " +str(round(jaccard_score(yset_test,ypred,average='macro'),5)))
