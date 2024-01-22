import random
import numpy
import matplotlib.pyplot as plt

bias = 1
aprendizado = 0.2
pesos = [random.random(),random.random(),random.random()]
print("Pesos iniciais aleatorios: " + str(pesos))
print("Taxa de aprendizado: " + str(aprendizado))
print("Bias: " + str(bias))

conjuntoand=numpy.array([[0,0,1,1],[0,1,0,1]])
valorand=numpy.array([0,0,0,1])
conjuntoor=numpy.array([[0,0,1,1],[0,1,0,1]])
valoror=numpy.array([0,1,1,1])
conjuntoxor=numpy.array([[0,0,1,1],[0,1,0,1]])
valorxor=numpy.array([0,1,1,0])
conjuntodados=numpy.array([[2,1,2,5,7,2,3,6,1,2,5,4,6,5],[2,3,3,3,3,4,4,4,5,5,5,6,6,7]])
valordados = numpy.array([0,0,0,1,1,0,0,1,0,0,1,1,1,1])

mapadecor=numpy.array(['r','b'])


def Perceptron(x, y, saidaEsperada):
	saidaPercep = x*pesos[0]+y*pesos[1]+bias*pesos[2]
	print("Resultado antes da função de ativação: " + str(saidaPercep))

	#função de ativação simples
	if saidaPercep > 0:
		saidaPercep = 1
	else:
		saidaPercep = 0
	#função de ativação sigmoide
	#saidaPercep = 1/(1+numpy.exp(-saidaPercep))

	#erro
	erro = saidaEsperada - saidaPercep
	print("Resultado do perceptron: " + str(saidaPercep))
	print("Erro: " + str(erro))

	#atualização dos parametros
	pesos[0] = pesos[0] + erro * x * aprendizado
	print("Peso x: " + str(pesos[0]))
	pesos[1] = pesos[1] + erro * y * aprendizado
	print("Peso y: " + str(pesos[1]))
	pesos[2] = pesos[2] + erro * bias * aprendizado
	print("Peso bias: " + str(pesos[2]))



conjunto = ()
valor = ()
escolha = int(input("Escolha qual base usar (1=AND, 2=OR, 3=XOR(Falha em perceptron de uma camada), 4=Dados predefinidos): "))
if escolha == 1:
	conjunto = conjuntoand
	valor = valorand
elif escolha == 2:
	conjunto = conjuntoor
	valor = valoror
elif escolha == 3:
	conjunto = conjuntoxor
	valor = valorxor
elif escolha == 4:
	conjunto = conjuntodados
	valor = valordados
else:
	print("Valor inválido, encerrando.")
	exit()

#gera o gráfico inicial sem separação baseado nos valores da variavel conjunto, feche o gráfico para continuar execução
plt.scatter(conjunto[0],conjunto[1],c=mapadecor[valor],s=40)
ax = plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()
print(xlim)
plt.show()


loop = int(input("Insira número de iterações do loop:"))
for i in range(loop):
	print("Iteração: " + str(i))
	for j in range(len(conjunto[0])):
		Perceptron(conjunto[0][j],conjunto[1][j],valor[j])
	print("\n")

print("Iterações finalizadas")
while True:
	print("Insira valores para teste:")
	x = float(input("x: "))
	y = float(input("y: "))

	print("Peso x atual: " + str(pesos[0]))
	print("Peso y atual: " + str(pesos[1]))
	print("Peso bias atual: " + str(pesos[2]))
	saidaTeste = x * pesos[0] + y*pesos[1] + bias*pesos[2]
	print("Saida antes da função de ativação: " + str(saidaTeste))
	if saidaTeste > 0:
		saidaTeste = 1
	else:
		saidaTeste = 0
	print("Resultado perceptron: " + str(saidaTeste))
	print("Representado no gráfico com um X da cor do seu conjunto.")
	print("\n")

	plt.scatter(conjunto[0],conjunto[1],c=mapadecor[valor],s=40)
	plt.scatter(x,y,c=mapadecor[saidaTeste],s=80,marker='X')
	slope = -(pesos[2]/pesos[1])/(pesos[2]/pesos[0])
	intercept = -pesos[2]/pesos[1]
	ax = plt.gca()
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	plt.axline(xy1=(0, intercept), slope=slope)

	plt.show()



