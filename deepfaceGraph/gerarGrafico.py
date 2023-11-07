import matplotlib.pyplot as plt

dados = open("/home/lorenzo/Python/TOYlab/deepFace/deepfaceGraph/gerarGrafico.py").readlines()
nomes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

for ng in range(1, 8):
    x = []
    y = []
    for i in range(len(dados)):
        if i != 0:
            linha = dados[i].split(",")
            x.append(int(linha[0]))
            y.append(float(linha[ng]))

    plt.plot(x, y, label=nomes[ng - 1])

plt.legend(loc=2, bbox_to_anchor=(1, 1))
plt.xlabel('Frames')
plt.ylabel('Expressões Faciais')
plt.show()
