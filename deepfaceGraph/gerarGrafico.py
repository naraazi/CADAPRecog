import matplotlib.pyplot as plt

dados = open(
    "C:/Users/loren/OneDrive/Documentos/DesenvolvimentoIA/CADAPRecog/deepfaceGraph/emotions_data.csv").readlines()
names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

for ng in range(1, 8):
    x = []
    y = []
    for i in range(len(dados)):
        if i != 0:
            line = dados[i].split(",")
            x.append(int(line[0]))
            y.append(float(line[ng]))

    plt.plot(x, y, label=names[ng - 1])

plt.legend(loc=2, bbox_to_anchor=(1, 1))
plt.xlabel('Frames')
plt.ylabel('Expressões Faciais')
plt.show()
