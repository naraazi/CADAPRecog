import matplotlib.pyplot as plt

with open("/home/lorenzo/Python/TOYlab/deepFace/deepfaceGraphSeconds/emotions_data_seconds.csv", "r") as file:
    lines = file.readlines()

nomes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# -- inicializa listas para armazenar os dados
dados_emocoes = [[] for _ in range(len(nomes))]

# -- inicializa uma lista para armazenar o tempo em segundos
tempo_segundos = []

# -- leitura dos dados
for linha in lines[1:]:
    valores = linha.strip().split(",")
    tempo = float(valores[0])
    emocoes = list(map(float, valores[1:]))

    for i, valor_emocao in enumerate(emocoes):
        dados_emocoes[i].append(valor_emocao)

    # -- armazena o tempo em segundos na lista
    tempo_segundos.append(tempo)

# -- gera o gráfico
for i, nome_emocao in enumerate(nomes):
    plt.plot(tempo_segundos, dados_emocoes[i], label=nome_emocao)

plt.legend(loc='upper right')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Expressões Faciais')
plt.title('Expressões Faciais ao Longo do Tempo')

plt.show()
