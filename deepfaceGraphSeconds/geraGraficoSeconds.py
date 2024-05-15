import matplotlib.pyplot as plt

with open("C:/Users/loren/OneDrive/Documentos/DesenvolvimentoIA/CADAPRecog/deepfaceGraphSeconds/emotions_data_seconds"
          ".csv", "r") as file:
    lines = file.readlines()

nomes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

emotion_data = [[] for _ in range(len(nomes))]

tempo_segundos = []

for line in lines[1:]:
    values = line.strip().split(",")
    tempo = float(values[0])
    emotions = list(map(float, values[1:]))

    for i, emotion_value in enumerate(emotions):
        emotion_data[i].append(emotion_value)

    tempo_segundos.append(tempo)

for i, emotion_name in enumerate(nomes[:-1]):
    plt.plot(tempo_segundos, emotion_data[i], label=emotion_name)

plt.legend(loc='upper right')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Expressões Faciais')
plt.title('Expressões Faciais ao Longo do Tempo')
plt.savefig("C:/Users/loren/OneDrive/Documentos/DesenvolvimentoIA/CADAPRecog/deepfaceGraphSeconds")
