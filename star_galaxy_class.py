import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Caminho para a pasta raiz do projeto (onde o script está)
base_dir = Path(__file__).resolve().parent

# Caminhos para as imagens, relativos à pasta do scstar.
galaxy = base_dir / 'archive' / 'Cutout Files' / 'galaxy'
star = base_dir / 'archive' / 'Cutout Files' / 'star'

# Lista apenas os arquivos (não diretórios)
nomes_galaxy = [f.name for f in galaxy.iterdir() if f.is_file()]
coluna_class = [1] * len(nomes_galaxy)

nomes_star = [f.name for f in star.iterdir() if f.is_file()]
coluna_class += [0] * len(nomes_star)

# Cria o DataFrame
df = pd.DataFrame(nomes_galaxy + nomes_star, columns=['nome_imagem'])
df['classe'] = coluna_class

# Exibir as primeiras linhas do DataFrame
print(df.head())

# Salvar o DataFrame em um arquivo CSV
df.to_csv('star_galaxy_class.csv', index=False)

def show(nome_imagem):
    # Caminho para a imagem
    if nome_imagem in nomes_galaxy:
        image_path = base_dir / 'archive' / 'Cutout Files' / 'galaxy' / nome_imagem
    elif nome_imagem in nomes_star:
        image_path = base_dir / 'archive' / 'Cutout Files' / 'star' / nome_imagem
    else:
        print(f"Imagem {nome_imagem} não encontrada.")
        return

    # Lê a imagem
    img = mpimg.imread(image_path)

    # Exibe a imagem
    plt.imshow(img)
    plt.axis('off')  # Esconde os eixos
    plt.title(nome_imagem)
    plt.show()

#show('IC3521-H01_1419_1705_6.jpg')

# Criar arrays
X = np.array(df['nome_imagem'])
y = np.array(df['classe'])

# Dividir os dados (exemplo com 80% treino e 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_copy = X_test # Manter uma cópia dos nomes das imagens de teste

# Ver quantas imagens há em cada conjunto
print(f'Treinamento: {len(X_train)} imagens')
print(f'Teste: {len(X_test)} imagens')

# Converter imagens para vetores de 64 pixels
from PIL import Image

# Função para vetorizar a imagem
def vetorizar(nome_imagem):
    if nome_imagem in nomes_galaxy:
        caminho = galaxy / nome_imagem
    else:
        caminho = star / nome_imagem
    img = Image.open(caminho).convert('L')  # tons de cinza
    img = img.resize((8, 8))
    return np.array(img).flatten() / 255.0

# Vetorizar as imagens de treino e teste
X_train = np.array([vetorizar(nome) for nome in X_train])
X_test = np.array([vetorizar(nome) for nome in X_test])

# Treinar modelos de classificação
from sklearn.svm import SVC                      # Support Vector Machine (SVM)
from sklearn.neural_network import MLPClassifier  # Multi-Layer Perceptron
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.metrics import accuracy_score, classification_report

modelos = [SVC(), MLPClassifier(), RandomForestClassifier()]

for modelo in modelos:
    print(f"Treinando modelo: {modelo.__class__.__name__}")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    # Métricas
    print("Acurácia:", accuracy_score(y_test, y_pred)*100, "%")
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Salvar o modelo treinado

import joblib

def salvar_modelo(modelo, nome_arquivo):
    caminho = base_dir / nome_arquivo
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em: {caminho}")

for modelo in modelos:
    nome_arquivo = f"{modelo.__class__.__name__.lower()}.joblib"
    salvar_modelo(modelo, nome_arquivo)

# Carregar o modelo salvo
def carregar_modelo(nome_arquivo):
    caminho = base_dir / nome_arquivo
    modelo = joblib.load(caminho)
    print(f"Modelo carregado de: {caminho}")
    return modelo

# Exemplo de uso do SVC carregado
modelo_carregado = carregar_modelo('svc.joblib')
# Exibir a classe da imagem de teste
print("Classe da imagem de teste:", y_test[0])
# Exibir a previsão do modelo carregado
y_pred = modelo_carregado.predict([vetorizar(X_test_copy[0])])
print("Previsão do modelo carregado:", y_pred[0])
# Exibir uma imagem de teste
show(X_test_copy[0])
