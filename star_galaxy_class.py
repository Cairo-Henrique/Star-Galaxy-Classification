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
        image_path = Path(r'C:\Users\Cairo Henrique\Downloads\archive\Cutout Files\galaxy') / nome_imagem
    elif nome_imagem in nomes_star:
        image_path = Path(r'C:\Users\Cairo Henrique\Downloads\archive\Cutout Files\star') / nome_imagem
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

# Separar features (x) e rótulos (y)
X = df['nome_imagem'] 
y = df['classe']

# Dividir os dados (exemplo com 80% treino e 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Criar arrays
x = np.array(df['nome_imagem'])
y = np.array(df['classe'])

# Ver quantas imagens há em cada conjunto
print(f'Treinamento: {len(X_train)} imagens')
print(f'Teste: {len(X_test)} imagens')
