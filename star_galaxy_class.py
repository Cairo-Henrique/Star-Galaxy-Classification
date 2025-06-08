import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import pandas as pd

# Caminho da pasta
galaxy = Path(r'C:\Users\Cairo Henrique\Downloads\archive\Cutout Files\galaxy')
star = Path(r'C:\Users\Cairo Henrique\Downloads\archive\Cutout Files\star')

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

show('IC3521-H01_1419_1705_6.jpg')