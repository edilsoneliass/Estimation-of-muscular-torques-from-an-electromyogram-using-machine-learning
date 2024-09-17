import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tamanho do gráfico em polegadas
plt.figure(figsize =(11, 6))

sns.set_style("whitegrid")

#Plotando o boxplot das espécies em relação ao tamanho das sépalas
ax = sns.boxplot( x = "species", y ="sepal_length",data = base_dados,
                  hue = "species",linewidth=5, palette = "Set3")

# Adicionando Título ao gráfico
plt.title("Boxplot da base de dados Íris", loc="center", fontsize=18)
plt.xlabel("Tipos de Flores")
plt.ylabel("Comprimento das sépalas")

plt.show()