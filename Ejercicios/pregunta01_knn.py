from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive #!cargar csv

#! configuracion csv
drive.mount('/content/drive')

# Lectura del archivo CSV
df = pd.read_csv('/content/drive/My Drive/persona.csv')

# Gráfico de dispersión entre "Pregnancies" y "Outcome"
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pregnancies', y='Outcome', data=df)
plt.title('Relación entre el número de embarazos y el resultado')
plt.xlabel('Número de embarazos')
plt.ylabel('Resultado (0: No diabetes, 1: Diabetes)')

# KNN
X = df[['Pregnancies']]  # Característica: Número de embarazos
y = df['Outcome']        # Variable de resultado: Resultado (0 o 1)

# Entrenamiento del modelo KNN
model = KNeighborsClassifier(n_neighbors=5)  # K=5 vecinos más cercanos
model.fit(X, y)

# Creación de los puntos de la línea de decisión del modelo KNN
x_values = pd.DataFrame({'Pregnancies': range(0, 18)})
y_values = model.predict(x_values)

# Gráfico de la línea de decisión del modelo KNN
plt.plot(x_values, y_values, color='red', linestyle='--', label='KNN Decision Boundary')
plt.legend()

plt.show()
