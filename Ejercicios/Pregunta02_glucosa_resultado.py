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

# Lectura del archivo CSV
df = pd.read_csv('/content/drive/My Drive/persona.csv')

# Regresión logística
X = df[['Glucose']]  # Característica: Nivel de glucosa
y = df['Outcome']     # Variable de resultado: Resultado (0 o 1)

# Entrenamiento del modelo de regresión logística
model = LogisticRegression()
model.fit(X, y)

# Creación de los puntos de la línea de regresión logística
x_values = pd.DataFrame({'Glucose': range(0, 300, 10)})
y_values = model.predict_proba(x_values)[:, 1]

# Gráfico de dispersión entre "Glucose" y "Outcome"
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Glucose', y='Outcome', data=df)
plt.title('Relación entre el nivel de glucosa y el resultado')
plt.xlabel('Nivel de glucosa')
plt.ylabel('Resultado (0: No diabetes, 1: Diabetes)')

# Gráfico de la línea de regresión logística
plt.plot(x_values, y_values, color='red', linestyle='--', label='Regresión Logística')
plt.legend()

plt.show()
