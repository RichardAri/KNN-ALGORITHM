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

# Regresión logística
X = df[['Glucose']]  # Variable independiente: Nivel de glucosa
y = df['Outcome']    # Variable dependiente: Resultado (0 o 1)

# Creación y entrenamiento del modelo de regresión logística
model = LogisticRegression()
model.fit(X, y)

# Función logística
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Valores de x para la función logística
x_values = np.linspace(df['Glucose'].min(), df['Glucose'].max(), 1000)
# Predicción de las probabilidades usando el modelo de regresión logística
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]

# Gráfico de la función logística
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Datos')
plt.plot(x_values, y_values, color='red', linestyle='--', label='Regresión Logística')
plt.title('Relación entre el nivel de glucosa y el resultado de diabetes')
plt.xlabel('Nivel de glucosa')
plt.ylabel('Probabilidad de tener diabetes')
plt.legend()
plt.grid(True)
plt.show()
