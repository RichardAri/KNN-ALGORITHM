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
df = pd.read_csv('/content/drive/My Drive/persona.csv') # persona = IRIS

# Definición de características (X) y variable objetivo (y)
X = df.drop('Outcome', axis=1)  # Características: todas las columnas excepto 'Outcome'
y = df['Outcome']                # Variable objetivo: 'Outcome'

# División del conjunto de datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo KNN
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión del modelo KNN: {accuracy:.2f}")
#! Precisión del modelo KNN: 0.66
