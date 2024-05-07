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

# Creamos una instancia con 1 vecino
model = KNeighborsClassifier(n_neighbors=1)

# Cargamos el conjunto de datos Iris desde seaborn
iris = sns.load_dataset("iris")

# Codificamos las etiquetas de clase utilizando LabelEncoder
le_iris = LabelEncoder()
iris["target"] = le_iris.fit(iris.species).transform(iris.species)

# Eliminamos la columna "species" ya que no la necesitamos para el modelo
iris = iris.drop("species", axis=1)

# Seleccionamos aleatoriamente 3 muestras para el conjunto de prueba
iris_test = iris.sample(3, random_state=1112)

# Creamos el conjunto de entrenamiento excluyendo las muestras de prueba
iris_train = iris.drop(iris_test.index)

iris_test

#* -------------------------------------------------

# Dividimos las características y las etiquetas para el conjunto de entrenamiento
y_train = iris_train.pop("target")
x_train = iris_train

# Dividimos las características y las etiquetas para el conjunto de prueba
y_test = iris_test.pop("target")
X_test = iris_test

#! Entrenamos el modelo con los datos de entrenamiento
model.fit(x_train, y_train)


#? prediccion sobre las 3 muestras -------------------------

predictions = model.predict(X_test)
print("Predicciones:", predictions) 
#! Predicciones: [1, 2, 0]
#* significa que el modelo predijo las siguientes clases para las muestras de prueba:
#* la primera muestra se predijo como clase 1 => Clase 1: Iris-versicolor
#* la segunda muestra se predijo como clase 2 => Clase 2: Iris-virginica
#* la tercera muestra se predijo como clase 0 => Clase 0: Iris-setosa

#? porcentaje de acierto con un vecino -------------------------

accuracy = model.score(X_test, y_test) #* Precision
print("Porcentaje de acierto con 1 vecino:", accuracy)
#! Porcentaje de acierto con 1 vecino: 0.6666666666666666
#* es el porcentaje de muestras de prueba que el modelo clasifico correctamente 
#* cuando se usó 1 vecino, el modelo acerto en aproximadamente el 
#* 66.67% de las veces cuando solo consideró a un vecino más cercano para clasificar una muestra

#? porcentaje de acierto para 3 vecinos

# modelo con 3 vecinos (neighbors)
model_3_neighbors = KNeighborsClassifier(n_neighbors=3)
model_3_neighbors.fit(x_train, y_train) #! entrenamos el modelo

# porcentaje de precision con el nuevo modelo
accuracy_3_neighbors = model_3_neighbors.score(X_test, y_test)
print("Porcentaje de acierto con 3 vecinos:", accuracy_3_neighbors)
#! Porcentaje de acierto con 3 vecinos: 1.0 (o 100%
#* el modelo clasificó correctamente el 100% cuando se usaron 3 vecinos


