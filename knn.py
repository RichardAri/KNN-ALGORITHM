from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

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

#* print(iris_test)

# Dividimos las características y las etiquetas para el conjunto de entrenamiento
y_train = iris_train.pop("target")
x_train = iris_train

# Dividimos las características y las etiquetas para el conjunto de prueba
y_test = iris_test.pop("target")
X_test = iris_test

#! Entrenamos el modelo con los datos de entrenamiento
model.fit(x_train, y_train)


#? prediccion sobre las 3 muestras

predictions = model.predict(X_test)
print("Predicciones:", predictions)


#? porcentaje de acierto con un vecino 

accuracy = model.score(X_test, y_test) #* Precision
print("Porcentaje de acierto con 1 vecino:", accuracy)

#? porcentaje de acierto para 3 vecinos

#! nuevo modelo con 3 vecinos (neighbors)
model_3_neighbors = KNeighborsClassifier(n_neighbors=3)
model_3_neighbors.fit(x_train, y_train) #! entrenamos el modelo

#! Calculamos el porcentaje de acierto con el nuevo modelo
accuracy_3_neighbors = model_3_neighbors.score(X_test, y_test)
print("Porcentaje de acierto con 3 vecinos:", accuracy_3_neighbors)
