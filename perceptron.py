# Actividad Retroalimentación Módulo 2
# Alejandro Somarriba Aguirre
# A01751277

# Se importan las librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Función de activación (sign)
def sign(x):
    if (x > 0):
        return 1
    else:
        return -1

# Se leen los datos del archivo CSV
df = pd.read_csv("BCds.csv")

# Se toma una muestra del 80% de los datos para luego formar conjuntos de entrenamiento y prueba
train = df.sample(frac = 0.8)
temp = pd.concat([df, train])
test = temp.drop_duplicates(keep=False)

# Separación de datos de entrada y salida para los conjuntos de entrenamiento y prueba
X_train = train.drop("Class", axis=1)
Y_train = train["Class"].replace(0, -1)

X_test = test.drop("Class", axis=1)
Y_test = test["Class"].replace(0, -1)

# Inicialización de parámetros al azar usando el módulo random de la librería Numpy
wi = np.random.random(X_train.shape[1])
w0 = np.random.random()

# Hiperparámetros
# Se escoge un valor de tasa de aprendizaje bajo
eta = 0.1

# Se inicializa un arreglo vacío que contendrá los valores que predice el modelo
Y_pred = np.empty(X_train.shape[0], dtype = np.int8)

# Se inicia un contador para manejar el tiempo que se entrena el modelo
epochs = 0

# Entrenamiento
print("Comenzando entrenamiento. Por favor esperar...")
print()

# Se inicializa un arreglo vacío que guardará los valores de accuracy a lo largo del entrenamiento
train_acc = []

# Se inicia un ciclo que corre por 50 iteraciones o hasta que todas las predicciones de entrenamiento sean correctas
while (epochs < 50 or ((Y_pred == Y_train).sum()) == Y_train.shape[0]):

    # Por cada observación (fila) de los datos, se hace la suma del producto de las variables de entrada con los pesos iniciales
    # El resultado de la suma se pasa por la función de activación y se guarda en la variable out
    for i in range(X_train.shape[0]):
        out = sign(sum(X_train.iloc[i] * wi) + w0)

        # Se guarda el valor de cada fila en un arreglo
        Y_pred[i] = out

        # Si la predicción es incorrecta, se calculan nuevos valores para los pesos a partir del error en la predicción
        if (Y_train.iloc[i] != out):
            dwi = eta*(Y_train.iloc[i] - out)*X_train.iloc[i].values
            dw0 = eta*(Y_train.iloc[i] - out)
            wi = wi + dwi
            w0 = w0 + dw0

    # Cada iteración, el contador aumenta por 1 para evitar un ciclo infinito
    epochs += 1
    # Se guarda el cálculo de Accuracy en el arreglo creado anteriormente
    train_acc.append((Y_train == Y_pred).sum()/Y_train.shape[0])
  
    # Finalmente, se imprimen los resultados del entrenamiento por cada iteración
    print("Epoch " + str(epochs))
    print((Y_train == Y_pred).sum()/Y_train.shape[0])
    print()

print("Entrenamiento terminado. Resultados Entrenamiento:")
print("Porcentaje de datos de entrenamiento clasificados correctamente: " + str(100*(Y_train == Y_pred).sum()/Y_train.shape[0]) + "%")

# Adicionalmente, se muestra una gráfica en la que es más sencillo observar el progreso del entrenamiento
plt.plot(train_acc)
plt.title("Accuracy durante el entrenamiento")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Inicialización de variables para métricas de evaluación
# Se inicializa un arreglo vacío para almacenar los resultados de las predicciones
Y_test_pred = np.empty(X_test.shape[0], dtype = np.int8)

# Se crean variables para almacenar la cantidad de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

# Prueba/Predicción
# Por cada observación (fila) del conjunto de datos de prueba, se usan los valores de los pesos para hacer predicciones
for i in range(X_test.shape[0]):
    pred = sign(sum(X_test.iloc[i] * wi) + w0)

    # Dependiendo de si la predicción fue correcta o incorrecta, aumentan los contadores inicializados anteriormente
    if ((Y_test.iloc[i] == pred) and (pred == 1)):
        true_pos += 1
    elif ((Y_test.iloc[i] == pred) and (pred == -1)):
        true_neg += 1
    elif ((Y_test.iloc[i] == -1) and (pred == 1)):
        false_pos += 1
    elif ((Y_test.iloc[i] == 1) and (pred == -1)):
        false_neg += 1

    # Se añaden los resultados de la predicción al arreglo
    Y_test_pred[i] = pred

# Se hace el cálculo de accuracy con los valores de las predicciones
accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

# Se hacen los cálculos para las otras métricas (precision, recall, specificity)
# Si llega a ocurrir que en una división el denominador es 0, se asigna un 0 a la métrica
if (true_pos + false_pos == 0):
    precision = 0
else:
    precision = (true_pos / (true_pos + false_pos))

if (true_pos + false_pos == 0):
    recall = 0
else:
    recall = (true_pos / (true_pos + false_neg))

if (true_neg + false_pos == 0):
    specificity = 0
else:
    specificity = (true_neg / (true_neg + false_pos))

# Por último, se imprimen los resultados de las pruebas
print()
print("Resultados Pruebas/Predicciones:")
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("Specificity: " + str(specificity))
print()
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_test_pred))