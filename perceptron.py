# Actividad Retroalimentación Módulo 2
# Alejandro Somarriba Aguirre
# A01751277

# Se importan las librerías

import pandas as pd
import numpy as np

# Función de activación (sign)
def sign(x):
    if (x > 0):
        return 1
    else:
        return -1

df = pd.read_csv("stroke.csv")
# df.head()

train = df.sample(frac = 0.8)
temp = pd.concat([df, train])
test = temp.drop_duplicates(keep=False)

# Separación de datos de entrenamiento y prueba
X_train = train.drop("stroke", axis=1)
Y_train = train["stroke"].replace(0, -1)

X_test = test.drop("stroke", axis=1)
Y_test = test["stroke"].replace(0, -1)

# Inicialización de parámetros al azar
wi = np.random.random(X_train.shape[1])
w0 = np.random.random()

# Hiperparámetros
eta = 0.1

Y_pred = np.empty(X_train.shape[0], dtype = np.int8)
epochs = 0

# Entrenamiento
print("Comenzando entrenamiento. Por favor esperar...")
print()

while (epochs < 50 or ((Y_pred == Y_train).sum()) == Y_train.shape[0]):

  for i in range(X_train.shape[0]):
    out = sign(sum(X_train.iloc[i] * wi) + w0)

    Y_pred[i] = out

    if (Y_train.iloc[i] != out):
      dwi = eta*(Y_train.iloc[i] - out)*X_train.iloc[i].values
      dw0 = eta*(Y_train.iloc[i] - out)
      wi = wi + dwi
      w0 = w0 + dw0

  epochs += 1

print("Entrenamiento terminado. Resultados Entrenamiento:")
print("Porcentaje de datos de entrenamiento clasificados correctamente: " + str(100*(Y_train == Y_pred).sum()/Y_train.shape[0]) + "%")

# Inicialización de variables para métricas
Y_test_pred = np.empty(X_test.shape[0], dtype = np.int8)

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

# Prueba/Predicción
for i in range(X_test.shape[0]):
  pred = sign(sum(X_test.iloc[i] * wi) + w0)

  if ((Y_test.iloc[i] == pred) and (pred == 1)):
    true_pos += 1
  elif ((Y_test.iloc[i] == pred) and (pred == -1)):
    true_neg += 1
  elif ((Y_test.iloc[i] == -1) and (pred == 1)):
    false_pos += 1
  elif ((Y_test.iloc[i] == 1) and (pred == -1)):
    false_neg += 1

  Y_test_pred[i] = pred

accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

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

print()
print("Resultados Pruebas/Predicciones:")
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("Specificity: " + str(specificity))
print()