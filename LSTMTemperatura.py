import numpy as np
import pandas as pd
import time
import multiprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt

# Función de nivelación de cargas
def nivelacionCargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]  # Elementos sobrantes
    t = int((len(D) - s) / n_p)  # Número de elementos por grupo
    out = [[] for _ in range(n_p)]  # Inicializar listas vacías para cada lote

    # Distribuir los elementos restantes entre los lotes
    index = 0
    for i in D[s:]:
        out[index].append(i)
        index = (index + 1) % n_p  # Distribuir de forma balanceada entre los lotes

    # Asignar los elementos sobrantes (n_D) a los primeros lotes
    for i in range(len(n_D)):
        out[i].append(n_D[i])

    return out

# Clase LSTM simplificada
class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Pesos para las puertas y salidas
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def step(self, x, h_prev, c_prev):
        combined = np.vstack((h_prev, x))
        ft = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        it = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
        c_next = ft * c_prev + it * c_tilde
        ot = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        h_next = ot * np.tanh(c_next)
        y = np.dot(self.Wy, h_next) + self.by
        return y, h_next, c_next

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Función de entrenamiento y evaluación
def entrenarYEvaluarLSTM(hidden_size, X_train, y_train, X_test, y_test, sequence_length, epochs=10, lr=0.001):
    lstm = LSTM(input_size=1, hidden_size=hidden_size, output_size=1)
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))

    # Entrenamiento
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x_seq = X_train[i].reshape(-1, 1)
            y_true = y_train[i].reshape(-1, 1)

            # Forward pass
            for t in range(x_seq.shape[0]):
                y_pred, h_prev, c_prev = lstm.step(x_seq[t], h_prev, c_prev)

            # Backprop simplificada (actualización manual de pesos)
            lstm.Wy -= lr * (y_pred - y_true).T @ h_prev.T

    # Validación
    predictions = []
    for i in range(len(X_test)):
        x_seq = X_test[i].reshape(-1, 1)
        for t in range(x_seq.shape[0]):
            y_pred, h_prev, c_prev = lstm.step(x_seq[t], h_prev, c_prev)
        predictions.append(y_pred.item())

    rmse = sqrt(mean_squared_error(y_test, predictions))
    return {'hidden_size': hidden_size, 'rmse': rmse, 'predictions': predictions}

# Función para crear secuencias para series temporales
def create_sequences(data, sequence_length):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)

# Entrenamiento en lotes
def entrenar_en_lote(lote, X_train, y_train, X_test, y_test, sequence_length):
    resultados = []
    for hidden_size in lote:
        resultado = entrenarYEvaluarLSTM(hidden_size, X_train, y_train, X_test, y_test, sequence_length)
        resultados.append(resultado)
    return resultados

# Dividir datos en entrenamiento y prueba (debes adaptarlo según tu conjunto de datos)
data = pd.read_csv(r"C:\Users\david\Downloads\data.csv", low_memory=False)
columnasI = ['TMAX', 'TMIN', 'TAVG', 'PRCP']  # Selección de columnas relevantes
data = data[columnasI].dropna()

# Normalizar las columnas
def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

for column in columnasI:
    data[column] = normalize_column(data[column])

# División de datos
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

sequence_length = 50  # Tamaño de la secuencia
X_train_tmax, y_train_tmax = create_sequences(train['TMAX'].values, sequence_length)
X_test_tmax, y_test_tmax = create_sequences(test['TMAX'].values, sequence_length)

X_train_tmin, y_train_tmin = create_sequences(train['TMIN'].values, sequence_length)
X_test_tmin, y_test_tmin = create_sequences(test['TMIN'].values, sequence_length)

# Configuración y paralelismo
if __name__ == '__main__':
    hidden_sizes = [5,10,20]
    cores = multiprocessing.cpu_count()
    print(f"Usando {cores} núcleos para paralelismo.")

    start_time = time.time()

    lotesParametros = nivelacionCargas(hidden_sizes, cores)

    # Medir el tiempo antes de la paralelización
    parallel_start_time = time.time()

    # Entrenamiento paralelo para TMAX
    with multiprocessing.Pool(processes=cores) as pool:
        resultadosTMAX = pool.starmap(
            entrenar_en_lote,
            [(lote, X_train_tmax, y_train_tmax, X_test_tmax, y_test_tmax, sequence_length) for lote in lotesParametros]
        )

    # Aplanar los resultados de TMAX
    resultadosTMAX = [resultado for sublista in resultadosTMAX for resultado in sublista]

    # Medir el tiempo después de la paralelización para TMAX
    parallel_end_time = time.time()

    # Buscar el mejor modelo para TMAX
    mejorModeloTMAX = min(resultadosTMAX, key=lambda x: x['rmse'])
    print(f"Mejor modelo LSTM para TMAX: {mejorModeloTMAX}")

    # Entrenamiento paralelo para TMIN
    with multiprocessing.Pool(processes=cores) as pool:
        resultadosTMIN = pool.starmap(
            entrenar_en_lote,
            [(lote, X_train_tmin, y_train_tmin, X_test_tmin, y_test_tmin, sequence_length) for lote in lotesParametros]
        )

    # Aplanar los resultados de TMIN
    resultadosTMIN = [resultado for sublista in resultadosTMIN for resultado in sublista]

    # Buscar el mejor modelo para TMIN
    mejorModeloTMIN = min(resultadosTMIN, key=lambda x: x['rmse'])
    print(f"Mejor modelo LSTM para TMIN: {mejorModeloTMIN}")

    end_time = time.time()

    # Imprimir el tiempo total
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")
