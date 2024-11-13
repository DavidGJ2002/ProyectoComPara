import itertools
import multiprocess
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Método para hacer nivelación de cargas
def nivelacionCargas(datos, numProcesos):
    sobrante = len(datos) % numProcesos
    datosNivelados = datos[:sobrante]
    tamanioLote = int((len(datos) - sobrante) / numProcesos)
    salida = []
    temporal = []
    for dato in datos[sobrante:]:
        temporal.append(dato)
        if len(temporal) == tamanioLote:
            salida.append(temporal)
            temporal = []
    for i in range(len(datosNivelados)):
        salida[i].append(datosNivelados[i])
    return salida

# Definir modelo LSTM
class ModeloLSTM(nn.Module):
    def __init__(self, tamEntrada, tamOculto, tamSalida, numCapas=1):
        super(ModeloLSTM, self).__init__()
        self.lstm = nn.LSTM(tamEntrada, tamOculto, numCapas, batch_first=True)
        self.fc = nn.Linear(tamOculto, tamSalida)

    def forward(self, x):
        salida, _ = self.lstm(x)
        salida = self.fc(salida[:, -1, :])
        return salida

data = pd.read_csv(r"C:\Users\david\OneDrive\Documentos\InteligenciaArtificial\IPN\ComputoParalelo\MXM00076683.csv",low_memory=False)
# Selección de las columnas relevantes
columns_to_keep = ['LATITUDE', 'LONGITUDE', 'TMAX', 'TMIN', 'TAVG', 'PRCP']  # Utiliza columnas normalizadas
data = data[columns_to_keep]

# Convertir las columnas seleccionadas a tensores
X = data[['LATITUDE', 'LONGITUDE', 'TMAX', 'TMIN', 'TAVG', 'PRCP']].values  # Características
y_max = data['TMAX'].values  # Etiqueta de temperatura máxima
y_min = data['TMIN'].values  # Etiqueta de temperatura mínima

# Organizar las características para LSTM (3D: [muestras, pasos de tiempo, características])
X = X.reshape(X.shape[0], 1, X.shape[1])  # Pasos de tiempo = 1 para simplificar

# Dividir en conjuntos de entrenamiento y validación
XEntrenamiento, XValidacion, yEntrenamientoMax, yValidacionMax = train_test_split(X, y_max, test_size=0.2)
XEntrenamiento, XValidacion, yEntrenamientoMin, yValidacionMin = train_test_split(X, y_min, test_size=0.2)

# Convertir a tensores de PyTorch
XEntrenamiento = torch.tensor(XEntrenamiento, dtype=torch.float32)
yEntrenamientoMax = torch.tensor(yEntrenamientoMax, dtype=torch.float32).view(-1, 1)
yEntrenamientoMin = torch.tensor(yEntrenamientoMin, dtype=torch.float32).view(-1, 1)
XValidacion = torch.tensor(XValidacion, dtype=torch.float32)
yValidacionMax = torch.tensor(yValidacionMax, dtype=torch.float32).view(-1, 1)
yValidacionMin = torch.tensor(yValidacionMin, dtype=torch.float32).view(-1, 1)

# Definir combinaciones de hiperparámetros
rejillaParametros = {
    'tamOculto': [64, 128],
    'tasaAprendizaje': [1e-2, 1e-3],
    'numCapas': [1, 2]
}
combinacionesParametros = list(itertools.product(rejillaParametros['tamOculto'], rejillaParametros['tasaAprendizaje'], rejillaParametros['numCapas']))

# Función para entrenar y evaluar el modelo
def entrenarYEvaluar(parametros):
    tamOculto, tasaAprendizaje, numCapas = parametros
    modelo = ModeloLSTM(tamEntrada=6, tamOculto=tamOculto, tamSalida=1, numCapas=numCapas)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=tasaAprendizaje)
    
    # Entrenamiento para la temperatura máxima
    for epoca in range(10):  # Número de épocas reducido para ejemplo
        modelo.train()
        salidasMax = modelo(XEntrenamiento)
        perdidaMax = criterio(salidasMax, yEntrenamientoMax)
        
        optimizador.zero_grad()
        perdidaMax.backward()
        optimizador.step()

    # Evaluación para la temperatura máxima
    modelo.eval()
    with torch.no_grad():
        salidasMaxValidacion = modelo(XValidacion)
        perdidaMaxValidacion = mean_squared_error(yValidacionMax.numpy(), salidasMaxValidacion.numpy())
    
    # Entrenamiento para la temperatura mínima
    for epoca in range(10):  # Número de épocas reducido para ejemplo
        modelo.train()
        salidasMin = modelo(XEntrenamiento)
        perdidaMin = criterio(salidasMin, yEntrenamientoMin)
        
        optimizador.zero_grad()
        perdidaMin.backward()
        optimizador.step()

    # Evaluación para la temperatura mínima
    with torch.no_grad():
        salidasMinValidacion = modelo(XValidacion)
        perdidaMinValidacion = mean_squared_error(yValidacionMin.numpy(), salidasMinValidacion.numpy())
    
    return {'parametros': parametros, 'perdidaMaxValidacion': perdidaMaxValidacion, 'perdidaMinValidacion': perdidaMinValidacion}

# Ejecutar en paralelo con nivelación de cargas
if __name__ == '__main__':
    cores = multiprocess.cpu_count()
    print(cores)
    tiempoInicio = time.time()
    lotesParametros = nivelacionCargas(combinacionesParametros, 7)
    
    with multiprocess.Pool(processes=multiprocess.cpu_count()) as pool:
        resultados = pool.map(lambda lote: [entrenarYEvaluar(param) for param in lote], lotesParametros)
    
    # Aplanar la lista de resultados
    resultados = [item for sublista in resultados for item in sublista]
    
    tiempoFin = time.time()

    # Recolectar y mostrar los mejores resultados
    mejoresParametros = min(resultados, key=lambda x: x['perdidaMaxValidacion'] + x['perdidaMinValidacion'])
    print(f"Mejores hiperparámetros encontrados: {mejoresParametros}")
    print(f"Tiempo total de optimización: {tiempoFin - tiempoInicio:.2f} segundos")