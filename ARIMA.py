import itertools
import multiprocessing
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Método para hacer nivelación de cargas
def nivelacionCargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

# Cargar y preparar los datos
data = pd.read_csv(r"data.csv", low_memory=False)

# Selección de las columnas relevantes
columnasI = ['TMAX', 'TMIN', 'TAVG', 'PRCP']  # Utiliza columnas normalizadas
data = data[columnasI]

# Normalizar datos
def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

data = data.dropna()  # Eliminar filas con NaN

for column in ['TMAX', 'TMIN', 'TAVG', 'PRCP']:
    data[column] = normalize_column(data[column])

# División temporal (no aleatoria) para series temporales
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

XEntrenamientoTMAX = train['TMAX'].values
XValidacionTMAX = test['TMAX'].values
yEntrenamientoTMAX = XEntrenamientoTMAX
yValidacionTMAX = XValidacionTMAX

XEntrenamientoTMIN = train['TMIN'].values
XValidacionTMIN = test['TMIN'].values
yEntrenamientoTMIN = XEntrenamientoTMIN
yValidacionTMIN = XValidacionTMIN

# Función para entrenar y evaluar un modelo ARIMA
def entrenarYEvaluar(parametros, serie, validacion):
    p, d, q = parametros
    try:
        # Entrenar modelo ARIMA
        modelo = ARIMA(serie, order=(p, d, q))
        resultados = modelo.fit()

        # Hacer predicciones
        predicciones = resultados.forecast(steps=len(validacion))

        # Calcular pérdida
        perdida = mean_squared_error(validacion, predicciones)
        rmse = sqrt(perdida)

        return {'parametros': parametros, 'rmse': rmse, 'predicciones': predicciones}
    except Exception as e:
        return {'parametros': parametros, 'rmse': float('inf'), 'error': str(e)}

# Función para ejecutar el entrenamiento en lotes para TMAX
def entrenar_en_lote_tmax(lote):
    return [entrenarYEvaluar(param, XEntrenamientoTMAX, yValidacionTMAX) for param in lote]

# Función para ejecutar el entrenamiento en lotes para TMIN
def entrenar_en_lote_tmin(lote):
    return [entrenarYEvaluar(param, XEntrenamientoTMIN, yValidacionTMIN) for param in lote]

# Definir combinaciones de hiperparámetros para ARIMA
parametrosARIMA = {
    'p': [0, 1, 2],
    'd': [0, 1],
    'q': [0, 1, 2]
}
combinacionesParametros = list(itertools.product(parametrosARIMA['p'], parametrosARIMA['d'], parametrosARIMA['q']))

# Ejecutar en paralelo con nivelación de cargas
if _name_ == '_main_':
    cores = multiprocessing.cpu_count()
    print(f"Número de núcleos disponibles: {cores}")
    tiempoInicio = time.time()

    # Dividir los parámetros en lotes y hacer predicciones para TMAX
    lotesParametrosTMAX = nivelacionCargas(combinacionesParametros, cores)
    with multiprocessing.Pool(processes=cores) as pool:
        resultadosTMAX = pool.map(entrenar_en_lote_tmax, lotesParametrosTMAX)

    # Aplanar la lista de resultados para TMAX
    resultadosTMAX = [item for sublista in resultadosTMAX for item in sublista]

    # Dividir los parámetros en lotes y hacer predicciones para TMIN
    lotesParametrosTMIN = nivelacionCargas(combinacionesParametros, cores)
    with multiprocessing.Pool(processes=cores) as pool:
        resultadosTMIN = pool.map(entrenar_en_lote_tmin, lotesParametrosTMIN)

    # Aplanar la lista de resultados para TMIN
    resultadosTMIN = [item for sublista in resultadosTMIN for item in sublista]

    tiempoFin = time.time()

    # Recolectar y mostrar los mejores resultados para TMAX
    mejoresParametrosTMAX = min(resultadosTMAX, key=lambda x: x['rmse'])
    print(f"Mejores hiperparámetros encontrados para TMAX: {mejoresParametrosTMAX}")

    # Recolectar y mostrar los mejores resultados para TMIN
    mejoresParametrosTMIN = min(resultadosTMIN, key=lambda x: x['rmse'])
    print(f"Mejores hiperparámetros encontrados para TMIN: {mejoresParametrosTMIN}")

    print(f"Tiempo total de optimización: {tiempoFin - tiempoInicio:.2f} segundos")

    # Hacer las predicciones finales para TMAX y TMIN
    prediccionesTMAX = mejoresParametrosTMAX['predicciones']
    prediccionesTMIN = mejoresParametrosTMIN['predicciones']

    # Mostrar las predicciones
    print("Predicciones de TMAX:", prediccionesTMAX)
    print("Predicciones de TMIN:", prediccionesTMIN)