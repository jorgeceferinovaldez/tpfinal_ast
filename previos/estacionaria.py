import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal

# Asumiendo que ya tienes tu DataFrame 'df' con la columna 'precio'

# Función para verificar estacionariedad con la prueba ADF
def check_stationarity(series, name=""):
    result = sts.adfuller(series.dropna())
    print(f'Prueba ADF para {name}:')
    print(f'Estadístico: {result[0]:.4f}')
    print(f'p-valor: {result[1]:.4f}')
    print(f'Valores críticos:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    
    # Interpretar resultado
    if result[1] <= 0.05:
        print(f"Conclusión: La serie {name} es estacionaria (p-valor <= 0.05)")
    else:
        print(f"Conclusión: La serie {name} no es estacionaria (p-valor > 0.05)")
    print("-" * 50)
    return result[1] <= 0.05

# Función para graficar series
def plot_series(original, transformed, title="Comparación de Series"):
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(original)
    plt.title("Serie Original")
    plt.subplot(212)
    plt.plot(transformed, color='red')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Verificar la serie original
print("ANÁLISIS DE LA SERIE ORIGINAL")
is_stationary = check_stationarity(df['precio'], "original")

# 1. DIFERENCIACIÓN
print("\n1. MÉTODO DE DIFERENCIACIÓN")
# Primera diferencia
diff1 = df['precio'].diff().dropna()
is_stationary_diff1 = check_stationarity(diff1, "primera diferencia")
plot_series(df['precio'], diff1, "Primera Diferencia")

# Si la primera diferencia no es suficiente, intentar con la segunda
if not is_stationary_diff1:
    diff2 = diff1.diff().dropna()
    is_stationary_diff2 = check_stationarity(diff2, "segunda diferencia")
    plot_series(df['precio'], diff2, "Segunda Diferencia")

# 2. TRANSFORMACIONES MATEMÁTICAS
print("\n2. TRANSFORMACIONES MATEMÁTICAS")

# Transformación logarítmica (solo si todos los valores son positivos)
if (df['precio'] > 0).all():
    log_transform = np.log(df['precio'])
    is_stationary_log = check_stationarity(log_transform, "logarítmica")
    plot_series(df['precio'], log_transform, "Transformación Logarítmica")
    
    # Si la transformación logarítmica no es suficiente, probar log + diferenciación
    if not is_stationary_log:
        log_diff = log_transform.diff().dropna()
        is_stationary_log_diff = check_stationarity(log_diff, "logarítmica + diferencia")
        plot_series(log_transform, log_diff, "Logarítmica + Primera Diferencia")
else:
    print("La serie contiene valores no positivos, omitiendo transformación logarítmica")

# Transformación de raíz cuadrada (solo si todos los valores son positivos)
if (df['precio'] > 0).all():
    sqrt_transform = np.sqrt(df['precio'])
    is_stationary_sqrt = check_stationarity(sqrt_transform, "raíz cuadrada")
    plot_series(df['precio'], sqrt_transform, "Transformación Raíz Cuadrada")
else:
    print("La serie contiene valores no positivos, omitiendo transformación de raíz cuadrada")

# Transformación Box-Cox (solo para valores positivos)
if (df['precio'] > 0).all():
    from scipy import stats
    # Encontrar el lambda óptimo para Box-Cox
    boxcox_transform, lambda_value = stats.boxcox(df['precio'])
    boxcox_transform = pd.Series(boxcox_transform, index=df.index)
    print(f"Lambda óptimo para Box-Cox: {lambda_value:.4f}")
    is_stationary_boxcox = check_stationarity(boxcox_transform, "Box-Cox")
    plot_series(df['precio'], boxcox_transform, f"Transformación Box-Cox (lambda={lambda_value:.4f})")
else:
    print("La serie contiene valores no positivos, omitiendo transformación Box-Cox")

# 3. SUSTRACCIÓN DE TENDENCIA O ESTACIONALIDAD
print("\n3. SUSTRACCIÓN DE TENDENCIA O ESTACIONALIDAD")

# Descomposición de series temporales
try:
    # Determinar frecuencia de datos (ajustar según tu serie)
    # Para datos diarios: freq=7, para datos mensuales: freq=12, etc.
    # Si los datos no tienen una frecuencia conocida, se puede usar un valor estimado
    freq = 12  # Ajustar según tu serie
    
    decomposition = seasonal_decompose(df['precio'], model='additive', period=freq)
    
    # Graficar descomposición
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(df['precio'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Tendencia')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Estacionalidad')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuos')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Serie sin tendencia
    detrended = df['precio'] - decomposition.trend
    detrended = detrended.dropna()
    is_stationary_detrended = check_stationarity(detrended, "sin tendencia")
    plot_series(df['precio'], detrended, "Serie sin Tendencia")
    
    # Serie sin estacionalidad
    deseasonalized = df['precio'] - decomposition.seasonal
    is_stationary_deseasonalized = check_stationarity(deseasonalized, "sin estacionalidad")
    plot_series(df['precio'], deseasonalized, "Serie sin Estacionalidad")
    
    # Serie con solo residuos (sin tendencia ni estacionalidad)
    residual = decomposition.resid
    residual = residual.dropna()
    is_stationary_residual = check_stationarity(residual, "residual")
    plot_series(df['precio'], residual, "Serie Residual")
    
except Exception as e:
    print(f"Error en la descomposición de la serie: {e}")
    print("Intenta ajustar la frecuencia (period) según tus datos")
    
    # Alternativa: Eliminar tendencia usando regresión lineal
    X = np.arange(len(df['precio']))
    y = df['precio'].values
    X = np.vstack([X, np.ones(len(X))]).T
    
    # Ajustar modelo de regresión lineal
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X[:, 0].reshape(-1, 1), y)
    
    # Predecir tendencia
    trend = model.predict(X[:, 0].reshape(-1, 1))
    
    # Eliminar tendencia
    detrended_linear = df['precio'] - trend
    detrended_linear = pd.Series(detrended_linear, index=df.index)
    is_stationary_detrended_linear = check_stationarity(detrended_linear, "sin tendencia lineal")
    plot_series(df['precio'], detrended_linear, "Serie sin Tendencia Lineal")

# 4. SUAVIZADO CON VENTANAS MÓVILES
print("\n4. SUAVIZADO CON VENTANAS MÓVILES")

# Media móvil
window_size = 7  # Ajustar según la serie
rolling_mean = df['precio'].rolling(window=window_size).mean()
# Restamos la media móvil para eliminar tendencias locales
detrended_ma = df['precio'] - rolling_mean
detrended_ma = detrended_ma.dropna()
is_stationary_ma = check_stationarity(detrended_ma, "sin tendencia por media móvil")
plot_series(df['precio'].iloc[window_size-1:], detrended_ma, f"Serie sin Tendencia (Media Móvil {window_size})")

# Suavizado exponencial
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple Exponential Smoothing
alpha = 0.2  # Factor de suavizado
smoothed = pd.Series(index=df.index)
smoothed.iloc[0] = df['precio'].iloc[0]

for t in range(1, len(df['precio'])):
    smoothed.iloc[t] = alpha * df['precio'].iloc[t] + (1 - alpha) * smoothed.iloc[t-1]

# Restamos el suavizado para obtener residuos
residuals_exp = df['precio'] - smoothed
is_stationary_exp = check_stationarity(residuals_exp.iloc[1:], "residuos de suavizado exponencial")
plot_series(df['precio'], residuals_exp, "Residuos de Suavizado Exponencial")

# Filtro Hodrick-Prescott
from statsmodels.tsa.filters.hp_filter import hpfilter

try:
    cycle, trend = hpfilter(df['precio'], lamb=1600)  # 1600 para datos trimestrales
    is_stationary_hp = check_stationarity(cycle, "ciclo de Hodrick-Prescott")
    plot_series(df['precio'], cycle, "Ciclo de Hodrick-Prescott")
except Exception as e:
    print(f"Error en el filtro Hodrick-Prescott: {e}")

# RESULTADOS Y RECOMENDACIONES
print("\nRESUMEN DE RESULTADOS:")
print("Verifica los resultados anteriores para determinar cuál método produjo una serie estacionaria.")
print("Recomendaciones:")
print("1. Si la diferenciación simple funcionó, usar ese método por su simplicidad.")
print("2. Para series con estacionalidad marcada, considerar la diferenciación estacional.")
print("3. Las transformaciones como Box-Cox pueden ayudar a estabilizar la varianza.")
print("4. Para análisis más profundos, combinar métodos (por ejemplo, log + diferenciación).")