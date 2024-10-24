# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:07:40 2023

@author: josgom
"""

import pandas as pd
ruta_inicial = r'C:/Users/josgom/Desktop/baseseg22.xlsx'
ruta_con_dobles_barras = ruta_inicial.replace('\\', '\\\\')
ruta_con_dobles_barras = ruta_con_dobles_barras.replace("\\", "/")
df = pd.read_excel(str(ruta_con_dobles_barras))
df = df.drop_duplicates(subset=["OBLIGACION"], keep="first")

datos=df


# estimacion numero de cluster optimos 


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Supongamos que tienes tus datos en un DataFrame llamado "datos"

# Seleccionar las variables para la segmentación
variables_segmentacion = datos

# Normalización de los datos
scaler = StandardScaler()
variables_normalizadas = scaler.fit_transform(variables_segmentacion)

# Número máximo de clústeres que deseas probar
max_clusters = 10

# Lista para almacenar los valores de SSW (para el método del codo)
ssw = []

# Lista para almacenar los valores de coeficiente de silueta (para el método de la silueta)
silhouette_scores = []

# Calcular la SSW y el coeficiente de silueta para diferentes valores de k
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(variables_normalizadas)
    ssw.append(kmeans.inertia_)
    if k > 1:
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(variables_normalizadas, labels)
        silhouette_scores.append(silhouette_avg)

# Gráfico del método del codo
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, max_clusters + 1), ssw, marker='o')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Suma de Cuadrados Dentro de Clúster (SSW)')
plt.title('Método del Codo para encontrar el número óptimo de clústeres')

# Gráfico del método de la silueta
plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Coeficiente de Silueta')
plt.title('Método de la Silueta para encontrar el número óptimo de clústeres')

plt.tight_layout()
plt.show()

# Encontrar el número óptimo de clústeres para el método del codo (Elbow Method)
# Buscar el "codo" en la curva
diferencias_ssw = np.diff(ssw)
indice_optimo_codo = np.argmax(diferencias_ssw) + 1
numero_optimo_codo = indice_optimo_codo + 1
print(f"Número óptimo de clústeres según el método del codo: {numero_optimo_codo}")

# Encontrar el número óptimo de clústeres para el método de la silueta (Silhouette Method)
# Buscar el valor máximo de coeficiente de silueta
indice_optimo_silueta = np.argmax(silhouette_scores) + 2
print(f"Número óptimo de clústeres según el método de la silueta: {indice_optimo_silueta}")





# normalizacion de las variables



import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

# Supongamos que tus datos están en un DataFrame llamado "datos"

# Paso 1: Realizar la normalización
# Opción 1: Estandarización (z-score)
scaler = StandardScaler()
variables_estandarizadas = scaler.fit_transform(datos)

# Opción 2: Normalización min-max
scaler = MinMaxScaler()
variables_normalizadas = scaler.fit_transform(datos)

# Paso 2: Aplicar la segmentación utilizando k-means en los datos normalizados
num_clusters = 5 # Número de clústeres que deseas obtener

# Opción 1: Utilizando variables estandarizadas
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(variables_estandarizadas)
etiquetas_clustering = kmeans.labels_

# Opción 2: Utilizando variables normalizadas
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(variables_normalizadas)
etiquetas_clustering = kmeans.labels_

# Paso 3: Agregar las etiquetas de clúster al DataFrame original
datos_segmentados = datos.copy()
datos_segmentados['Cluster'] = etiquetas_clustering

# Paso 4: Analizar los resultados de la segmentación
# Puedes realizar análisis descriptivos de cada clúster y visualizaciones para comprender las características de cada grupo resultante.


resumen_por_cluster = datos_segmentados.groupby('Cluster').describe()
print(resumen_por_cluster)


import matplotlib.pyplot as plt
import numpy as np

# Supongamos que tus datos están en un DataFrame llamado "datos_segmentados" y tienes cinco variables "Variable1" a "Variable5"

# Calcula el promedio de cada variable por clústers
promedios_por_cluster = datos_segmentados.groupby('Cluster').mean()


# Número de clústeres y variables
num_clusters = len(promedios_por_cluster)
num_variables = len(promedios_por_cluster.columns)

# Ángulos para el radar chart
angulos = np.linspace(0, 2 * np.pi, num_variables, endpoint=False)

# Cierra el polígono
angulos = np.concatenate((angulos, [angulos[0]]))

# Verifica si el número de variables coincide con el número de ángulos
if num_variables == len(angulos):
    # Crear la figura
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Plotea los datos de cada clúster
    for i in range(num_clusters):
        valores = promedios_por_cluster.iloc[i].values
        valores = np.concatenate((valores, [valores[0]])) # Cierra el polígono
        ax.plot(angulos, valores, label=f'Cluster {i}')

    # Añade leyendas, títulos y límites
    ax.set_thetagrids(angulos * 180 / np.pi, promedios_por_cluster.columns)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)
    plt.title('Visualización de la segmentación por radar chart')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.show()
else:
    print("El número de variables no coincide con el número de ángulos para el radar chart.")


# cantidad de cluster optimos 


import matplotlib.pyplot as plt

# Supongamos que tus datos están en un DataFrame llamado "datos_segmentados" y la columna "Cluster" contiene las etiquetas de los clústeres asignados por k-means

# Contar la cantidad de individuos en cada clúster
distribucion_por_cluster = datos_segmentados['Cluster'].value_counts().sort_index()

# Crear el gráfico de barras
plt.figure(figsize=(8, 6))
plt.bar(distribucion_por_cluster.index, distribucion_por_cluster.values)

# Etiquetas de las barras
for i, v in enumerate(distribucion_por_cluster.values):
    plt.text(i, v, str(v), ha='center', va='bottom')

# Etiquetas de los ejes y título
plt.xlabel('Clúster')
plt.ylabel('Cantidad de Individuos')
plt.title('Distribución de Individuos en cada Clúster')

# Leyenda
nombres_clusters = [f'Cluster {i}' for i in distribucion_por_cluster.index]
plt.legend(nombres_clusters)

# Mostrar el gráfico
plt.show()





import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# Validación de la segmentación
# Coeficiente de Silueta
silhouette_avg = silhouette_score(variables_normalizadas, etiquetas_clustering)

# Suma de Cuadrados Dentro de Clúster (SSW)
ssw = kmeans.inertia_

# Suma de Cuadrados Entre Clústeres (SSB)
ssb = calinski_harabasz_score(variables_normalizadas, etiquetas_clustering)

# Mostrar las métricas de validación
print(f'Coeficiente de Silueta: {silhouette_avg}')
print(f'SSW: {ssw}')
print(f'SSB: {ssb}')




### analisis por cluster edad

datos_segmentados.columns


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un gráfico separado por cada cluster para la variable "edad"

# grafico general 
datos_filtrados = datos_segmentados[datos_segmentados['Edad'] > 0]

# Crear un gráfico separado por cada cluster para la variable "Edad"
g = sns.FacetGrid(datos_filtrados, col="Cluster", col_wrap=3)
g.map_dataframe(sns.histplot, x="Edad", bins=10)
g.set_axis_labels("Edad", "Frecuencia")
g.set_titles("Cluster {col_name}")
plt.show()


# grafico individuales
clusters_unicos = datos_segmentados['Cluster'].unique()

# Crear un gráfico separado para cada segmento de cluster
for cluster in clusters_unicos:
    datos_filtrados = datos_segmentados[(datos_segmentados['Cluster'] == cluster) & (datos_segmentados['Edad'] > 0)]
    
    if not datos_filtrados.empty:
        plt.figure()
        sns.histplot(data=datos_filtrados, x='Edad', bins=10)
        plt.title(f'Histograma de Edad - Cluster {cluster}')
        plt.xlabel('Edad')
        plt.ylabel('Frecuencia')
        plt.show()
    else:
        print(f"No hay datos mayores a 0 en la variable Edad para el Cluster {cluster}")


# acierta originacion 

# Crear un gráfico separado por cada cluster para la variable "Edad"
g = sns.FacetGrid(datos_segmentados, col="Cluster", col_wrap=3)
g.map_dataframe(sns.histplot, x="Acierta_Master Originacion", bins=10)
g.set_axis_labels("Acierta originacion", "Frecuencia")
g.set_titles("Cluster {col_name}")
plt.show()


# Crear un gráfico separado para cada segmento de cluster
for cluster in clusters_unicos:
    datos_filtrados = datos_segmentados[(datos_segmentados['Cluster'] == cluster) ]
    
    if not datos_filtrados.empty:
        plt.figure()
        sns.histplot(data=datos_filtrados, x='Acierta_Master Originacion', bins=10)
        plt.title(f'Histograma de Acierta originacion - Cluster {cluster}')
        plt.xlabel('Acierta originacion')
        plt.ylabel('Frecuencia')
        plt.show()
    else:
        print(f"No hay datos mayores a 0 en la variable Edad para el Cluster {cluster}")



# acierta actual 

# Crear un gráfico separado por cada cluster para la variable "Edad"
g = sns.FacetGrid(datos_segmentados, col="Cluster", col_wrap=3)
g.map_dataframe(sns.histplot, x="Acierta_Master Actual (scoreExperian)", bins=10)
g.set_axis_labels("Acierta actual", "Frecuencia")
g.set_titles("Cluster {col_name}")
plt.show()


# Crear un gráfico separado para cada segmento de cluster
for cluster in clusters_unicos:
    datos_filtrados = datos_segmentados[(datos_segmentados['Cluster'] == cluster) ]
    
    if not datos_filtrados.empty:
        plt.figure()
        sns.histplot(data=datos_filtrados, x='Acierta_Master Actual (scoreExperian)', bins=10)
        plt.title(f'Histograma de Acierta originacion - Cluster {cluster}')
        plt.xlabel('Acierta actual')
        plt.ylabel('Frecuencia')
        plt.show()
    else:
        print(f"No hay datos mayores a 0 en la variable Edad para el Cluster {cluster}")


# recaudo

# Crear un gráfico separado por cada cluster para la variable "Edad"
g = sns.FacetGrid(datos_segmentados, col="Cluster", col_wrap=3)
g.map_dataframe(sns.histplot, x="Recaudo Promedio ultimos 18 meses", bins=10)
g.set_axis_labels("Recaudo historico", "Frecuencia")
g.set_titles("Cluster {col_name}")
plt.show()


# Crear un gráfico separado para cada segmento de cluster
for cluster in clusters_unicos:
    datos_filtrados = datos_segmentados[(datos_segmentados['Cluster'] == cluster) ]
    
    if not datos_filtrados.empty:
        plt.figure()
        sns.histplot(data=datos_filtrados, x='Recaudo Promedio ultimos 18 meses', bins=10)
        plt.title(f'Histograma de Acierta originacion - Cluster {cluster}')
        plt.xlabel('Recaudo historico')
        plt.ylabel('Frecuencia')
        plt.show()
    else:
        print(f"No hay datos mayores a 0 en la variable Edad para el Cluster {cluster}")




# saldo total productos

# Crear un gráfico separado por cada cluster para la variable "Edad"
g = sns.FacetGrid(datos_segmentados, col="Cluster", col_wrap=3)
g.map_dataframe(sns.histplot, x="SaldototalProductos", bins=10)
g.set_axis_labels("saldo total productos", "Frecuencia")
g.set_titles("Cluster {col_name}")
plt.show()


# Crear un gráfico separado para cada segmento de cluster
for cluster in clusters_unicos:
    datos_filtrados = datos_segmentados[(datos_segmentados['Cluster'] == cluster) ]
    
    if not datos_filtrados.empty:
        plt.figure()
        sns.histplot(data=datos_filtrados, x='SaldototalProductos', bins=10)
        plt.title(f'Histograma de Acierta originacion - Cluster {cluster}')
        plt.xlabel('saldo total productos')
        plt.ylabel('Frecuencia')
        plt.show()
    else:
        print(f"No hay datos mayores a 0 en la variable Edad para el Cluster {cluster}")


## relacion endeudamiento acierta actual 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Supongamos que tienes un DataFrame llamado datos_segmentados con las columnas 'Cluster', 'Endeudamiento originacion' y 'Endeudamiento Actual'

# Agrupar por cluster y calcular el valor medio de las variables
cluster_means = datos_segmentados.groupby('Cluster').mean()

cluster_means['variaciondeuda'] = (cluster_means['Endeudamiento Actual'] - cluster_means['Endeudamiento originacion']) / cluster_means['Endeudamiento originacion'] * 100

cluster_means['variacionacierta'] = (cluster_means['Acierta_Master Actual (scoreExperian)'] - cluster_means['Acierta_Master Originacion']) / cluster_means['Acierta_Master Originacion'] * 100
cluster_means['variacionproductos'] = (cluster_means['totalProductosActivos'] - cluster_means['TotalProductos']) / cluster_means['TotalProductos'] * 100

# Redondear los valores de las variaciones a números enteros y reemplazar valores no finitos por ceros
cluster_means['variaciondeuda'] = cluster_means['variaciondeuda'].round().fillna(0).astype(int)
cluster_means['variacionacierta'] = cluster_means['variacionacierta'].round().fillna(0).astype(int)
cluster_means['variacionproductos'] = cluster_means['variacionproductos'].round().fillna(0).astype(int)
# Ordenar el DataFrame por cluster en orden ascendente
cluster_means = cluster_means.sort_values(by='Cluster')

# Crear el gráfico de puntos
plt.figure(figsize=(10, 6))

# Obtener los nombres de los clusters
clusters = cluster_means.index

# Ancho de cada barra para separar los puntos principales
bar_width = 0.2

# Desplazamiento adicional para separar los puntos adicionales
extra_space = 0.2

# Eje x para el primer conjunto de puntos (deuda inicial)
x1 = np.arange(len(clusters))
# Eje x para el segundo conjunto de puntos (deuda final)
x2 = x1 + bar_width

# Eje x para el tercer conjunto de puntos (variación de deuda)
x3 = x2 + extra_space
# Eje x para el cuarto conjunto de puntos (variación de acierta)
x4 = x3 + bar_width

# Crear el gráfico de puntos para deuda inicial
plt.scatter(x1, cluster_means['Endeudamiento originacion'], color='blue', label='Deuda Inicial', alpha=0.7)

# Crear el gráfico de puntos para deuda final
plt.scatter(x2, cluster_means['Endeudamiento Actual'], color='green', label='Deuda Final', alpha=0.7)

# Crear el gráfico de puntos para la variación de deuda
plt.scatter(x3, cluster_means['variaciondeuda'], color='red', label='Variación de Deuda', alpha=0.7)

# Crear el gráfico de puntos para la variación de acierta
plt.scatter(x4, cluster_means['variacionacierta'], color='orange', label='Variación de Acierta', alpha=0.7)

# Ajustar el eje x
plt.xticks((x1 + x2) / 2, clusters)

# Añadir etiquetas para los porcentajes de variación de deuda en los puntos correspondientes
for i in range(len(x3)):
    plt.text(x3[i], cluster_means['variaciondeuda'][i] + 6, f'{cluster_means["variaciondeuda"][i]}%', ha='center', va='bottom', color='black', fontsize=9)

# Añadir etiquetas para los porcentajes de variación de acierta en los puntos correspondientes
for i in range(len(x4)):
    plt.text(x4[i], cluster_means['variacionacierta'][i] + 7, f'{cluster_means["variacionacierta"][i]}%', ha='center', va='bottom', color='black', fontsize=9)

# Añadir etiquetas y título
plt.xlabel('Cluster')
plt.ylabel('Deuda')
plt.legend(fontsize=8)
plt.title('Gráfico de Puntos - Deuda Inicial, Deuda Final, Variación de Deuda y Variación de Acierta por Cluster (Valores Medios)')

# Mostrar el gráficosssss
plt.show()


datos_segmentados.columns


# Especificar la ruta completa para guardar el archivo Excel
ruta_guardado_excel = 'C:\\Users\\josgom\\Desktop\\datos3.xlsx'

# Exportar el DataFrame a un archivo Excel
datos_segmentados.to_excel(ruta_guardado_excel, index=False)











