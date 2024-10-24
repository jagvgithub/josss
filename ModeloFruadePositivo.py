# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:34:07 2023

@author: josgom
"""


# Ruta del archivo txt
ruta_archivo = 'C:/Users/josgom/Desktop/Credenciales.txt'

# 1. Abrir el archivo txt en modo lectura
with open(ruta_archivo, 'r') as archivo:
    # 2. Leer el contenido del archivo
    lineas = archivo.readlines()
# 3. Procesar los datos (opcional)


import numpy as np
# import connectorx as cx
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns 
import urllib
import pyodbc
import random
import time
import sys
import logging
import colorsys
import os
import json
import ast
import sqlalchemy
import pymssql
from sqlalchemy import create_engine
from multiprocessing import Pool
from sqlalchemy import text
import re
from datetime import datetime, timedelta
pd.options.mode.chained_assignment=None


def connect_to_database(server, database, user, password):
    try:
        conn = pymssql.connect(server=server, database=database, user=user, password=password)
        print(f"OKAY: Conexión exitosa a la base de datos {database} en el servidor {server}")
        return conn
    except Exception as e:
        raise Exception(f"ERROR: No se pudo establecer una conexión a la base de datos {database} en el servidor {server}: {e} archivo conexion_400")



def conexion_fabogriesgo():
    server_riesgo = "192.168.60.152:49505"
    database_riesgo = "Productos y transaccionalidad"
    # user='Usr_lkfraude'
    # password='Sq5q7v@K67nw'
    user='FINANDINA\josgom'
    password=str(lineas[1].strip())
    conn_riesgo = connect_to_database(server_riesgo, database_riesgo, user, password)
  
    return conn_riesgo


## catracterizacion 3ds 

query='''SELECT *
  FROM [Productos y transaccionalidad].[dbo].[Base3ds]'''
df = pd.read_sql(query,conexion_fabogriesgo())


# comportamiento por columna 


for columna in df.columns:
    # Cuenta la frecuencia de cada valor en la columna
    frecuencias = df[columna].value_counts()
    
    # Crea un gráfico de barras para las frecuencias
    plt.figure(figsize=(8, 6))
    frecuencias.plot(kind='bar')
    
    # Configura etiquetas y título
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.title(f'Frecuencia de valores en la columna {columna}')
    
    # Muestra el gráfico
    plt.show()


for columna in df.columns:
    # Cuenta la frecuencia de cada valor en la columna y calcula los porcentajes
    frecuencias = df[columna].value_counts(normalize=True) * 100
    
    # Crea un gráfico de barras para las frecuencias
    plt.figure(figsize=(8, 6))
    frecuencias.plot(kind='bar')
    
    # Configura etiquetas y título con los porcentajes
    plt.xlabel(columna)
    plt.ylabel('Porcentaje')
    plt.title(f'Porcentaje de valores en la columna {columna}')
    
    # Agrega los porcentajes a las etiquetas de las barras
    for i, v in enumerate(frecuencias):
        plt.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Muestra el gráfico
    plt.show()



# Convierte la columna DATE_TIME en un objeto DateTime
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

# Define las categorías de tiempo
categorias = ['00-05 am', '05-10 am', '10 am - 3 pm', '3 pm - 7 pm', '7 pm - 11 pm', '11 pm - 00']

# Función para asignar la categoría de tiempo
def asignar_categoria_tiempo(fecha_hora):
    hora = fecha_hora.time()
    if hora >= pd.to_datetime('00:00:00').time() and hora <= pd.to_datetime('04:59:59').time():
        return categorias[0]
    elif hora >= pd.to_datetime('05:00:00').time() and hora <= pd.to_datetime('09:59:59').time():
        return categorias[1]
    elif hora >= pd.to_datetime('10:00:00').time() and hora <= pd.to_datetime('14:59:59').time():
        return categorias[2]
    elif hora >= pd.to_datetime('15:00:00').time() and hora <= pd.to_datetime('18:59:59').time():
        return categorias[3]
    elif hora >= pd.to_datetime('19:00:00').time() and hora <= pd.to_datetime('22:59:59').time():
        return categorias[4]
    else:
        return categorias[5]

# Aplica la función para asignar la categoría de tiempo a cada registross
df['Categoria_tiempo'] = df['DATE_TIME'].apply(asignar_categoria_tiempo)




import locale

# Establece la configuración regional en español
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

df['Dia_de_la_semana'] = df['DATE_TIME'].dt.day_name()

df['Dia_del_mes'] = df['DATE_TIME'].dt.day


df['Description MCC2'] = df['Description MCC'] + '-' + '(' + df['MCC'] + ')'

### validación crosstab()




df['BIN'] = np.where(df['HPAN'].str[:6] == '414892', 'debito', 'credito')


list(df.columns)





[col for col in df.columns if 'amount' in col]



conteomcc= df['Terminal'].value_counts().reset_index()



conteomcc= df['País'].value_counts()



res = df.groupby('MCC')['bin_amount'].agg(['mean', 'count', 'sum', 'min', 'max', 'median', lambda x: x.quantile(0.7)]).reset_index()
res = res.rename(columns={'<lambda>': 'percentile_70'})


res=df.groupby('Description MCC')['bin_amount'].agg(['mean', 'count','sum','min','max','median']).reset_index()



grouped = df.groupby(['HPAN', 'Nombre_Comercio', 'Dia_del_mes']).size().reset_index(name='Cantidad_de_Registros')



grouped['Cantidad_de_Registros'].mean()




res=df.groupby('Nombre_Comercio')['bin_amount'].agg(['mean', 'count','sum']).reset_index()


res=df.groupby('Ciudad')['bin_amount'].agg(['mean', 'count','sum']).reset_index()


a=pd.crosstab(df['Nombre_Comercio'], df['Categoria_tiempo'],values=df['bin_amount'],aggfunc='sum')

list(df.columns)

conteo_y_porcentaje_mcc = df['Transacciones_Previas'].value_counts(normalize=True) * 100


conteo_y_porcentaje_mcc = df['Transacciones_Previas'].value_counts(normalize=True) * 100


a=(pd.crosstab(df['MCC'], df['Categoria_tiempo'],values=df['bin_amount'],aggfunc='sum')).reset_index()



# 


df = df[df['COUNTRY_CODE'].isin(['NG', 'AZ', 'TW', 'RW','TZ','IN','UA'])]





df2 = df.sort_values(by=['HPAN', 'DATE_TIME'])

# Calcula la diferencia en minutos entre transacciones para cada tarjeta
df2['Diferencia_Minutos'] = df2.groupby('HPAN')['DATE_TIME'].diff().dt.total_seconds() / 60
df2['Transacciones_Realizadas'] = df2.groupby('HPAN').cumcount() + 1

# Calcula el promedio de las diferencias para cada tarjeta

promedio_diferencia_por_tarjeta = df2.groupby('HPAN')['Diferencia_Minutos'].agg(['mean', 'count']).reset_index()
promedio_diferencia_por_tarjeta = promedio_diferencia_por_tarjeta.rename(columns={'mean': 'Promedio_Diferencia_Minutos', 'count': 'Cantidad_TX_Promedio'})


promedio_diferencia_por_tarjeta = df2.groupby('HPAN')['Diferencia_Minutos'].mean().reset_index()


promedio_diferencia_por_tarjeta['Diferencia_Minutos'].mean()


filtro = promedio_diferencia_por_tarjeta[(promedio_diferencia_por_tarjeta['Promedio_Diferencia_Minutos'] <= 3) & (promedio_diferencia_por_tarjeta['Cantidad_TX_Promedio'] > 1)]


filtro['Cantidad_TX_Promedio'].mean()


a=pd.crosstab(df['País'], df['BIN'],values=df['bin_amount'],aggfunc='sum')

a=pd.crosstab(df['País'], df['Débito - Crédito'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['País'], df['POS_ENTRY_MODE'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['MCC'], df['POS_ENTRY_MODE'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['MCC'], df['POS_CONDITION_CODE'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['País'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['MCC'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['Débito - Crédito'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['Dia_de_la_semana'], df['MCC'],values=df['bin_amount'],aggfunc='sum')


a=pd.crosstab(df['Dia_de_la_semana'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['Dia_del_mes'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['Dia_del_mes'], df['Débito - Crédito'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['Nombre_Comercio'], df['Dia_del_mes'],values=df['Monto transaccion'],aggfunc='sum')



a=pd.crosstab(df['Terminal'], df['Débito - Crédito'],values=df['Monto transaccion'],aggfunc='sum')
##analisis de redes sociales

a=pd.crosstab(df['MCC'], df['Terminal'],values=df['Monto transaccion'],aggfunc='sum')



## validacion si ha realizado tx en el mismo comercio antes 




df.sort_values(by=['HPAN', 'DATE_TIME'], inplace=True)

# Crea una nueva columna llamada 'Transacciones_Previas' para almacenar el resultado
df['Transacciones_Previas'] = False

# Itera a través de las filas del DataFrame
for index, row in df.iterrows():
    tarjeta_actual = row['HPAN']
    comercio_actual = row['MCC']
    fecha_actual = row['DATE_TIME']

    # Filtra las transacciones anteriores para la misma tarjeta y comercio
    transacciones_anteriores = df[(df['HPAN'] == tarjeta_actual) & (df['MCC'] == comercio_actual) & (df['DATE_TIME'] < fecha_actual)]

    # Si hay transacciones anteriores, marca la fila actual como True en 'Transacciones_Previas'
    if not transacciones_anteriores.empty:
        df.at[index, 'Transacciones_Previas'] = True









################################





query='''SELECT *
  FROM [Productos y transaccionalidad].[dbo].[BaseRiesgoConsolidado]
    where [Validacion Fraude] =1'''
df = pd.read_sql(query,conexion_fabogriesgo())

# query='''SELECT top(15000) *
#   FROM [Productos y transaccionalidad].[dbo].[BaseRiesgoConsolidado]
#     where [Validacion Fraude] =0'''
# df = pd.read_sql(query,conexion_fabogriesgo())




df.columns





for columna in df.columns:
    # Cuenta la frecuencia de cada valor en la columna
    frecuencias = df[columna].value_counts()
    
    # Crea un gráfico de barras para las frecuencias
    plt.figure(figsize=(8, 6))
    frecuencias.plot(kind='bar')
    
    # Configura etiquetas y título
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.title(f'Frecuencia de valores en la columna {columna}')
    
    # Muestra el gráfico
    plt.show()







for columna in df.columns:
    # Cuenta la frecuencia de cada valor en la columna y calcula los porcentajes
    frecuencias = df[columna].value_counts(normalize=True) * 100
    
    # Crea un gráfico de barras para las frecuencias
    plt.figure(figsize=(8, 6))
    frecuencias.plot(kind='bar')
    
    # Configura etiquetas y título con los porcentajes
    plt.xlabel(columna)
    plt.ylabel('Porcentaje')
    plt.title(f'Porcentaje de valores en la columna {columna}')
    
    # Agrega los porcentajes a las etiquetas de las barras
    for i, v in enumerate(frecuencias):
        plt.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Muestra el gráfico
    plt.show()


participacion_porcentaje = df['País'].value_counts(normalize=True) * 100




import pandas as pd



# Convierte la columna DATE_TIME en un objeto DateTime
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

# Define las categorías de tiempo
categorias = ['00-05 am', '05-10 am', '10 am - 3 pm', '3 pm - 7 pm', '7 pm - 11 pm', '11 pm - 00']

# Función para asignar la categoría de tiempo
def asignar_categoria_tiempo(fecha_hora):
    hora = fecha_hora.time()
    if hora >= pd.to_datetime('00:00:00').time() and hora <= pd.to_datetime('04:59:59').time():
        return categorias[0]
    elif hora >= pd.to_datetime('05:00:00').time() and hora <= pd.to_datetime('09:59:59').time():
        return categorias[1]
    elif hora >= pd.to_datetime('10:00:00').time() and hora <= pd.to_datetime('14:59:59').time():
        return categorias[2]
    elif hora >= pd.to_datetime('15:00:00').time() and hora <= pd.to_datetime('18:59:59').time():
        return categorias[3]
    elif hora >= pd.to_datetime('19:00:00').time() and hora <= pd.to_datetime('22:59:59').time():
        return categorias[4]
    else:
        return categorias[5]

# Aplica la función para asignar la categoría de tiempo a cada registross
df['Categoria_tiempo'] = df['DATE_TIME'].apply(asignar_categoria_tiempo)

print(df)




import locale

# Establece la configuración regional en español
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

df['Dia_de_la_semana'] = df['DATE_TIME'].dt.day_name()

df['Dia_del_mes'] = df['DATE_TIME'].dt.day


df['Description MCC2'] = df['Description MCC'] + '-' + '(' + df['MCC'] + ')'

### validación crosstab()


a=pd.crosstab(df['País'], df['BIN'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['País'], df['Débito - Crédito'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['País'], df['POS_ENTRY_MODE'],values=df['Monto transaccion'],aggfunc='sum')















a=pd.crosstab(df['MCC'], df['POS_ENTRY_MODE'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['MCC'], df['POS_CONDITION_CODE'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['País'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['MCC'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['Débito - Crédito'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['Dia_de_la_semana'], df['MCC'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['Dia_de_la_semana'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['Dia_del_mes'], df['Categoria_tiempo'],values=df['Monto transaccion'],aggfunc='sum')

a=pd.crosstab(df['Dia_del_mes'], df['Débito - Crédito'],values=df['Monto transaccion'],aggfunc='sum')


a=pd.crosstab(df['Nombre_Comercio'], df['Dia_del_mes'],values=df['Monto transaccion'],aggfunc='sum')



a=pd.crosstab(df['Terminal'], df['Débito - Crédito'],values=df['Monto transaccion'],aggfunc='sum')
##analisis de redes sociales

a=pd.crosstab(df['MCC'], df['Terminal'],values=df['Monto transaccion'],aggfunc='sum')



## validacion si ha realizado tx en el mismo comercio antes 




df.sort_values(by=['HPAN', 'DATE_TIME'], inplace=True)

# Crea una nueva columna llamada 'Transacciones_Previas' para almacenar el resultado
df['Transacciones_Previas'] = False

# Itera a través de las filas del DataFrame
for index, row in df.iterrows():
    tarjeta_actual = row['HPAN']
    comercio_actual = row['MCC']
    fecha_actual = row['DATE_TIME']

    # Filtra las transacciones anteriores para la misma tarjeta y comercio
    transacciones_anteriores = df[(df['HPAN'] == tarjeta_actual) & (df['MCC'] == comercio_actual) & (df['DATE_TIME'] < fecha_actual)]

    # Si hay transacciones anteriores, marca la fila actual como True en 'Transacciones_Previas'
    if not transacciones_anteriores.empty:
        df.at[index, 'Transacciones_Previas'] = True














a=pd.crosstab(df['MCC'], df['Transacciones_Previas'],values=df['Monto transaccion'],aggfunc='sum')



ruta_completa = r'C:\Users\josgom\Desktop\BORRAR\datosfruadepositivoceros15K.xlsx'  # Usar 'r' para interpretar la cadena como una ruta cruda

# Exportar el DataFrame a la ubicación deseada
df.to_excel(ruta_completa, index=False)

# validacion modelo 





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generar un conjunto de datos ficticio de compras reconocidas (1) y no reconocidas (0)
data = {
    'Monto': [100, 50, 200, 80, 120, 60, 180, 90],
    'Hora': [10, 15, 14, 11, 9, 16, 13, 12],
    'Reconocida': [1, 0, 1, 0, 1, 0, 1, 0]
}

df2 = pd.DataFrame(data)

# Dividir el conjunto de datos en entrenamiento y prueba
X = df2[['Monto', 'Hora']]
y = df2['Reconocida']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de clasificación (Random Forest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')


















import pandas as pd
import networkx as nx

# Crear un DataFrame de ejemplo
data = {'Nodo': ['A', 'B', 'C', 'D'],
        'Relacion': [('A', 'B'), ('B', 'C'), ('A', 'D'), ('C', 'D')]}

df = pd.DataFrame(data)

# Crear un objeto de grafo de NetworkX
G = nx.Graph()

# Agregar nodos y bordes desde el DataFrame
for _, row in df.iterrows():
    G.add_node(row['Nodo'])
    G.add_edge(row['Relacion'][0], row['Relacion'][1])

# Visualizar el grafo (opcional)
import matplotlib.pyplot as plt

nx.draw(G, with_labels=True)
plt.show()

# Realizar análisis de redes con NetworkX
# Por ejemplo, calcular la centralidad de grado
degree_centrality = nx.degree_centrality(G)
print(degree_centrality)











# pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org networkx



import networkx as nx
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame llamado 'df' con columnas 'MCC' y 'pais'
G = nx.Graph()

# Agregar nodos MCC
for mcc in df['MCC'].unique():
    G.add_node(mcc, type='MCC')

# Agregar nodos países
for country in df['País'].unique():
    if country is not None:  # Verificar si el valor no es None
        G.add_node(country, type='Country')

# Agregar conexiones entre MCC y países
for index, row in df.iterrows():
    mcc = row['MCC']
    country = row['País']
    if country is not None:  # Verificar si el valor no es None
        G.add_edge(mcc, country)

# Dibujar el grafo
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=100, font_size=8)
plt.show()




import networkx as nx
import matplotlib.pyplot as plt

# Ejemplo de lista de nodos y conexiones (reemplaza esto con tus propios datos)
nodes_mcc = ['MCC1', 'MCC2', 'MCC3']
nodes_countries = ['Country1', 'Country2', 'Country3', 'Country4']
edges = [('MCC1', 'Country1'), ('MCC2', 'Country1'), ('MCC2', 'Country2')]

# Crear un objeto de grafo
G = nx.Graph()

# Agregar nodos MCC
for mcc in nodes_mcc:
    G.add_node(mcc, type='MCC')

# Agregar nodos países
for country in nodes_countries:
    G.add_node(country, type='Country')

# Agregar conexiones entre MCC y países
for edge in edges:
    mcc, country = edge
    G.add_edge(mcc, country)

# Dibujar el grafo con mejoras
pos = nx.spring_layout(G, seed=42)  # Seed para la disposición determinista
plt.figure(figsize=(10, 8))  # Tamaño de la figura
node_labels = {node: node for node in G.nodes()}  # Etiquetas de nodos

# Dibujar nodos MCC con color azul y países con color rojo
nx.draw_networkx_nodes(G, pos, nodelist=nodes_mcc, node_color='blue', node_size=200)
nx.draw_networkx_nodes(G, pos, nodelist=nodes_countries, node_color='red', node_size=200)

# Dibujar etiquetas de nodos
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

# Dibujar bordes
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

plt.title("Gráfico de Redes Sociales")
plt.show()




import networkx as nx
import matplotlib.pyplot as plt

# Ejemplo de datos (reemplaza esto con tus propios datos)
data = [
    ('MCC1', 'Country1'),
    ('MCC2', 'Country1'),
    ('MCC2', 'Country2'),
    ('MCC3', 'Country3'),
    ('MCC4', 'Country2'),
]


data= df 
data=data.rename(columns={'País':'country'})


# Crear un objeto de grafo
G = nx.Graph()

# Agregar nodos y conexiones
for mcc, country in data:
    G.add_node(mcc, type='MCC')
    G.add_node(country, type='Country')
    G.add_edge(mcc, country)

# Dibujar el grafo con mejoras
pos = nx.spring_layout(G, seed=42)  # Seed para la disposición determinista
plt.figure(figsize=(10, 8))  # Tamaño de la figura
node_labels = {node: node for node in G.nodes()}  # Etiquetas de nodos

# Dibujar nodos MCC con color azul y países con color rojo
node_colors = ['blue' if 'MCC' in node else 'red' for node in G.nodes()]
node_sizes = [200 if 'MCC' in node else 300 for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

# Dibujar etiquetas de nodos
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

# Dibujar bordes
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

plt.title("Gráfico de Redes Sociales")
plt.show()


import networkx as nx
import matplotlib.pyplot as plt

# Ejemplo de datos (reemplaza esto con tus propios datos)
data = [
    ('MCC1', 'País1'),
    ('MCC2', 'País1'),
    ('MCC2', 'País2'),
    ('MCC3', 'País3'),
    ('MCC4', None),  # Ejemplo de valor nulo en "country"
]

# Crear un objeto de grafo
G = nx.Graph()

# Agregar nodos y conexiones
for index, row in data.iterrows():
    mcc = row['MCC']
    country = row['country']
    
    if mcc is not None and country is not None:
        G.add_node(mcc, type='MCC')
        G.add_node(country, type='Country')
        G.add_edge(mcc, country)

# Dibujar el grafo con mejoras
pos = nx.spring_layout(G, seed=42)  # Seed para la disposición determinista
plt.figure(figsize=(10, 8))  # Tamaño de la figura
node_labels = {node: node for node in G.nodes()}  # Etiquetas de nodos

# Dibujar nodos MCC con color azul y países con color rojo
node_colors = ['blue' if 'MCC' in node else 'red' for node in G.nodes()]
node_sizes = [200 if 'MCC' in node else 300 for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

# Dibujar etiquetas de nodos
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

# Dibujar bordes
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

plt.title("Gráfico de Redes Sociales")
plt.show()







## analisis exploratorio


resumen=df.describe()










# El análisis de asociación es una técnica utilizada para descubrir patrones de co-ocurrencia y relaciones entre variables en conjuntos de datos transaccionales. El algoritmo Apriori es uno de los enfoques más comunes para llevar a cabo el análisis de asociación. Aquí tienes una guía básica de cómo realizarlo:

# Preparación de los Datos:
# Tienes un conjunto de datos en el que cada fila representa una transacción y cada columna representa un artículo o elemento. Por ejemplo, en un supermercado, cada transacción podría ser un carrito de compras y cada elemento podría ser un producto.

# Cálculo de Frecuencias:
# Calcula la frecuencia de ocurrencia de cada elemento individual en las transacciones. Esto te ayudará a identificar los elementos más populares.

# Establecimiento de umbral mínimo:
# Define un umbral mínimo de frecuencia para considerar los elementos relevantes. Por ejemplo, puedes decidir que solo estás interesado en elementos que aparecen en al menos el 5% de las transacciones.

# Generación de Conjuntos de Elementos (Itemsets):
# Comienza generando conjuntos de elementos (itemsets) con un solo elemento (itemsets de un solo ítem). Luego, genera itemsets de dos elementos, tres elementos y así sucesivamente.

# Cálculo de Frecuencias de Itemsets:
# Calcula la frecuencia de ocurrencia de cada itemset en las transacciones.

# Aplicación del Algoritmo Apriori:
# El algoritmo Apriori busca itemsets frecuentes (que cumplan el umbral establecido) utilizando el principio de que cualquier subconjunto de un itemset frecuente también debe ser frecuente. El algoritmo comienza con itemsets de un solo ítem y genera itemsets más grandes en cada iteración.

# Generación de Reglas de Asociación:
# A partir de los itemsets frecuentes, se generan reglas de asociación. Cada regla consiste en un antecedente y un consecuente, y la confianza de la regla se calcula como la proporción de transacciones que contienen tanto el antecedente como el consecuente en comparación con las transacciones que contienen solo el antecedente.

# Selección de Reglas Relevantes:
# Puedes filtrar las reglas según ciertos criterios, como confianza y soporte mínimo. Esto te ayudará a identificar las reglas más fuertes y relevantes.

# Interpretación de las Reglas:
# Examina las reglas de asociación para identificar patrones interesantes y comprender las relaciones entre elementos. Las reglas pueden proporcionar información sobre los hábitos de compra y otras relaciones en tus datos.

# El análisis de asociación puede llevarse a cabo utilizando bibliotecas como mlxtend en Python. Aquí hay un ejemplo básico de cómo usar esta biblioteca para el análisis de asociación:

    

# pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org mlxtend


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df.columns

columns_to_drop = ['DATE_TIME', 'HPAN']

# Eliminar las columnas en la lista del DataFrame
df2 = df.drop(columns_to_drop, axis=1)




# Generar itemsets frecuentes
frequent_itemsets = apriori(df2, min_support=0.05, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filtrar y mostrar reglas relevantes
relevant_rules = rules[rules['confidence'] > 0.7]
print(relevant_rules)



## analisis de asociacion 


# Instala mlxtend si aún no lo tienes instalado
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Crear un DataFrame de ejemplo
data = {
    'Pan': [1, 0, 1, 1, 0],
    'Leche': [1, 1, 0, 1, 1],
    'Huevos': [1, 1, 1, 0, 0],
    'Cereal': [0, 1, 1, 0, 1],
    'Café': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Obtener conjuntos de elementos frecuentes utilizando Apriori
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Mostrar las reglas de asociación
print(rules)




import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Crear un DataFrame de ejemplo con variables continuas
data = {
    'Edad': [25, 30, 22, 35, 40],
    'Ingresos': [50000, 60000, 45000, 70000, 80000],
    'Compras': [3, 5, 1, 6, 7],
}

df = pd.DataFrame(data)

# Discretizar las variables continuas en rangos
bins = [20, 30, 40, 50]  # Definir los rangos que deseas
df['Edad_Rango'] = pd.cut(df['Edad'], bins)
df['Ingresos_Rango'] = pd.cut(df['Ingresos'], bins)
df['Compras_Rango'] = pd.cut(df['Compras'], bins)

# Eliminar las columnas originales
df.drop(['Edad', 'Ingresos', 'Compras'], axis=1, inplace=True)

# Obtener conjuntos de elementos frecuentes utilizando Apriori
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Mostrar las reglas de asociación
print(rules)





import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Crear un DataFrame de ejemplo con variables continuas
data = {
    'Edad': [25, 30, 22, 35, 40],
    'Ingresos': [50000, 60000, 45000, 70000, 80000],
    'Compras': [3, 5, 1, 6, 7],
}

df = pd.DataFrame(data)

# Discretizar las variables continuas en rangos
bins = [20, 30, 40, 50]  # Definir los rangos que deseas
labels = ['20-30', '30-40', '40-50']
df['Edad_Rango'] = pd.cut(df['Edad'], bins, labels=labels)
df.drop('Edad', axis=1, inplace=True)  # Eliminamos la columna original

# Realizar discretización de las otras variables de manera similar

# Obtener conjuntos de elementos frecuentes utilizando Apriori
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Mostrar las reglas de asociación
print(rules)












# pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pyfpgrowth

####

# Instala pyfpgrowth si aún no lo tienes instalado
# pip install pyfpgrowth

import pandas as pd
import pyfpgrowth

# Crear un DataFrame de ejemplo con variables categóricas
data = {
    'Producto1': ['A', 'B', 'A', 'C', 'B'],
    'Producto2': ['B', 'A', 'B', 'A', 'C'],
    'Producto3': ['C', 'B', 'A', 'B', 'A'],
}

df = pd.DataFrame(data)

# Convertir el DataFrame en una lista de transacciones
transactions = df.values.tolist()

# Aplicar el algoritmo FP-Growth
patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support=2)

# Mostrar los patrones frecuentes
print(patterns)


8345076/5937584-1








