# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:23:44 2023

@author: josgom
"""


'''
#####################################################################################################
                                      BANCO FINANDINA BIC    
#####################################################################################################
                                Modelo de Segmentación SARLAFT        
#####################################################################################################
                                    PERSONA JURÍDICA - NATURAL   
#####################################################################################################
'''



print("Ingrese la segmentación que desea correr PN o PJ")

segmentacion = "PN"


'''
#####################################################################################################
                                       CONEXIÓN USUARIO    
#####################################################################################################
'''


# Ruta del archivo txt
ruta_archivo = 'C:/Users/josgom/Desktop/Credenciales.txt'

# 1. Abrir el archivo txt en modo lectura
with open(ruta_archivo, 'r') as archivo:
    # 2. Leer el contenido del archivo
    lineas = archivo.readlines()
# 3. Procesar los datos (opcional)



 # %% [1] Librerias Necesarias
'''
#####################################################################################################
                                      LIBRERÍAS NECESARIAS    
#####################################################################################################
'''

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib
import pyodbc
import random
import time
import sys
import logging
import colorsys
import os
import sqlalchemy
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,QuantileTransformer,RobustScaler
from sklearn.cluster import KMeans,MeanShift,AgglomerativeClustering,AffinityPropagation,OPTICS,DBSCAN,estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score,silhouette_samples
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics.pairwise import euclidean_distances
from os import mkdir
from sklearn.impute import SimpleImputer
from datetime import datetime
import connectorx as cx
import math 
from datetime import timedelta
from unidecode import unidecode
import re



 # %% [2] CONEXIONES A BASES DE DATOS  
 

'''
#####################################################################################################
                                      CONEXIONES A BASES DE DATOS    
#####################################################################################################
'''

def logger_config(nombre_modelo):
    logger = logging.getLogger('Modelo {}'.format(nombre_modelo))
    logger.setLevel(logging.INFO)
    consoleHandle = logging.StreamHandler(sys.stdout)
    consoleHandle.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandle.setFormatter(formatter)
    logger.addHandler(consoleHandle)
    return logger

def config_db_trusted(con_trusted,con_driver,con_server,con_databaseName):
    conn_str=(r'Trusted_Connection={val_trusted};' r'Driver={val_driver};' r'Server={val_server};' r'Database={val_dbname};').format(val_trusted=con_trusted,val_driver=con_driver,val_server=con_server,val_dbname=con_databaseName)
    conn_format=urllib.parse.quote(conn_str)
    engine=sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={conn_format}', fast_executemany=True)
    conn_engine=engine.connect()
    return engine,conn_engine  

def load_data(query: str,con_config,model_logger):
    cols_return = pd.read_sql(query, con=con_config)
    model_logger.info('Datos cargados con éxito')
    return cols_return


#### Configuración conexión

logger=logger_config('Segmentación de riesgo')
config_db=config_db_trusted(con_trusted='yes',
                            con_driver='ODBC Driver 17 for SQL Server',
                            con_server='FABOGRIESGO\RIESGODB',
                            con_databaseName='ModelosSARLAFT')



server_name = 'FABOGSQL01\\AUDITORIA,52715'
database_name = 'AUDITORIA_COMPARTIDA'
integrated_security = 'yes'  # Para autenticación de Windows


SQL_SERVER_RIESGO= "fabogriesgo:49505"
SQL_DB_RIESGO = "AlertasFraude"
sql_connection = f"mssql://{SQL_SERVER_RIESGO}/{SQL_DB_RIESGO}?trusted_connection=true"


# %% [3] AUTOMATIZACIÓN REPORTE HISTÓRICO 

'''
#####################################################################################################
                                  AUTOMATIZACIÓN REPORTE HISTÓRICO    
#####################################################################################################
'''


#sns.set() 
#pd.set_option('mode.chained_assignment',None)
pd.options.mode.chained_assignment=None

fecha_actual=dt.datetime.now()
startTime = time.time()


year= dt.datetime.now().year
juridico= 'dbo.ClienteProductoJuridico_'
natural = 'dbo.ClienteProductoNatural_'

df = pd.DataFrame(np.array([[year,  '01', year-1,1,'12','C:/Users/josgom/Desktop/historicoSARLAFT/'], 
                             [year, '02', year,2,'01'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '03', year,3,'02'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '04', year,4,'03'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '05', year,5,'04'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '06', year,6,'05'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '07', year,7,'06'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '08', year,8,'07'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '09', year,9,'08'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '10', year,10,'09' ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '11', year,11,'10' ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '12', year,12,'11' ,'C:/Users/josgom/Desktop/historicoSARLAFT/']]),
                    columns=['year', 'month', 'yy','nummonth','mesbase','prueba'])

df['basejuridico'] = juridico + df.yy.str.cat(df.mesbase)
df['basenatural'] = natural + df.yy.str.cat(df.mesbase)
df['ruta'] = df.yy.str.cat(df.mesbase)
df['carpeta'] = df.prueba.str.cat(df.ruta)


'''
#####################################################################################################
                    CARPETA DONDE SE ALOJARÁN LOS RESULTADOS HISTÓRICOS    
#####################################################################################################
'''

# mkdir(str(df[(df.nummonth ==str( dt.datetime.now().month))].iloc[0, 9]))


'''
#####################################################################################################
                    FUNCIONES NECESARIAS PARA EL DESARROLLO CORRECTO DE LA SEGMENTACIÓN    
#####################################################################################################
'''

 # %% [4] FUNCIONES NECESARIAS PARA EL DESARROLLO CORRECTO DE LA SEGMENTACIÓN  

'''# Esta función permite ajustar el formato de los campos que poseen formato juliana, dejando el campo en formato estándar  #'''

def convertir_fecha_juliana(fecha_juliana):
    try:
        anno = int(fecha_juliana[:4])
        dia_del_anno = int(fecha_juliana[4:])
        fecha_estandar = datetime(anno, 1, 1) + timedelta(days=dia_del_anno - 1)
        return fecha_estandar.strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return None  # Maneja los valoconsolidado incorrectos proporcionando un valor nulo


'''# Esta función permite ajustar el formato de los campos que no tienen el largo correcto, para garantizar el formato de fecha estándar  #'''

def agregar_cero(valor):
    if len(valor) == 5:
        return '0' + valor
    else:
        return valor

'''# Esta función permite ajustar el formato de los campos que no tienen el largo correcto, para garantizar el formato de fecha estándar  #'''


def agregar_20(valor):
    if len(valor) == 6:
        return valor[:4] + '20' + valor[4:]
    else:
        return valor

'''# Esta función permite ajustar el formato de los campos que no tienen el largo correcto, para garantizar el formato de fecha estándar  #'''


def convertir_fecha(fecha_str):
    try:
        fecha_obj = datetime.strptime(fecha_str, "%d%m%Y")
        return fecha_obj.strftime("%Y-%m-%d")
    except ValueError:
        return None

'''# Esta función permite actualizar la información demográfica del cup0003 con la información que se tiene en tierra (pailita)  #'''


def actualizar_columnas(df, columnas):
    for columna in columnas:
        condicion = df[columna].apply(lambda x: isinstance(x, str) and x.strip() != "")
        df[columna] = np.where(condicion, df[columna], df[f'{columna}_actualizado'])


'''# Esta función permite conectar con los servidores del Banco  #'''

def connect_to_database(server, database):
    try:
        conn = pyodbc.connect(driver='{SQL Server}', server=server, database=database, trusted_connection='yes',fast_executemany=True)
        print(f"OKAY: Conexión exitosa a la base de datos {database} en el servidor {server}")
        return conn
    except Exception as e:
        raise Exception(f"ERROR: No se pudo establecer una conexión a la base de datos {database} en el servidor {server}: {e} archivo conexion_400")

def conexion_fabogsqlclu():
    server_riesgo = "FABOGRIESGO\RIESGODB"
    database_riesgo = "Productos y transaccionalidad"
    conn_riesgo = connect_to_database(server_riesgo, database_riesgo)
    return conn_riesgo

conn = conexion_fabogsqlclu()


'''# Esta función cargar datos directamente del CORE bancario en menos tiempo  #'''

def cargue_openquery(conn_riesgo,sqlquery):
   
    chunk_size = 50000  # Define el tamaño del lote según tus necesidades

    # Crea un generador de lotes para recuperar datos en pedazos
    data_chunks = pd.read_sql(sqlquery, conn_riesgo, chunksize=chunk_size)

    # Inicializa una lista para almacenar los resultados
    result_chunks = []

    # Procesa cada lote por separado y almacena los resultados
    chunk_number = 0

    # Procesa cada lote por separado y almacena los resultados
    for chunk in data_chunks:
        chunk_number += 1
        print(f'Procesando chunk {chunk_number}...')
        result_chunks.append(chunk)

    # Concatena todos los lotes en un DataFrame único
    Datos_Nuevos = pd.concat(result_chunks, ignore_index=True)

    return Datos_Nuevos


'''# Esta función cargar datos directamente del CORE bancario en menos tiempo por paquetes (No se usa en esta segmentación)  #'''

def split_and_execute_queries(documentos_unicos, batch_size=300):
    # Divide la lista de documentos en lotes del tamaño especificado
    batches = [documentos_unicos[i:i + batch_size] for i in range(0, len(documentos_unicos), batch_size)]

    # Inicializa una lista para almacenar los resultados de las consultas
    results = []

    for batch in batches:
        # Convierte los documentos del lote actual en un formato adecuado para la consulta SQL
        documentos_con_comillas = ','.join(["''" + doc + "''" for doc in batch])

        # Crea la consulta SQL para el lote actual
        query_cup = f'''select * from openquery(DB2400_182,'select cuna1,cuna2,cuna3,cussnr,cuclph,cuopdt,cuinc,cuema1,cumtnd from BNKPRD01.cup003 where cussnr in({documentos_con_comillas})')'''

        # Ejecuta la consulta y guarda los resultados en una lista
        result = cargue_openquery(conn, query_cup)

        # Agrega los resultados a la lista de resultados
        results.append(result)

    # Combina los resultados en un solo DataFrame
    consolidated_results = pd.concat(results, ignore_index=True)

    # Retorna el DataFrame res
    return consolidated_results

'''# Esta función permite calcular el riesgo producto de acuerdo a la documentación y los anexos que duspuso el área de cumpliento  #'''

def riesgo_producto(row):
    if (row['Crédito libre inversión'] >= 1 or
        row['Cuenta corriente'] >= 1 or
        row['Ahorro de la red'] >= 1 or
        row['FlexiDigital'] >= 1 or
        row['Castigados'] >= 1 or
        row['Otros ahorros'] >= 1):
        return 3
    elif (row['TDC Digital'] >= 1 or
          row['TDC Física'] >= 1 or
          row['Crédito vehículo'] >= 1 or
          row['Leasing vehículo'] >= 1 or
          row['Crédito hipotecario'] >= 1 or
          row['Otros activos'] >= 1 or
          row['CDT'] >= 1 or
          row['Plan mayor'] >= 1):
        return 2
    elif (row['Cartera vendida'] >= 1 or
          row['Nomina Finandina'] >= 1 or
          row['Libranza'] >= 1 or
          row['Maquina agrícola'] >= 1 or
          row['Redescuentos'] >= 1):
        return 1
    else:
        return np.nan


'''# Esta función permite homologar datos de jurisdicción de cara a la estimación del Riesgo inherente  #'''

def asignar_departamento(df, columna_ciudad, valor_ciudad, valor_asignado):
    condicion = df[columna_ciudad] == valor_ciudad
    df['Departamento'] = np.where(condicion, valor_asignado, df['Departamento'])


'''# Esta función permite calcular el riesgo ahorro de acuerdo a la documentación y los anexos que duspuso el área de cumpliento  #'''

def calcular_ahorro(row):
    if (row['Ahorro de la red'] > 0) or (row['FlexiDigital'] > 0) or (row['Nomina Finandina'] > 0) or (row['Otros ahorros'] > 0):
        return 1
    elif row['Cuenta corriente'] > 0:
        return 2
    elif row['CDT'] > 0:
        return 4
    else:
        return 0


'''# Esta función permite calcular el riesgo crédito de acuerdo a la documentación y los anexos que duspuso el área de cumpliento  #'''

def calcular_credito(row):
    if (row['Maquina agrícola'] > 0) or (row['Plan mayor'] > 0) or (row['Otros activos'] > 0):
        return 6
    elif (row['Crédito hipotecario'] > 0) or (row['Otros activos'] > 0):
        return 1
    elif (row['TDC Digital'] > 0) or (row['TDC Física'] > 0):
        return 5
    elif (row['Leasing vehículo'] > 0):
        return 2
    elif (row['Crédito libre inversión'] > 0) or (row['Libranza'] > 0) or (row['Crédito vehículo'] > 0) or (row['Redescuentos'] > 0) or (row['Cartera vendida'] > 0) or (row['Castigados'] > 0):
        return 7
    else:
        return 0


 # %% [5] SEGMENTACIÓN A EJECUTAR (PJ:PERSONA JURÍDICA) (PN:PERSONA NATURAL)   

'''
####################################################################################################
                SEGMNETACIÓN A EJECUTAR (PJ:PERSONA JURÍDICA) (PN:PERSONA NATURAL)   
#####################################################################################################
'''



if segmentacion == "PJ":
    '''
    #####################################################################################################
                                          PERSONA JURÍDICA   
    #####################################################################################################
    '''
    
    # mkdir(str(df[(df.nummonth ==str( dt.datetime.now().month))].iloc[0, 9]))
    
    # consolidado_pj = '''SELECT distinct CAST(NIT AS VARCHAR(255)) AS DocumentoCliente,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
    #   FROM [ModelosSARLAFT].[dbo].[CONSOLIDADOPJ]'''
    # consolidado_pj = cx.read_sql(conn = sql_connection, query = consolidado_pj, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


    consolidado_pj = '''SELECT CAST([Documento] AS VARCHAR(20)) AS DocumentoCliente,Nombre,Tipo_identificacion ,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
      FROM [ModelosSARLAFT].[dbo].[CONSOLIDADOPJ]'''
    consolidado_pj = cx.read_sql(conn = sql_connection, query = consolidado_pj, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])
    
    

    '''# mapeo de clientes que tienen obligaciones propiedad de incomercio  #'''

    clientes_incomercio = '''SELECT distinct [IDENTIFICACION]  as DocumentoCliente,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
      FROM [CarteraFinaInco].[dbo].[PLANO_MAYOR_CLIENTE]
      where PROPIEDAD = 'INCOMERCIO' '''
    clientes_incomercio = cx.read_sql(conn = sql_connection, query = clientes_incomercio, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])
    clientes_incomercio=clientes_incomercio.drop_duplicates()
    clientes_incomercio['DocumentoCliente'] = clientes_incomercio['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()
    clientes_incomercio = clientes_incomercio.loc[clientes_incomercio['DocumentoCliente'].apply(lambda x: x.startswith(('8', '9')) or len(x) == 9)]
    
    consolidado_pj = ((pd.concat([consolidado_pj, clientes_incomercio])).drop_duplicates(subset='DocumentoCliente')).reset_index(drop=True)
    consolidado_pj['Incomercio'] = consolidado_pj['DocumentoCliente'].isin(clientes_incomercio['DocumentoCliente']).astype(int)
    consolidado_pj = consolidado_pj[consolidado_pj['DocumentoCliente']!='0']
    consolidado_pj = consolidado_pj[consolidado_pj['DocumentoCliente']!=0]
else:
    '''
    #####################################################################################################
                                          PERSONA NATURAL   
    #####################################################################################################
    '''
    # consolidado_pj = '''SELECT distinct CAST([Numero ID] AS VARCHAR(255)) AS DocumentoCliente,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
    #   FROM [ModelosSARLAFT].[dbo].[CONSOLIDADOPN]'''
    # consolidado_pj = cx.read_sql(conn = sql_connection, query = consolidado_pj, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


    consolidado_pj = '''SELECT CAST([Documento] AS VARCHAR(20)) AS DocumentoCliente,Nombre,Tipo_identificacion ,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
      FROM [ModelosSARLAFT].[dbo].[CONSOLIDADOPN]'''
    consolidado_pj = cx.read_sql(conn = sql_connection, query = consolidado_pj, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


    '''# mapeo de clientes que tienen obligaciones propiedad de incomercio  #'''

    clientes_incomercio = '''SELECT distinct [IDENTIFICACION]  as DocumentoCliente,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
      FROM [CarteraFinaInco].[dbo].[PLANO_MAYOR_CLIENTE]
      where PROPIEDAD = 'INCOMERCIO' '''
    clientes_incomercio = cx.read_sql(conn = sql_connection, query = clientes_incomercio, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])
    clientes_incomercio=clientes_incomercio.drop_duplicates()
    clientes_incomercio['DocumentoCliente'] = clientes_incomercio['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()
    clientes_incomercio = clientes_incomercio.loc[~clientes_incomercio['DocumentoCliente'].apply(lambda x: x.startswith(('8', '9')) or len(x) == 9)]
    
    consolidado_pj = ((pd.concat([consolidado_pj, clientes_incomercio])).drop_duplicates(subset='DocumentoCliente')).reset_index(drop=True)
    consolidado_pj['Incomercio'] = consolidado_pj['DocumentoCliente'].isin(clientes_incomercio['DocumentoCliente']).astype(int)

    consolidado_pj = ((pd.concat([consolidado_pj, clientes_incomercio])).drop_duplicates(subset='DocumentoCliente')).reset_index(drop=True)
    consolidado_pj['Incomercio'] = consolidado_pj['DocumentoCliente'].isin(clientes_incomercio['DocumentoCliente']).astype(int)
    consolidado_pj = consolidado_pj[consolidado_pj['DocumentoCliente']!='0']
    consolidado_pj = consolidado_pj[consolidado_pj['DocumentoCliente']!=0]



# %% [6] VALIDACIÓN PRODUCTOS ACTIVOS DEL CLIENTE

'''# consolidado de productos   #'''

consolidado_productos = '''SELECT DocumentoCliente		
 ,CASE WHEN TipoProducto='TDC' AND LineaProducto LIKE ('%Virtual%') THEN 'TDC Digital'
			      WHEN TipoProducto='TDC' AND LineaProducto NOT LIKE ('%Virtual%') THEN 'TDC Física'
			      ELSE LineaProducto
             END LineaProducto,TipoProducto,EstadoCuenta,1 as cantidad,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	  		 FROM [Productos y transaccionalidad].[dbo].[ConsolidadoProductos]
			 WHERE EstadoCuenta='Activa' '''
consolidado_productos = cx.read_sql(conn = sql_connection, query = consolidado_productos, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

# consolidado_productos['DocumentoCliente']=consolidado_productos['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

consolidado_productos['DocumentoCliente']=consolidado_productos['DocumentoCliente'].astype(str).str.strip()


# consolidado_pj['DocumentoCliente']=consolidado_pj['DocumentoCliente'].str.strip().astype(str)


'''# producto pivoteado   #'''

productos_pivoteado = (
    consolidado_productos
    .pivot_table(index='DocumentoCliente', columns='LineaProducto', values='cantidad', aggfunc='sum')
    .fillna(0)
    .astype('int64')
    .reset_index(drop=False)
    .loc[(consolidado_productos['DocumentoCliente'] != "0") & (consolidado_productos['DocumentoCliente'] != '')]
)



'''# último producto aperturado   #'''

ultimo_producto_aperturado = '''SELECT DocumentoCliente,max(FechaApertura) as FechaApertura,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
 FROM [Productos y transaccionalidad].[dbo].[ConsolidadoProductos]			 
GROUP BY DocumentoCliente'''
ultimo_producto_aperturado = (cx.read_sql(conn = sql_connection, query = ultimo_producto_aperturado, partition_on="Rank", partition_num=10,  return_type="pandas")).groupby('DocumentoCliente').first().reset_index().drop(columns=['Rank'])




# %% [7] INFORMACIÓN DEMOGRÁFICA DEL CLIENTE CUP003

'''# tiempo de carga cup003  6 minutos   #'''


consolidado_cup = '''select * from openquery(DB2400_182,'select cuna1,cuna2,cussnr,cuclph,cuopdt,cuinc,cuema1,cumtnd,cucens from BNKPRD01.cup003 ')'''

consolidado_cup = (cargue_openquery(conn, consolidado_cup)).rename(columns={'CUNA2':'DireccionActual'
                                                                          ,'CUSSNR':'DocumentoCliente'
                                                                          ,'CUCLPH':'Celular'
                                                                          ,'CUOPDT':'FechaVinculacion'
                                                                          ,'CUINC':'MontoIngresos'
                                                                          ,'CUEMA1':'Correo'
                                                                          ,'CUMTND':'FechaUltimaActualizacionCore'
                                                                          ,'CUNA1':'NombreCliente'
                                                                          ,'CUCENS' : 'CMCODICBS'})

consolidado_cup['CMCODICBS'] = consolidado_cup['CMCODICBS'].astype(str).str.replace('.', '')
consolidado_cup['CMCODICBS'] = consolidado_cup['CMCODICBS'].astype(int)
 
consolidado_cup['DocumentoCliente'] = consolidado_cup['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()
consolidado_cup['FechaUltimaActualizacionCore'] = consolidado_cup['FechaUltimaActualizacionCore'].astype(int).astype(str)
consolidado_cup['FechaUltimaActualizacionCore'] = consolidado_cup['FechaUltimaActualizacionCore'].apply(convertir_fecha_juliana)
consolidado_cup['FechaVinculacion'] = consolidado_cup['FechaVinculacion'].astype(int).astype(str)
consolidado_cup['FechaVinculacion']=consolidado_cup['FechaVinculacion'].apply(agregar_cero)
consolidado_cup['FechaVinculacion']=consolidado_cup['FechaVinculacion'].apply(agregar_20)
consolidado_cup['FechaVinculacion'] = np.where(consolidado_cup['FechaVinculacion'].str.len() == 8, consolidado_cup['FechaVinculacion'], np.nan)
consolidado_cup['FechaVinculacion'] = consolidado_cup['FechaVinculacion'].astype(str)
consolidado_cup['FechaVinculacion']=consolidado_cup['FechaVinculacion'].apply(lambda x: convertir_fecha(x))
consolidado_cup['Correo']=consolidado_cup['Correo'].astype(str).str.strip()
consolidado_cup['Correo'] = consolidado_cup['Correo'].apply(lambda x: None if x == '' else x)
consolidado_cup['DireccionActual']=consolidado_cup['DireccionActual'].astype(str).str.strip()
consolidado_cup['DireccionActual'] = consolidado_cup['DireccionActual'].apply(lambda x: None if x == '' else x)

# multiplicamos los ingconsolidadoos del cup003 por mil para manejar la misma escala 
consolidado_cup['MontoIngresos'] *= 1000

consolidado_cup['MontoIngresos'] = consolidado_cup['MontoIngresos'].apply(lambda x: None if x <= 0 else x)

consolidado_cup['Celular'] = consolidado_cup['Celular'].apply(lambda x: None if x <= 0 else x)



munidian = '''select CMCODICBS,CMNOMMUNI as CiudadActual,CMNOMDEPA as Departamento,CMNOMMUNI2 as Municipio
  from openquery(DB2400_182,'select CMCODICBS,CMNOMMUNI,CMNOMDEPA,CMNOMMUNI as CMNOMMUNI2 from interfaces.munidian')'''




munidian = (cargue_openquery(conn, munidian)).rename(columns={'CMNOMMUNI':'CiudadActual'
                                                                          ,'CMNOMDEPA':'Departamento','CMNOMMUNI':'Municipio'})

def quitar_parentesis(texto):
    return re.sub(r'\([^)]*\)', '', texto)





munidian['Municipio']=munidian['Municipio'].apply(lambda x: quitar_parentesis(x)).str.strip().str.upper()
munidian['Departamento'] = munidian['Departamento'].apply(lambda x: unidecode(x)).str.upper()
munidian['Departamento'] = munidian['Departamento'].str.replace('[^a-zA-Z ]', '', regex=True).str.upper()




consolidado_cup=pd.merge(consolidado_cup,munidian[['CMCODICBS','CiudadActual', 'Departamento','Municipio']],on='CMCODICBS',how='left')
consolidado_cup['Departamento']=consolidado_cup['Departamento'].str.strip()
consolidado_cup['Departamento'] = np.where(consolidado_cup['Departamento'] == 'BOGOTA DC', 'BOGOTA', consolidado_cup['Departamento'])
consolidado_cup=consolidado_cup.groupby('DocumentoCliente').first().reset_index()



'''# carga de bases demográficas para mejorar completitud de la base a entregar #'''


# %% [8] INFORMACIÓN DEMOGRÁFICA BASES CIENCIA DE DATOS

demografico_pj = '''SELECT DocumentoCliente AS DocumentoCliente
       ,NombreCliente
	   ,TipoPersona
	   ,CASE WHEN CiudadActual IS NOT NULL THEN dbo.RemoveNonAlphaCharacters(REPLACE(REPLACE(REPLACE(UPPER(LEFT(CiudadActual,CHARINDEX('(',CiudadActual+'(')-1)),'D.C',''),'FLORIDA BLANCA','FLORIDABLANCA'),'SANTAFE DE BOGOTA DC','BOGOTA')) COLLATE SQL_Latin1_General_CP1253_CI_AI
	         ELSE CiudadActual
        END CiudadActual
	   ,FechaVinculacion
	   ,MontoIngresos
	   ,CodigoCIIU
	   ,Celular
	   ,Correo
	   ,DireccionActual
       ,ActividadEconomica
	   ,FechaUltimoMantenimiento AS FechaUltimaActualizacionCore,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [Productos y transaccionalidad].[dbo].[ConsolidadoJuridicaDemografia]'''
demografico_pj = cx.read_sql(conn = sql_connection, query = demografico_pj, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


demografico_pj['FechaVinculacion'] = pd.to_datetime(demografico_pj['FechaVinculacion']).dt.strftime('%Y-%m-%d')


demografico_pj['FechaUltimaActualizacionCore'] = (pd.to_datetime(demografico_pj['FechaUltimaActualizacionCore'])).dt.strftime('%Y-%m-%d')


demografico_pn = '''SELECT DocumentoCliente AS DocumentoCliente
      ,NombreCliente
	   ,TipoPersona
	   ,CASE WHEN CiudadActual IS NOT NULL THEN dbo.RemoveNonAlphaCharacters(REPLACE(REPLACE(REPLACE(UPPER(LEFT(CiudadActual,CHARINDEX('(',CiudadActual+'(')-1)),'D.C',''),'FLORIDA BLANCA','FLORIDABLANCA'),'SANTAFE DE BOGOTA DC','BOGOTA')) COLLATE SQL_Latin1_General_CP1253_CI_AI
	         ELSE CiudadActual
        END CiudadActual
	   ,FechaVinculacion
	   ,MontoIngresos
	   ,CodigoCIIU
	   ,Celular
	   ,Correo
	   ,DireccionActual
       ,ActividadEconomica
	   ,FechaUltimoMantenimiento AS FechaUltimaActualizacionCore,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [Productos y transaccionalidad].[dbo].[ConsolidadoNaturalDemografia]'''
demografico_pn = cx.read_sql(conn = sql_connection, query = demografico_pn, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])
demografico_pn['FechaVinculacion'] = pd.to_datetime(demografico_pn['FechaVinculacion']).dt.strftime('%Y-%m-%d')
demografico_pn['FechaUltimaActualizacionCore'] = (pd.to_datetime(demografico_pn['FechaUltimaActualizacionCore'])).dt.strftime('%Y-%m-%d')


demografico = (pd.concat([demografico_pj, demografico_pn], axis=0)).groupby('DocumentoCliente').first().reset_index()
demografico['DocumentoCliente'] = demografico['DocumentoCliente'].astype(str).str.replace('-', '')



consolidado_pj = (pd.merge(consolidado_pj, productos_pivoteado,on='DocumentoCliente',how='left')).fillna(0)
consolidado_pj = pd.merge(consolidado_pj, ultimo_producto_aperturado,on='DocumentoCliente',how='left')


# data=consolidado_pj=data

consolidado_pj = pd.merge(consolidado_pj, consolidado_cup,on='DocumentoCliente',how='left')



# valores_nulos_por_columna1 = consolidado_pj.isnull().sum().reset_index()






consolidado_pj['NombreCliente'] = consolidado_pj['NombreCliente'].astype(str).str.strip()
consolidado_pj['NombreCliente'] = consolidado_pj['NombreCliente'].apply(lambda x: None if x == '' else x)
consolidado_pj['NombreCliente'] = consolidado_pj['NombreCliente'].apply(lambda x: None if x == 'nan' else x)
consolidado_pj['NombreCliente'] = consolidado_pj['NombreCliente'].apply(lambda x: None if x == 'NULL' else x)
consolidado_pj['NombreCliente'] = consolidado_pj['NombreCliente'].apply(lambda x: None if x == 'null' else x)

consolidado_pj['DireccionActual'] = consolidado_pj['DireccionActual'].astype(str).str.strip()
consolidado_pj['DireccionActual'] = consolidado_pj['DireccionActual'].apply(lambda x: None if x == '' else x)
consolidado_pj['DireccionActual'] = consolidado_pj['DireccionActual'].apply(lambda x: None if x == 'nan' else x)
consolidado_pj['DireccionActual'] = consolidado_pj['DireccionActual'].apply(lambda x: None if x == 'NULL' else x)
consolidado_pj['DireccionActual'] = consolidado_pj['DireccionActual'].apply(lambda x: None if x == 'null' else x)
consolidado_pj['DireccionActual'] = consolidado_pj['DireccionActual'].apply(lambda x: None if x == 'NA' else x)
consolidado_pj['DireccionActual'] = consolidado_pj['DireccionActual'].apply(lambda x: None if x == 'N/A' else x)
consolidado_pj['Direccionlargo'] = consolidado_pj['DireccionActual'].astype(str).str.len()
consolidado_pj['DireccionActual_valida'] = np.where(consolidado_pj['Direccionlargo']<=6,0,1)
consolidado_pj['DireccionActual'] = np.where(consolidado_pj['DireccionActual_valida'] == 0, None, consolidado_pj['DireccionActual'])



# ajustes campo celular 


consolidado_pj['Celular'] = consolidado_pj['Celular'].fillna(0).astype('int64').astype(str).str.strip()

consolidado_pj['Celularlargo'] = consolidado_pj['Celular'].astype(str).str.len()

consolidado_pj['Celular_valido'] = np.where(consolidado_pj['Celularlargo']!=10,0,1)

consolidado_pj['Celular'] = np.where(consolidado_pj['Celular_valido'] <= 0, None, consolidado_pj['Celular'])





# Validación estructura del corrreo


def validar_correo(correo):
    if isinstance(correo, str):
        patron_correo = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(patron_correo, correo))
    else:
        return False

# validación patrones anómalos campo Correo

consolidado_pj['Correo'] = consolidado_pj['Correo'].astype(str).str.strip().str.upper()

def clean_correo(x):
    if x in ('', 'NAN', 'NULL', '000@000.COM', 'NO@NO.com', '000@000.CO'):
        return None
    return x

consolidado_pj['Correo'] = consolidado_pj['Correo'].astype(str).str.strip().apply(clean_correo)



consolidado_pj['Correo_valido'] = consolidado_pj['Correo'].apply(validar_correo)


consolidado_pj['Correo'] = np.where(consolidado_pj['Correo_valido'] == False, None, consolidado_pj['Correo'])




# valores_nulos_por_columna2 = consolidado_pj.isnull().sum().reset_index()

#valores_nulos_por_columna1 = valores_nulos_por_columna1.reset_index()

# consolidado = pd.merge(valores_nulos_por_columna1,valores_nulos_por_columna2,on='index',how='left',suffixes=('_original', '_validacion1'))

# consolidado = pd.merge(consolidado,valores_nulos_por_columna3,on='index',how='left')


# %% [9] INFORMACIÓN DEMOGRÁFICA REPORTE EXPERIAN


'''# carga de bases demográficas para mejorar completitud de la base a entregar #'''


# Extrae el año y el mes
año_actual = dt.datetime.now().year
mes_actual = dt.datetime.now().month

# mes_actual = 6

if mes_actual < 5:
    año_actual += -1    
if mes_actual == 4:
    mes_actual = "12"
elif mes_actual == 3:
    mes_actual = "11"
elif mes_actual == 2:
    mes_actual = "10"
elif mes_actual == 1:
    mes_actual = "9"
else:
    mes_actual -= 4
    
cierre_exp = int(str(año_actual) + str(mes_actual).zfill(2))



experian = f'''select CAST([nid] AS VARCHAR(255)) as DocumentoCliente, ingresos as MontoIngresos, nombrecliente as NombreCliente, ciudad as CiudadActual, direccion1 as DireccionActual, celular1 as Celular, email1 as Correo, [Deuda financiera-TodasObligFinanc] AS Pasivos, CuotaPotencial AS Egresos, cierre, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
FROM fabogcubox.[Finandina_cartera].dbo.[011 BaseSegmentacionDinamica 201911>]
WHERE cierre >= {cierre_exp} '''

experian = cx.read_sql(conn = sql_connection, query = experian, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])




indices_maximos = experian.groupby('DocumentoCliente')['cierre'].idxmax()

# Filtra el DataFrame original utilizando los índices obtenidos
experian = experian.loc[indices_maximos].drop(columns='cierre')

# experian['Celular'] = experian['Celular'].astype(int).astype(str).str.strip()
# experian['Celular'] = experian['Celular'].fillna(0).astype(int).astype(str).str.strip()
experian['Celular'] = experian['Celular'].fillna(0).astype('int64').astype(str).str.strip()

experian['DocumentoCliente'] = experian['DocumentoCliente'].astype(int).astype(str).str.strip()
experian['CiudadActual'] = experian['CiudadActual'].astype(str).str.strip()
experian['MontoIngresos'] = experian['MontoIngresos'].astype(int)
experian['NombreCliente'] = experian['NombreCliente'].astype(str).str.strip()
# jose=experian.head(100)


consolidado_pj = consolidado_pj.merge(experian, on='DocumentoCliente', how='left', suffixes=('', '_experian'))
consolidado_pj['Celular_experian'] = consolidado_pj['Celular_experian'].apply(lambda x: None if x == '0' else x)

consolidado_pj['Nombre'] = consolidado_pj['Nombre'].apply(lambda x: None if x == 0 else x)



'''# mejora de completitud de información con consultas experian últimos 5 meses   #'''

consolidado_pj['NombreCliente'] = np.where((consolidado_pj['NombreCliente'].isnull()), consolidado_pj['Nombre'], consolidado_pj['NombreCliente'])
consolidado_pj['NombreCliente'] = np.where((consolidado_pj['NombreCliente'].isnull()), consolidado_pj['NombreCliente_experian'], consolidado_pj['NombreCliente'])
consolidado_pj['CiudadActual'] = np.where((consolidado_pj['CiudadActual'].isnull()), consolidado_pj['CiudadActual_experian'], consolidado_pj['CiudadActual'])
consolidado_pj['Celular'] = np.where((consolidado_pj['Celular'].isnull()) | (consolidado_pj['Celular'] == '0'), consolidado_pj['Celular_experian'], consolidado_pj['Celular'])
consolidado_pj['Correo'] = np.where((consolidado_pj['Correo'].isnull()), consolidado_pj['Correo_experian'], consolidado_pj['Correo'])
consolidado_pj['DireccionActual'] = np.where((consolidado_pj['DireccionActual'].isnull()), consolidado_pj['DireccionActual_experian'], consolidado_pj['DireccionActual'])
consolidado_pj['MontoIngresos'] = np.where((consolidado_pj['MontoIngresos'].isnull()) | (abs(consolidado_pj['MontoIngresos'] - consolidado_pj['MontoIngresos_experian']) >= 1000000 ), consolidado_pj['MontoIngresos_experian'], consolidado_pj['MontoIngresos'])

columnas_a_eliminar_experian = [columna for columna in consolidado_pj.columns if 'experian' in columna]
consolidado_pj = consolidado_pj.drop(columns=columnas_a_eliminar_experian)
consolidado_pj['FechaVinculacion'] = pd.to_datetime(consolidado_pj['FechaVinculacion'])

# segundo ajuste 

consolidado_pj['FechaApertura'] = np.where((consolidado_pj['FechaApertura'].isnull()), consolidado_pj['FechaVinculacion'], consolidado_pj['FechaApertura'])

'''# mejora de completitud de información   #'''


consolidado_pj = consolidado_pj.merge(demografico, on='DocumentoCliente', how='left', suffixes=('', '_actualizado'))
consolidado_pj['CiudadActual']=consolidado_pj['CiudadActual'].str.strip()


columnas_a_actualizar = ['DireccionActual', 'CiudadActual', 'Celular', 'FechaVinculacion', 'MontoIngresos', 'Correo', 'FechaUltimaActualizacionCore', 'NombreCliente']

actualizar_columnas(consolidado_pj, columnas_a_actualizar)
columnas_a_eliminar = [columna for columna in consolidado_pj.columns if 'actualizado' in columna]

consolidado_pj = consolidado_pj.drop(columns=columnas_a_eliminar)


# valores_nulos_por_columna = consolidado_pj.isnull().sum().reset_index()




## validacion con informacion de AGIL y LP




agil_lp = '''select *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank 
from [ModelosSARLAFT].[dbo].[Informacion_Cliente_AGIL_LP] '''
agil_lp = cx.read_sql(conn=sql_connection, query=agil_lp, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])


agil_lp['DocumentoCliente'] = agil_lp['DocumentoCliente'].astype(str).str.strip()

agil_lp = agil_lp.groupby('DocumentoCliente').first().reset_index()


consolidado = pd.merge(consolidado_pj,agil_lp,on='DocumentoCliente',how='left',suffixes=('', '_validacion_agil_lp')) # quitar el agil lp
consolidado['Correo'] = np.where(consolidado['Correo'].isnull(),consolidado['CorreoOficina'],consolidado['Correo'])
consolidado['Celular'] = np.where(consolidado['Celular'].isnull(),consolidado['Celular_validacion_agil_lp'],consolidado['Celular'])
consolidado['Celular'] = np.where(consolidado['Celular'].isnull(),consolidado['TelefonoOficina'],consolidado['Celular'])
consolidado['Tipo_identificacion'] = np.where(consolidado['Tipo_identificacion'].isnull(),consolidado['TipoDocumento'],consolidado['Tipo_identificacion'])
consolidado['DireccionActual'] = np.where(consolidado['DireccionActual'].isnull(),consolidado['DireccionResidencia'],consolidado['DireccionActual'])
consolidado['DireccionActual'] = np.where(consolidado['DireccionActual'].isnull(),consolidado['DireccionOficina'],consolidado['DireccionActual'])
consolidado['Nombre'] = np.where(consolidado['Nombre'].isnull(),consolidado['Nombre_validacion_agil_lp'],consolidado['Nombre'])
consolidado['MontoIngresos'] = np.where(consolidado['MontoIngresos'].isnull(),consolidado['TotalIngresosMensuales'],consolidado['MontoIngresos'])
consolidado['Egresos'] = np.where(consolidado['Egresos'].isnull(),consolidado['TotalEgresosMensuales'],consolidado['Egresos'])
consolidado['CodigoCIIU'] = np.where(consolidado['CodigoCIIU'].isnull(),consolidado['CIUU'],consolidado['CodigoCIIU'])
consolidado['Departamento'] = np.where(consolidado['Departamento'].isnull(),consolidado['Departamento_validacion_agil_lp'],consolidado['Departamento'])


# list(consolidado.columns)
# Out[57]: 
# ['DocumentoCliente',
#  'Nombre',
#  'Tipo_identificacion',
#  'Incomercio',
#  'Ahorro de la red',
#  'CDT',
#  'Cartera vendida',
#  'Castigados',
#  'Crédito hipotecario',
#  'Crédito libre inversión',
#  'Crédito vehículo',
#  'Cuenta corriente',
#  'FlexiDigital',
#  'Leasing vehículo',
#  'Libranza',
#  'Maquina agrícola',
#  'Nomina Finandina',
#  'Otros activos',
#  'Otros ahorros',
#  'Plan mayor',
#  'Redescuentos',
#  'TDC Digital',
#  'TDC Física',
#  'FechaApertura',
#  'NombreCliente',
#  'DireccionActual',
#  'Celular',
#  'FechaVinculacion',
#  'MontoIngresos',
#  'Correo',
#  'FechaUltimaActualizacionCore',
#  'CMCODICBS',
#  'CiudadActual',
#  'Departamento',
#  'Municipio',
#  'Direccionlargo',
#  'DireccionActual_valida',
#  'Celularlargo',
#  'Celular_valido',
#  'Correo_valido',
#  'Pasivos',
#  'Egresos',
#  'TipoPersona',
#  'CodigoCIIU',
#  'ActividadEconomica',
#  'facta',
#  'TotalActivo',
#  'TotalPasivo',
#  'TotalPatrimonio',
#  'TipoDocumento',
#  'Nombre_validacion_agil_lp',
#  'Celular_validacion_agil_lp',
#  'Correo_validacion_agil_lp',
#  'TelefonoResidencia',
#  'DireccionResidencia',
#  'DireccionOficina',
#  'TotalIngresosMensuales',
#  'TotalEgresosMensuales',
#  'Empresa',
#  'TelefonoOficina',
#  'CIUU',
#  'Departamento_validacion_agil_lp',
#  'CorreoOficina',
#  'FechaExpedicion',
#  'FechaNacimiento']







# actualizacion de campos 









# jose = consolidado[consolidado['TotalActivo'].notnull()]


# consolidado = pd.merge(valores_nulos_por_columna1,valores_nulos_por_columna2,on='index',how='left',suffixes=('_original', '_validacion1'))

# consolidado = pd.merge(consolidado,valores_nulos_por_columna3,on='index',how='left',suffixes=('', '_validacion_experian'))

# ruta_completa_archivo_excel = r'C:\Users\josgom\Desktop\NOBORRAR\calidaddatosconbodega.xlsx'

# # Exporta el DataFrame a un archivo Excel en la ruta especificada
# consolidado.to_excel(ruta_completa_archivo_excel, index=False)



consolidado_pj['CiudadActual']=consolidado_pj['CiudadActual'].str.rstrip().str.upper()



consolidado_pj['NombreCliente'] = consolidado_pj['NombreCliente'].apply(lambda x: None if x == 'null' else x)

# %% [10] HOMOLOGACIÓN INFORMACIÓN DEPARTAMENTAL



# consolidado_pj['Departamento'].isnull().sum()
# jose = (consolidado_pj[consolidado_pj['Departamento'].isnull()])[['CiudadActual','Departamento', 'Municipio']]


'''# homologación Departamentos   #'''

asignar_departamento(consolidado_pj, 'CiudadActual', 'CHAPINERO ALTO', 'BOGOTA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO WILCHES', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA MONTANITA', 'CAQUETA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BOGOTA, D.C.', 'BOGOTA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'QUIMABAYA', 'QUINDIO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'QUIMABAYA', 'QUINDIO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BOGOTA', 'BOGOTA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BOGOTA DC', 'BOGOTA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CALI', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CALI (VALLE)', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CARTAGO', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTIAGO DE CALI', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BUGA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'YUMBO', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TULUA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BUENAVENTURA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PALMIRA (VALLE)', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TULUA (VALLE)', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GUADALAJARA DE BUGA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ROLDANILLO', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ANSERMANUEVO', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BOSCONIA', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PELAYA', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTIAGO DE TOLU', 'SUCRE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ACANDI', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CERTEGUI', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ATRATO', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO ASIS', 'PUTUMAYO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TUMACO', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GUACARI', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA UNION', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'JAMUNDI', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PALMIRA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GUAMAL', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'YOPAL', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SOMONDOCO', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BELLO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MEDELLIN', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SABANETA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'RIONEGRO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SOPETRAN', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'COCORNA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CANASGORDAS', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CAUCASIA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'EL CARMEN DE VIBORAL', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'NARINO', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO ALEGRIA', 'AMAZONAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MANIZALES', 'CALDAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VILLAVICENCIO', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'V/CIO', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'RESTREPO META', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VILLAVICENCIO (META)', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ACACIAS (META)', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN CARLOS DE GUAROA', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TUNUNGUA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ROVIRA', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CARTAGENA', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BUCARAMANGA', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LEBRIJA', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN GIL', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'COTA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ITAGUI', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BARRANQUILLA', 'ATLANTICO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SABANALARGA', 'ATLANTICO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'NEIVA', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'HUILA', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TESALIA', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SUAZA', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PITALITO', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ALTAMIRA', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GARZON', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PACARNI', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PITALITO (HUILA)', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PASTO', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'IPIALES', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PLATO (MAGDALENA)', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ARACATACA', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ALGARROBO', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO BERRIO', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ZONA BANANERA', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CIENAGA', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'NUEVA GRANADA', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CHIVOLO', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BERBEO', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTANDER DE QUILICH', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTANDER DE QUILICHAO', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JOSE DEL GUAVIARE', 'GUAVIARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MIRAFLORES', 'GUAVIARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TAMARA', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'OROCUE', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'AGUAZUL', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'NARI#O', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JUAN DE PASTO', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PEREIRA', 'RISARALDA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ABEJORRAL', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CONCORDIA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA ESTRELLA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTA MARTA', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PRADO  SEVILLA', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO LEGUIZAMO', 'PUTUMAYO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ENVIGADO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GUARNE', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA ESTRELLA -ANTIOQUIA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TOLEDO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CACERES', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTAFE DE ANTIOQUIA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'EL SANTUARIO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GIRARDOTA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FLORIDABLANCA', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'IBAGUE', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CARMEN DE APICALA', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CALDAS', 'CALDAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MELGAR', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CORINTO', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MINGUEO', 'LA GUAJIRA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ZIPAQUIRA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'DUITAMA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FUNZA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FOMEQUE', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CHIA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CAJICA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FUSAGASUGA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'AGUA DE DIOS', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FACATATIVA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TOCAIMA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GUTIERREZ', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'UBAQUE', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'COGUA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TENJO', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LENGUAZAQUE', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GIRARDOT', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'UBATE', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SUESCA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SOACHA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SOPO', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CHOCONTA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TOCANCIPA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GACHANCIPA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ANAPOIMA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'EL ROSAL', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ANOLAIMA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JUAN DE RIOSECO', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MOSQUERA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PULI', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA CALERA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CHALAN', 'SUCRE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PORE', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SUPIA', 'CALDAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA DORADA', 'CALDAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'JAMBALO', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ARAUCA', 'ARAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ISTMINA', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'QUIBDO', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN FRANCISCO DE QUIBDO', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MEDIO BAUDO', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'EL LITORAL DEL SAN J', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'QUIMBAYA', 'QUINDIO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SINCELEJO', 'SUCRE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MISTRATO', 'RISARALDA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA PALMA (CUND)', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO BOYACA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'RAQUIRA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BETEITIVA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MONIQUIRA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GUAMO', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LEIVA', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SALENTO', 'QUINDIO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ARMENIA', 'QUINDIO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CIRCASIA', 'QUINDIO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'POPAYAN', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'Yopal (Casanare)', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CUCUTA', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TIBU', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PAMPLONITA', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SARDINATA', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN CALIXTO', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VILLA DEL ROSARIO', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MUTISCUA', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CONTADERO', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BOLIVAR', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'DIBULLA', 'LA GUAJIRA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BARRANCABERMEJA', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SABANA DE TORRES', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'DOSQUEBRADAS', 'RISARALDA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ORITO', 'PUTUMAYO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'RIOHACHA', 'LA GUAJIRA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GIRON', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ACACIAS', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CUBARRAL', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TUNJA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PAIPA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MONGUI', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO COLOMBIA', 'ATLANTICO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SALDANA', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'AYAPEL', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CHINU', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MONITOS', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO LIBERTADOR', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'COTORRA', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA APARTADA', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CANALETE', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTA CRUZ DE LORICA', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MONTELIBANO', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VILLANUEVA', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PALERMO', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ALVARADO', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FLANDES', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'EL ESPINAL', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'COYAIMA', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CAJIBIO', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ARGELIA', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CHIRIGUANA', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'AGUACHICA', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VALLEDUPAR', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'AGUSTIN CODAZZI', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUEBLO BELLO', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PIEDECUESTA', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TONA', 'SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA JAGUA D IBIRICO', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA JAGUA DE IBIRICO', 'CESAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CARTAGENA DE INDIAS', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CANTAGALLO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TIQUISIO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ALTOS DEL ROSARIO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JACINTO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CASCAJAL', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'RIO VIEJO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TURBACO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TALAIGUA NUEVO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JUAN NEPOMUCENO', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TURBANA', 'BOLIVAR')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CARTAGENA DEL CHAIRA', 'CAQUETA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FLORENCIA', 'CAQUETA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ALBANIA', 'CAQUETA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JOSE DEL FRAGUA', 'CAQUETA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LETICIA', 'AMAZONAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SARAVENA', 'ARAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TAME', 'ARAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MONTERIA', 'CORDOBA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SUAREZ', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MERCADERES', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CALARCA', 'QUINDIO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTA FE DE ANTIOQUIA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FUNDACION', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA PINTADA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUERTO GAITAN', 'META')
asignar_departamento(consolidado_pj, 'CiudadActual', 'POLICARPA', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CHACHAGI', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TUQUERRES', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ANCUYA', 'NARINO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SOTARA', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTUARIO', 'RISARALDA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TOLU VIEJO', 'SUCRE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PRADERA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'FLORIDA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VERGARA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VILLAMARIA', 'CALDAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MARMATO', 'CALDAS')
asignar_departamento(consolidado_pj, 'CiudadActual', 'GINEBRA', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JERONIMO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'COPACABANA', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUEBLORRICO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'NECOCLI', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'YUTO', 'CHOCO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'ZARZAL', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'EL CERRITO', 'VALLE DEL CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PURIFICACION', 'TOLIMA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SANTA ROSA DE CABAL', 'RISARALDA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'AMALFI', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SOGAMOSO', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SAN JUAN DE MOMBITA', 'BOYACA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'TARQUI', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'YONDO', 'ANTIOQUIA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BELEN DE UMBRIA', 'RISARALDA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PUEBLOVIEJO', 'MAGDALENA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MIRANDA', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'PATIA', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'BUENOS AIRES', 'CAUCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'LA PLATA', 'HUILA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'SILOS', 'NORTE DE SANTANDER')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MEDINA', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MADRID', 'CUNDINAMARCA')
asignar_departamento(consolidado_pj, 'CiudadActual', 'CANDELARIA', 'ATLANTICO')
asignar_departamento(consolidado_pj, 'CiudadActual', 'MONTERREY', 'CASANARE')
asignar_departamento(consolidado_pj, 'CiudadActual', 'VILLA RICA', 'TOLIMA')



# jose = (consolidado_pj[consolidado_pj['Departamento'].isnull()])[['CiudadActual','Departamento', 'Municipio']]


# prueba=consolidado_pj.head(100)


# %% [11] FACTOR DE RIESGO CANAL


''' # AGREGAR CANAL DE ENTRADA #'''

canal_entrada_pj = '''select DocumentoCliente
	          ,DescripcionSucursal
	          ,CASE WHEN DescripcionSucursal like '%DIRECCION GENERAL%' THEN 'Oficina'
					WHEN DescripcionSucursal like '%DIRECTO%' THEN 'Oficina'
					WHEN DescripcionSucursal LIKE '%Oficina%' THEN 'Oficina'
					WHEN DescripcionSucursal LIKE '%CENTRO ANDINO%' THEN 'Oficina'
					WHEN DescripcionSucursal like '%Concesionario%' THEN 'Concesionario'
					WHEN DescripcionSucursal like '%CORRETAJE%' THEN 'Concesionario'
					WHEN DescripcionSucursal LIKE '%APP%' THEN 'APP'			   	    
					WHEN DescripcionSucursal like '%canal especializado%' THEN 'CVD'
			   	    WHEN DescripcionSucursal like '%Fuerza%' THEN 'CVD'     
			   	    WHEN DescripcionSucursal like '%FMV%' THEN 'CVD'   
			   	    WHEN DescripcionSucursal like '%SUCURSAL VIRTUAL%' THEN 'Internet'  
			   	    WHEN DescripcionSucursal like '%ATM TELLER DEFAULT%' THEN 'Internet'
					WHEN DescripcionSucursal IS NULL THEN 'Extremo'
		       ELSE 'Oficina' END CanalEntrada,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
 FROM [Productos y transaccionalidad].[dbo].[DemografiaJuridicaCore]'''		 

canal_entrada_pj = (cx.read_sql(conn = sql_connection, query = canal_entrada_pj, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])


canal_entrada_pn = '''select DocumentoCliente
	          ,DescripcionSucursal
	          ,CASE WHEN DescripcionSucursal like '%DIRECCION GENERAL%' THEN 'Oficina'
					WHEN DescripcionSucursal like '%DIRECTO%' THEN 'Oficina'
					WHEN DescripcionSucursal LIKE '%Oficina%' THEN 'Oficina'
					WHEN DescripcionSucursal LIKE '%CENTRO ANDINO%' THEN 'Oficina'
					WHEN DescripcionSucursal like '%Concesionario%' THEN 'Concesionario'
					WHEN DescripcionSucursal like '%CORRETAJE%' THEN 'Concesionario'
					WHEN DescripcionSucursal LIKE '%APP%' THEN 'APP'			   	    
					WHEN DescripcionSucursal like '%canal especializado%' THEN 'CVD'
			   	    WHEN DescripcionSucursal like '%Fuerza%' THEN 'CVD'     
			   	    WHEN DescripcionSucursal like '%FMV%' THEN 'CVD'   
			   	    WHEN DescripcionSucursal like '%SUCURSAL VIRTUAL%' THEN 'Internet'  
			   	    WHEN DescripcionSucursal like '%ATM TELLER DEFAULT%' THEN 'Internet'
					WHEN DescripcionSucursal IS NULL THEN 'Extremo'
		       ELSE 'Oficina' END CanalEntrada,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
 FROM [Productos y transaccionalidad].[dbo].[DemografiaNaturalCore]'''		 

canal_entrada_pn = (cx.read_sql(conn = sql_connection, query = canal_entrada_pn, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])

consolidado_canal_entrada = pd.concat([canal_entrada_pj, canal_entrada_pn], axis=0).groupby('DocumentoCliente').first().reset_index()

consolidado_canal_entrada['DocumentoCliente'] = consolidado_canal_entrada['DocumentoCliente'].astype(int).astype(str).str.strip()
consolidado_canal_entrada = consolidado_canal_entrada.drop_duplicates(subset=['DocumentoCliente'])

# prueba = canal_entrada_pn[canal_entrada_pn['DocumentoCliente'].astype(str).str.contains('52225830')]
# prueba['DocumentoCliente'] = prueba['DocumentoCliente'].astype(str).str.rstrip('.0')
# df_filtrado = consolidado_canal_entrada.loc[consolidado_canal_entrada['DocumentoCliente'].isin(['1020836017', '1002182815', '52225830','1072448136'])]

consolidado_pj = pd.merge(consolidado_pj,consolidado_canal_entrada,on='DocumentoCliente',how='left')




### consulta bodega de datos AZURE


'''# validación informacion bodega #'''

bodega_azure = '''select *
,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank FROM [ModelosSARLAFT].[dbo].[Informacion_Bodega_AZURE]'''		 

bodega_azure = (cx.read_sql(conn = sql_connection, query = bodega_azure, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])

bodega_azure['DocumentoCliente'] = bodega_azure['DocumentoCliente'].fillna(0).astype('int64').astype(str).str.strip()

bodega_azure = bodega_azure[bodega_azure['DocumentoCliente'].isin(consolidado_pj['DocumentoCliente'])]

bodega_azure = bodega_azure.groupby('DocumentoCliente').first().reset_index()
bodega_azure['FechaInicioCliente'] = pd.to_datetime(bodega_azure['FechaInicioCliente'])

# ajuste campos nombre, celular,correo,fechavinculacion,ingresos,direccionactual

consolidado_pj=pd.merge(consolidado_pj,bodega_azure,on='DocumentoCliente',how='left')


# ajustes nombre

consolidado_pj['NombreCliente'] = np.where((consolidado_pj['NombreCliente'].isnull()), consolidado_pj['NombreCliente_bodega'], consolidado_pj['NombreCliente'])
consolidado_pj['CiudadActual'] = np.where((consolidado_pj['CiudadActual'].isnull()), consolidado_pj['CiudadActual_bodega'], consolidado_pj['CiudadActual'])
consolidado_pj['Celular'] = np.where((consolidado_pj['Celular'].isnull()) | (consolidado_pj['Celular'] == '0'), consolidado_pj['telefonoContacto'], consolidado_pj['Celular'])
consolidado_pj['Correo'] = np.where((consolidado_pj['Correo'].isnull()), consolidado_pj['emailContacto'], consolidado_pj['Correo'])
consolidado_pj['DireccionActual'] = np.where((consolidado_pj['DireccionActual'].isnull()), consolidado_pj['direccionContacto'], consolidado_pj['DireccionActual'])
consolidado_pj['MontoIngresos'] = np.where((consolidado_pj['MontoIngresos'].isnull()) | (abs(consolidado_pj['MontoIngresos'] - consolidado_pj['MontoIngresos_bodega']) >= 1000000 ), consolidado_pj['MontoIngresos_bodega'], consolidado_pj['MontoIngresos'])
consolidado_pj['FechaApertura'] = np.where((consolidado_pj['FechaApertura'].isnull()), consolidado_pj['FechaInicioCliente'], consolidado_pj['FechaApertura'])


consolidado_pj['Pasivos'] = np.where((consolidado_pj['Pasivos'].isnull()), consolidado_pj['valorTotalPasivos'], consolidado_pj['Pasivos'])
consolidado_pj['Egresos'] = np.where((consolidado_pj['Egresos'].isnull()), consolidado_pj['valorTotalEgresosMensuales'], consolidado_pj['Egresos'])
consolidado_pj['CodigoCIIU'] = np.where((consolidado_pj['CodigoCIIU'].isnull()), consolidado_pj['actividadEconomicacodigoCiiuId'], consolidado_pj['CodigoCIIU'])
consolidado_pj['CanalEntrada'] = np.where((consolidado_pj['CanalEntrada'].isnull()), consolidado_pj['canalDesc'], consolidado_pj['CanalEntrada'])
consolidado_pj['CiudadActual'] = np.where((consolidado_pj['CiudadActual'].isnull()), consolidado_pj['CiudadActual_bodega'], consolidado_pj['CiudadActual'])
consolidado_pj['Departamento'] = np.where((consolidado_pj['Departamento'].isnull()), consolidado_pj['Departamento_bodega'], consolidado_pj['Departamento'])


# valores_nulos_por_columna4 = consolidado_pj.isnull().sum().reset_index()


# consolidado = pd.merge(valores_nulos_por_columna1,valores_nulos_por_columna2,on='index',how='left',suffixes=('_original', '_validacion1'))

# consolidado = pd.merge(consolidado,valores_nulos_por_columna4,on='index',how='left',suffixes=('', '_validacion_bodega'))




def limpiar_nombre(nombre):
    # Retirar números al principio del nombre utilizando expresiones regulares
    nombre_sin_numeros = re.sub(r'^\d+', '', nombre)

    # Capitalizar la primera letra de cada palabra
    nombre_formateado = ' '.join(word.capitalize() for word in nombre_sin_numeros.split())

    return nombre_formateado

# Aplicar la función a la columna 'Nombre'
consolidado_pj['NombreCliente'] = consolidado_pj['NombreCliente'].astype(str).apply(limpiar_nombre)



# %% [12] INFORMACIÓN TRANSACCIONAL ENTRADAS

'''# validación información transaccional Corta (1 mes), Media (3 meses) y Larga (6 meses) por producto entrada #'''


# Obtén la fecha del último día del mes anterior
fecha_actual = datetime.now()
primer_dia_mes_actual = fecha_actual.replace(day=1)
ultimo_dia_mes_anterior = (primer_dia_mes_actual - timedelta(days=1))

# Calcula las fechas para 1 mes antes, 3 meses antes y 6 meses antes
fecha_un_mes_antes = (ultimo_dia_mes_anterior - timedelta(days=30)) # Aproximadamente 30 días por mes
fecha_tres_meses_antes = ultimo_dia_mes_anterior - timedelta(days=90)  # Aproximadamente 90 días por 3 meses
fecha_seis_meses_antes = ultimo_dia_mes_anterior - timedelta(days=180)  # Aproximadamente 180 días por 6 meses

ultimo_dia_mes_anterior = ultimo_dia_mes_anterior.strftime("%Y-%m-%d")
fecha_un_mes_antes = fecha_un_mes_antes.strftime("%Y-%m-%d")
fecha_tres_meses_antes = fecha_tres_meses_antes.strftime("%Y-%m-%d")
fecha_seis_meses_antes = fecha_seis_meses_antes.strftime("%Y-%m-%d")


query_tx_ahorro_entrada = '''select DocumentoCliente,FechaTransaccionEfectiva,MontoTransaccion,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesAhorro]
where CaracterTransaccion = 'Entrada'
and DescripcionTransaccional4 
in ('DEPOSITO DE CUENTA DE AHORROS SIN LIBRETA',
'DEPOSITO DEL CLIENTE',
'MEMO DE CREDITO') '''
ahorro_entradas = cx.read_sql(conn=sql_connection, query=query_tx_ahorro_entrada, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])




ahorro_entradas['FechaTransaccionEfectiva'] = ahorro_entradas['FechaTransaccionEfectiva'].astype(int).astype(str).str.strip()

ahorro_entradas['DocumentoCliente'] = ahorro_entradas['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

ahorro_entradas['FechaTransaccionEfectiva'] = ahorro_entradas['FechaTransaccionEfectiva'].apply(convertir_fecha_juliana)



ahorro_entrada_corta = ahorro_entradas[(ahorro_entradas['FechaTransaccionEfectiva'] >= fecha_un_mes_antes) & (ahorro_entradas['FechaTransaccionEfectiva'] <= ultimo_dia_mes_anterior)]

ahorro_entrada_media = ahorro_entradas[(ahorro_entradas['FechaTransaccionEfectiva'] >= fecha_tres_meses_antes) & (ahorro_entradas['FechaTransaccionEfectiva'] <= ultimo_dia_mes_anterior)]

ahorro_entrada_larga = ahorro_entradas[(ahorro_entradas['FechaTransaccionEfectiva'] >= fecha_seis_meses_antes) & (ahorro_entradas['FechaTransaccionEfectiva'] <= ultimo_dia_mes_anterior)]


 

resultados_entradas_corta_ahorro = ahorro_entrada_corta.groupby('DocumentoCliente').agg(
    MontoEntradasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='sum'),
    CantidadEntradasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='count'),
    MediaEntradasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='mean'),
    MedianaEntradasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='median')
).reset_index()


resultados_entradas_media_ahorro = ahorro_entrada_media.groupby('DocumentoCliente').agg(
    MontoEntradasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='sum'),
    CantidadEntradasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='count'),
    MediaEntradasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='mean'),
    MedianaEntradasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='median')
).reset_index()

resultados_entradas_larga_ahorro = ahorro_entrada_larga.groupby('DocumentoCliente').agg(
    MontoEntradasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='sum'),
    CantidadEntradasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='count'),
    MediaEntradasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='mean'),
    MedianaEntradasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='median')
).reset_index()

Entradas_ahorro = pd.merge(pd.merge(resultados_entradas_larga_ahorro,resultados_entradas_media_ahorro, on = 'DocumentoCliente',how='left'),resultados_entradas_corta_ahorro,on = 'DocumentoCliente',how='left')




query_tx_activo_entrada = '''select DocumentoCliente,FechaPublicacion,MontoCapital,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesActivo]
where CaracterTransaccion = 'Entrada' '''
activo_entradas = cx.read_sql(conn=sql_connection, query=query_tx_activo_entrada, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])




activo_entradas['FechaPublicacion'] = activo_entradas['FechaPublicacion'].astype(int).astype(str).str.strip()

activo_entradas['DocumentoCliente'] = activo_entradas['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

activo_entradas['FechaPublicacion'] = activo_entradas['FechaPublicacion'].apply(convertir_fecha_juliana)



activo_entrada_corta = activo_entradas[(activo_entradas['FechaPublicacion'] >= fecha_un_mes_antes) & (activo_entradas['FechaPublicacion'] <= ultimo_dia_mes_anterior)]

activo_entrada_media = activo_entradas[(activo_entradas['FechaPublicacion'] >= fecha_tres_meses_antes) & (activo_entradas['FechaPublicacion'] <= ultimo_dia_mes_anterior)]

activo_entrada_larga = activo_entradas[(activo_entradas['FechaPublicacion'] >= fecha_seis_meses_antes) & (activo_entradas['FechaPublicacion'] <= ultimo_dia_mes_anterior)]





resultados_entradas_corta_activo = activo_entrada_corta.groupby('DocumentoCliente').agg(
    MontoEntradasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='sum'),
    CantidadEntradasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='count'),
    MediaEntradasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='mean'),
    MedianaEntradasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='median')
).reset_index()


resultados_entradas_media_activo = activo_entrada_media.groupby('DocumentoCliente').agg(
    MontoEntradasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='sum'),
    CantidadEntradasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='count'),
    MediaEntradasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='mean'),
    MedianaEntradasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='median')
).reset_index()

resultados_entradas_larga_activo = activo_entrada_larga.groupby('DocumentoCliente').agg(
    MontoEntradasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='sum'),
    CantidadEntradasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='count'),
    MediaEntradasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='mean'),
    MedianaEntradasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='median')
).reset_index()

Entradas_activo = pd.merge(pd.merge(resultados_entradas_larga_activo,resultados_entradas_media_activo, on = 'DocumentoCliente',how='left'),resultados_entradas_corta_activo,on = 'DocumentoCliente',how='left')




query_tx_cdt_entrada = '''select DocumentoCliente,FechaEfectiva,[MontoTransaccion1],ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesCDT]
where CaracterTransaccion = 'Entrada' '''
cdt_entradas = cx.read_sql(conn=sql_connection, query=query_tx_cdt_entrada, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])



cdt_entradas['FechaEfectiva'] = cdt_entradas['FechaEfectiva'].astype(int).astype(str).str.strip()

cdt_entradas['DocumentoCliente'] = cdt_entradas['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

cdt_entradas['FechaEfectiva'] = cdt_entradas['FechaEfectiva'].apply(convertir_fecha_juliana)



cdt_entrada_corta = cdt_entradas[(cdt_entradas['FechaEfectiva'] >= fecha_un_mes_antes) & (cdt_entradas['FechaEfectiva'] <= ultimo_dia_mes_anterior)]

cdt_entrada_media = cdt_entradas[(cdt_entradas['FechaEfectiva'] >= fecha_tres_meses_antes) & (cdt_entradas['FechaEfectiva'] <= ultimo_dia_mes_anterior)]

cdt_entrada_larga = cdt_entradas[(cdt_entradas['FechaEfectiva'] >= fecha_seis_meses_antes) & (cdt_entradas['FechaEfectiva'] <= ultimo_dia_mes_anterior)]


resultados_entradas_corta_cdt = cdt_entrada_corta.groupby('DocumentoCliente').agg(
    MontoEntradasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='sum'),
    CantidadEntradasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='count'),
    MediaEntradasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='mean'),
    MedianaEntradasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='median')
).reset_index()


resultados_entradas_media_cdt = cdt_entrada_media.groupby('DocumentoCliente').agg(
    MontoEntradasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='sum'),
    CantidadEntradasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='count'),
    MediaEntradasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='mean'),
    MedianaEntradasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='median')
).reset_index()

resultados_entradas_larga_cdt = cdt_entrada_larga.groupby('DocumentoCliente').agg(
    MontoEntradasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='sum'),
    CantidadEntradasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='count'),
    MediaEntradasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='mean'),
    MedianaEntradasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='median')
).reset_index()

Entradas_cdt = pd.merge(pd.merge(resultados_entradas_larga_cdt,resultados_entradas_media_cdt, on = 'DocumentoCliente',how='left'),resultados_entradas_corta_cdt,on = 'DocumentoCliente',how='left')


## entradas TDC 



# query_tx_tdc_entrada = f'''SELECT *    FROM OPENQUERY(DB2400_182,' select PersonalTDC.NUMDOC, TransaccionesTDC.IMPFAC,TransaccionesTDC.FECFAC,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
#   from INTTARCRE.SATMOVEXT as TransaccionesTDC
#  								 left join INTTARCRE.SATTARJET as TarjetaTDC
#  								 ON TransaccionesTDC.CUENTAME=TarjetaTDC.CUENTA and
# 								    TransaccionesTDC.PANME=TarjetaTDC.PAN
#  								 left join INTTARCRE.SATBENEFI as CuentaTDC
# 							     on TarjetaTDC.CUENTA=CuentaTDC.CUENTA and
# 								    TarjetaTDC.NUMBENCTA=CuentaTDC.NUMBENCTA
#  								 left join INTTARCRE.SATDACOPE as PersonalTDC
#  								 on CuentaTDC.IDENTCLI=PersonalTDC.IDENTCLI 
#  								 where TransaccionesTDC.TIPOFAC in (''67'',''253'')  and TransaccionesTDC.FECFAC >= ''{fecha_seis_meses_antes}'' and PersonalTDC.NUMDOC is NOT NULL
#  								 ' ) '''
# tdc_entradas = (cx.read_sql(conn=sql_connection, query=query_tx_tdc_entrada, partition_on='Rank', partition_num=2, return_type='pandas')).rename(columns={'NUMDOC':'DocumentoCliente'})




query_tx_tdc_entrada = f'''SELECT *    FROM OPENQUERY(DB2400_182,' select PersonalTDC.NUMDOC, TransaccionesTDC.IMPFAC,TransaccionesTDC.FECFAC
  from INTTARCRE.SATMOVEXT as TransaccionesTDC
 								 left join INTTARCRE.SATTARJET as TarjetaTDC
 								 ON TransaccionesTDC.CUENTAME=TarjetaTDC.CUENTA and
								    TransaccionesTDC.PANME=TarjetaTDC.PAN
 								 left join INTTARCRE.SATBENEFI as CuentaTDC
							     on TarjetaTDC.CUENTA=CuentaTDC.CUENTA and
								    TarjetaTDC.NUMBENCTA=CuentaTDC.NUMBENCTA
 								 left join INTTARCRE.SATDACOPE as PersonalTDC
 								 on CuentaTDC.IDENTCLI=PersonalTDC.IDENTCLI 
 								 where TransaccionesTDC.TIPOFAC in (''67'',''253'')  and TransaccionesTDC.FECFAC >= ''{fecha_seis_meses_antes}'' and PersonalTDC.NUMDOC is NOT NULL
 								 ' ) '''
tdc_entradas = (cargue_openquery(conn, query_tx_tdc_entrada)).rename(columns={'NUMDOC':'DocumentoCliente'})



tdc_entradas['FECFAC'] = (pd.to_datetime(tdc_entradas['FECFAC'])).dt.strftime("%Y-%m-%d")
tdc_entradas['DocumentoCliente'] = tdc_entradas['DocumentoCliente'].astype(int).astype(str).str.strip()



tdc_entrada_corta = tdc_entradas[(tdc_entradas['FECFAC'] >= fecha_un_mes_antes) & (tdc_entradas['FECFAC'] <= ultimo_dia_mes_anterior)]

tdc_entrada_media = tdc_entradas[(tdc_entradas['FECFAC'] >= fecha_tres_meses_antes) & (tdc_entradas['FECFAC'] <= ultimo_dia_mes_anterior)]

tdc_entrada_larga = tdc_entradas[(tdc_entradas['FECFAC'] >= fecha_seis_meses_antes) & (tdc_entradas['FECFAC'] <= ultimo_dia_mes_anterior)]




resultados_entradas_corta_tdc = tdc_entrada_corta.groupby('DocumentoCliente').agg(
    MontoEntradasCortatdc=pd.NamedAgg(column='IMPFAC', aggfunc='sum'),
    CantidadEntradasCortatdc=pd.NamedAgg(column='IMPFAC', aggfunc='count'),
    MediaEntradasCortatdc=pd.NamedAgg(column='IMPFAC', aggfunc='mean'),
    MedianaEntradasCortatdc=pd.NamedAgg(column='IMPFAC', aggfunc='median')
).reset_index()


resultados_entradas_media_tdc = tdc_entrada_media.groupby('DocumentoCliente').agg(
    MontoEntradasMediatdc=pd.NamedAgg(column='IMPFAC', aggfunc='sum'),
    CantidadEntradasMediatdc=pd.NamedAgg(column='IMPFAC', aggfunc='count'),
    MediaEntradasMediatdc=pd.NamedAgg(column='IMPFAC', aggfunc='mean'),
    MedianaEntradasMediatdc=pd.NamedAgg(column='IMPFAC', aggfunc='median')
).reset_index()

resultados_entradas_larga_tdc = tdc_entrada_larga.groupby('DocumentoCliente').agg(
    MontoEntradasLargatdc=pd.NamedAgg(column='IMPFAC', aggfunc='sum'),
    CantidadEntradasLargatdc=pd.NamedAgg(column='IMPFAC', aggfunc='count'),
    MediaEntradasLargatdc=pd.NamedAgg(column='IMPFAC', aggfunc='mean'),
    MedianaEntradasLargatdc=pd.NamedAgg(column='IMPFAC', aggfunc='median')
).reset_index()

Entradas_tdc = pd.merge(pd.merge(resultados_entradas_larga_tdc,resultados_entradas_media_tdc, on = 'DocumentoCliente',how='left'),resultados_entradas_corta_tdc,on = 'DocumentoCliente',how='left')



# %% [13] INFORMACIÓN TRANSACCIONAL SALIDAS

'''# validación infromación transaccional Corta (1 mes), Media (3 meses) y Larga (6 meses) por prodcuto salidas #'''



query_tx_ahorro_salida = '''select DocumentoCliente,FechaTransaccionEfectiva,MontoTransaccion,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesAhorro]
where CaracterTransaccion = 'Salida'
 '''
ahorro_salidas = cx.read_sql(conn=sql_connection, query=query_tx_ahorro_salida, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])



ahorro_salidas['FechaTransaccionEfectiva'] = ahorro_salidas['FechaTransaccionEfectiva'].astype(int).astype(str).str.strip()

ahorro_salidas['DocumentoCliente'] = ahorro_salidas['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

ahorro_salidas['FechaTransaccionEfectiva'] = ahorro_salidas['FechaTransaccionEfectiva'].apply(convertir_fecha_juliana)



ahorro_salida_corta = ahorro_salidas[(ahorro_salidas['FechaTransaccionEfectiva'] >= fecha_un_mes_antes) & (ahorro_salidas['FechaTransaccionEfectiva'] <= ultimo_dia_mes_anterior)]

ahorro_salida_media = ahorro_salidas[(ahorro_salidas['FechaTransaccionEfectiva'] >= fecha_tres_meses_antes) & (ahorro_salidas['FechaTransaccionEfectiva'] <= ultimo_dia_mes_anterior)]

ahorro_salida_larga = ahorro_salidas[(ahorro_salidas['FechaTransaccionEfectiva'] >= fecha_seis_meses_antes) & (ahorro_salidas['FechaTransaccionEfectiva'] <= ultimo_dia_mes_anterior)]


 

resultados_salidas_corta_ahorro = ahorro_salida_corta.groupby('DocumentoCliente').agg(
    MontoSalidasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='sum'),
    CantidadSalidasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='count'),
    MediaSalidasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='mean'),
    MedianaSalidasCortaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='median')
).reset_index()


resultados_salidas_media_ahorro = ahorro_salida_media.groupby('DocumentoCliente').agg(
    MontoSalidasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='sum'),
    CantidadSalidasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='count'),
    MediaSalidasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='mean'),
    MedianaSalidasMediaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='median')
).reset_index()

resultados_salidas_larga_ahorro = ahorro_salida_larga.groupby('DocumentoCliente').agg(
    MontoSalidasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='sum'),
    CantidadSalidasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='count'),
    MediaSalidasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='mean'),
    MedianaSalidasLargaAhorro=pd.NamedAgg(column='MontoTransaccion', aggfunc='median')
).reset_index()

Salidas_ahorro = pd.merge(pd.merge(resultados_salidas_larga_ahorro,resultados_salidas_media_ahorro, on = 'DocumentoCliente',how='left'),resultados_salidas_corta_ahorro,on = 'DocumentoCliente',how='left')



query_tx_activo_salida = '''select DocumentoCliente,FechaPublicacion,MontoCapital,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesActivo]
where CaracterTransaccion = 'Salida' and DescripcionTransaccion  in 
('Desembolso adicional','Desembolso inicial') '''
activo_salidas = cx.read_sql(conn=sql_connection, query=query_tx_activo_salida, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])



activo_salidas['FechaPublicacion'] = activo_salidas['FechaPublicacion'].astype(int).astype(str).str.strip()

activo_salidas['DocumentoCliente'] = activo_salidas['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

activo_salidas['FechaPublicacion'] = activo_salidas['FechaPublicacion'].apply(convertir_fecha_juliana)



activo_salida_corta = activo_salidas[(activo_salidas['FechaPublicacion'] >= fecha_un_mes_antes) & (activo_salidas['FechaPublicacion'] <= ultimo_dia_mes_anterior)]

activo_salida_media = activo_salidas[(activo_salidas['FechaPublicacion'] >= fecha_tres_meses_antes) & (activo_salidas['FechaPublicacion'] <= ultimo_dia_mes_anterior)]

activo_salida_larga = activo_salidas[(activo_salidas['FechaPublicacion'] >= fecha_seis_meses_antes) & (activo_salidas['FechaPublicacion'] <= ultimo_dia_mes_anterior)]




resultados_salidas_corta_activo = activo_salida_corta.groupby('DocumentoCliente').agg(
    MontoSalidasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='sum'),
    CantidadSalidasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='count'),
    MediaSalidasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='mean'),
    MedianaSalidasCortaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='median')
).reset_index()


resultados_salidas_media_activo = activo_salida_media.groupby('DocumentoCliente').agg(
    MontoSalidasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='sum'),
    CantidadSalidasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='count'),
    MediaSalidasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='mean'),
    MedianaSalidasMediaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='median')
).reset_index()

resultados_salidas_larga_activo = activo_salida_larga.groupby('DocumentoCliente').agg(
    MontoSalidasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='sum'),
    CantidadSalidasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='count'),
    MediaSalidasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='mean'),
    MedianaSalidasLargaActivo=pd.NamedAgg(column='MontoCapital', aggfunc='median')
).reset_index()

Salidas_activo = pd.merge(pd.merge(resultados_salidas_larga_activo,resultados_salidas_media_activo, on = 'DocumentoCliente',how='left'),resultados_salidas_corta_activo,on = 'DocumentoCliente',how='left')





query_tx_cdt_salida = '''select DocumentoCliente,FechaEfectiva,[MontoTransaccion1],ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesCDT]
where CaracterTransaccion = 'Salida' '''
cdt_salidas = cx.read_sql(conn=sql_connection, query=query_tx_cdt_salida, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])



cdt_salidas['FechaEfectiva'] = cdt_salidas['FechaEfectiva'].astype(int).astype(str).str.strip()

cdt_salidas['DocumentoCliente'] = cdt_salidas['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

cdt_salidas['FechaEfectiva'] = cdt_salidas['FechaEfectiva'].apply(convertir_fecha_juliana)



cdt_salida_corta = cdt_salidas[(cdt_salidas['FechaEfectiva'] >= fecha_un_mes_antes) & (cdt_salidas['FechaEfectiva'] <= ultimo_dia_mes_anterior)]

cdt_salida_media = cdt_salidas[(cdt_salidas['FechaEfectiva'] >= fecha_tres_meses_antes) & (cdt_salidas['FechaEfectiva'] <= ultimo_dia_mes_anterior)]

cdt_salida_larga = cdt_salidas[(cdt_salidas['FechaEfectiva'] >= fecha_seis_meses_antes) & (cdt_salidas['FechaEfectiva'] <= ultimo_dia_mes_anterior)]


resultados_salidas_corta_cdt = cdt_salida_corta.groupby('DocumentoCliente').agg(
    MontoSalidasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='sum'),
    CantidadSalidasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='count'),
    MediaSalidasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='mean'),
    MedianaSalidasCortacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='median')
).reset_index()


resultados_salidas_media_cdt = cdt_salida_media.groupby('DocumentoCliente').agg(
    MontoSalidasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='sum'),
    CantidadSalidasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='count'),
    MediaSalidasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='mean'),
    MedianaSalidasMediacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='median')
).reset_index()

resultados_salidas_larga_cdt = cdt_salida_larga.groupby('DocumentoCliente').agg(
    MontoSalidasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='sum'),
    CantidadSalidasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='count'),
    MediaSalidasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='mean'),
    MedianaSalidasLargacdt=pd.NamedAgg(column='MontoTransaccion1', aggfunc='median')
).reset_index()

Salidas_cdt = pd.merge(pd.merge(resultados_salidas_larga_cdt,resultados_salidas_media_cdt, on = 'DocumentoCliente',how='left'),resultados_salidas_corta_cdt,on = 'DocumentoCliente',how='left')





# SALIDAS TDC

query_tx_tdc_salida = f'''SELECT NUMDOC as DocumentoCliente,FECFACOC,IMPFACOC    FROM OPENQUERY(DB2400_182,'select C.NUMDOC,A.FECFACOC,A.IMPFACOC 
  FROM  INTTARCRE.SATOPECUO A 
  LEFT join INTTARCRE.SATBENEFI B
  on A.CUENTAOC = B.CUENTA
  LEFT JOIN  INTTARCRE.SATDACOPE C
  ON B.IDENTCLI = C.IDENTCLI
  where  A.FECFACOC >= ''{fecha_seis_meses_antes}'' ') '''


tdc_salidas =cargue_openquery(conn,query_tx_tdc_salida)



tdc_salidas['DocumentoCliente'] = tdc_salidas['DocumentoCliente'].astype(str).str.strip()


tdc_salidas['FECFACOC'] = (pd.to_datetime(tdc_salidas['FECFACOC'])).dt.strftime("%Y-%m-%d")


tdc_salida_corta = tdc_salidas[(tdc_salidas['FECFACOC'] >= fecha_un_mes_antes) & (tdc_salidas['FECFACOC'] <= ultimo_dia_mes_anterior)]

tdc_salida_media = tdc_salidas[(tdc_salidas['FECFACOC'] >= fecha_tres_meses_antes) & (tdc_salidas['FECFACOC'] <= ultimo_dia_mes_anterior)]

tdc_salida_larga = tdc_salidas[(tdc_salidas['FECFACOC'] >= fecha_seis_meses_antes) & (tdc_salidas['FECFACOC'] <= ultimo_dia_mes_anterior)]




resultados_salidas_corta_tdc = tdc_salida_corta.groupby('DocumentoCliente').agg(
    MontoSalidasCortatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='sum'),
    CantidadSalidasCortatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='count'),
    MediaSalidasCortatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='mean'),
    MedianaSalidasCortatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='median')
).reset_index()


resultados_salidas_media_tdc = tdc_salida_media.groupby('DocumentoCliente').agg(
    MontoSalidasMediatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='sum'),
    CantidadSalidasMediatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='count'),
    MediaSalidasMediatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='mean'),
    MedianaSalidasMediatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='median')
).reset_index()

resultados_salidas_larga_tdc = tdc_salida_larga.groupby('DocumentoCliente').agg(
    MontoSalidasLargatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='sum'),
    CantidadSalidasLargatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='count'),
    MediaSalidasLargatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='mean'),
    MedianaSalidasLargatdc=pd.NamedAgg(column='IMPFACOC', aggfunc='median')
).reset_index()

Salidas_tdc = pd.merge(pd.merge(resultados_salidas_larga_tdc,resultados_salidas_media_tdc, on = 'DocumentoCliente',how='left'),resultados_salidas_corta_tdc,on = 'DocumentoCliente',how='left')






## construcción consolidado unico de tx de entrada y salida 



# Supongamos que tienes 6 DataFrames llamados df1, df2, df3, df4, df5 y df6, todos con una columna 'Documento'.
# Puedes guardarlos en una lista para facilitar la manipulación.

dataframes = [Entradas_ahorro,Entradas_activo,Entradas_cdt, Entradas_tdc, Salidas_ahorro, Salidas_activo, Salidas_cdt, Salidas_tdc]

# Extraer los documentos únicos de cada DataFrame y almacenarlos en una lista
documentos_unicos_consolidado = [df['DocumentoCliente'].unique() for df in dataframes]

# Concatenar los documentos únicos en un solo DataFrame
documentos_df = pd.concat([pd.Series(doc) for doc in documentos_unicos_consolidado], ignore_index=True)


documentos_df = pd.DataFrame({'DocumentoCliente': documentos_df}).drop_duplicates().reset_index(drop=True)


consolidado_movimientos = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(documentos_df, Entradas_ahorro, on='DocumentoCliente', how='left'), Entradas_activo, on='DocumentoCliente', how='left'), Entradas_cdt, on='DocumentoCliente', how='left'), Entradas_tdc, on='DocumentoCliente', how='left'), Salidas_ahorro, on='DocumentoCliente', how='left'), Salidas_activo, on='DocumentoCliente', how='left'), Salidas_cdt, on='DocumentoCliente', how='left'), Salidas_tdc, on='DocumentoCliente', how='left')


## Entrada y Salida larga 


consolidado_movimientos['MontoEntradasLarga'] = consolidado_movimientos[['MontoEntradasLargaAhorro', 'MediaEntradasLargaActivo', 'MontoEntradasLargacdt', 'MontoEntradasLargatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['CantidadEntradasLarga'] = consolidado_movimientos[['CantidadEntradasLargaAhorro', 'CantidadEntradasLargaActivo', 'CantidadEntradasLargacdt', 'CantidadEntradasLargatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['MediaEntradasLarga'] = consolidado_movimientos[['MediaEntradasLargaAhorro', 'MediaEntradasLargaActivo', 'MediaEntradasLargacdt', 'MediaEntradasLargatdc']].mean(axis=1, skipna=True)

consolidado_movimientos['MedianaEntradasLarga'] = consolidado_movimientos[['MedianaEntradasLargaAhorro', 'MedianaEntradasLargaActivo', 'MedianaEntradasLargacdt', 'MediaEntradasLargatdc']].median(axis=1, skipna=True)


consolidado_movimientos['MontoSalidasLarga'] = consolidado_movimientos[['MontoSalidasLargaAhorro', 'MediaSalidasLargaActivo', 'MontoSalidasLargacdt', 'MontoSalidasLargatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['CantidadSalidasLarga'] = consolidado_movimientos[['CantidadSalidasLargaAhorro', 'CantidadSalidasLargaActivo', 'CantidadSalidasLargacdt', 'CantidadSalidasLargatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['MediaSalidasLarga'] = consolidado_movimientos[['MediaSalidasLargaAhorro', 'MediaSalidasLargaActivo', 'MediaSalidasLargacdt', 'MediaSalidasLargatdc']].mean(axis=1, skipna=True)

consolidado_movimientos['MedianaSalidasLarga'] = consolidado_movimientos[['MedianaSalidasLargaAhorro', 'MedianaSalidasLargaActivo', 'MedianaSalidasLargacdt', 'MediaSalidasLargatdc']].median(axis=1, skipna=True)


## Entrada y Salida media


consolidado_movimientos['MontoEntradasMedia'] = consolidado_movimientos[['MontoEntradasMediaAhorro', 'MediaEntradasMediaActivo', 'MontoEntradasMediacdt', 'MontoEntradasMediatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['CantidadEntradasMedia'] = consolidado_movimientos[['CantidadEntradasMediaAhorro', 'CantidadEntradasMediaActivo', 'CantidadEntradasMediacdt', 'CantidadEntradasMediatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['MediaEntradasMedia'] = consolidado_movimientos[['MediaEntradasMediaAhorro', 'MediaEntradasMediaActivo', 'MediaEntradasMediacdt', 'MediaEntradasMediatdc']].mean(axis=1, skipna=True)


consolidado_movimientos['MedianaEntradasMedia'] = consolidado_movimientos[['MedianaEntradasMediaAhorro', 'MedianaEntradasMediaActivo', 'MedianaEntradasMediacdt', 'MediaEntradasMediatdc']].median(axis=1, skipna=True)

consolidado_movimientos['MontoSalidasMedia'] = consolidado_movimientos[['MontoSalidasMediaAhorro', 'MediaSalidasMediaActivo', 'MontoSalidasMediacdt', 'MontoSalidasMediatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['CantidadSalidasMedia'] = consolidado_movimientos[['CantidadSalidasMediaAhorro', 'CantidadSalidasMediaActivo', 'CantidadSalidasMediacdt', 'CantidadSalidasMediatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['MediaSalidasMedia'] = consolidado_movimientos[['MediaSalidasMediaAhorro', 'MediaSalidasMediaActivo', 'MediaSalidasMediacdt', 'MediaSalidasMediatdc']].mean(axis=1, skipna=True)

consolidado_movimientos['MedianaSalidasMedia'] = consolidado_movimientos[['MedianaSalidasMediaAhorro', 'MedianaSalidasMediaActivo', 'MedianaSalidasMediacdt', 'MediaSalidasMediatdc']].median(axis=1, skipna=True)



## Entrada y Salida corta


consolidado_movimientos['MontoEntradasCorta'] = consolidado_movimientos[['MontoEntradasCortaAhorro', 'MediaEntradasCortaActivo', 'MontoEntradasCortacdt', 'MontoEntradasCortatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['CantidadEntradasCorta'] = consolidado_movimientos[['CantidadEntradasCortaAhorro', 'CantidadEntradasCortaActivo', 'CantidadEntradasCortacdt', 'CantidadEntradasCortatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['MediaEntradasCorta'] = consolidado_movimientos[['MediaEntradasCortaAhorro', 'MediaEntradasCortaActivo', 'MediaEntradasCortacdt', 'MediaEntradasCortatdc']].mean(axis=1, skipna=True)


consolidado_movimientos['MedianaEntradasCorta'] = consolidado_movimientos[['MedianaEntradasCortaAhorro', 'MedianaEntradasCortaActivo', 'MedianaEntradasCortacdt', 'MediaEntradasCortatdc']].median(axis=1, skipna=True)

consolidado_movimientos['MontoSalidasCorta'] = consolidado_movimientos[['MontoSalidasCortaAhorro', 'MediaSalidasCortaActivo', 'MontoSalidasCortacdt', 'MontoSalidasCortatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['CantidadSalidasCorta'] = consolidado_movimientos[['CantidadSalidasCortaAhorro', 'CantidadSalidasCortaActivo', 'CantidadSalidasCortacdt', 'CantidadSalidasCortatdc']].sum(axis=1, skipna=True)

consolidado_movimientos['MediaSalidasCorta'] = consolidado_movimientos[['MediaSalidasCortaAhorro', 'MediaSalidasCortaActivo', 'MediaSalidasCortacdt', 'MediaSalidasCortatdc']].mean(axis=1, skipna=True)

consolidado_movimientos['MedianaSalidasCorta'] = consolidado_movimientos[['MedianaSalidasCortaAhorro', 'MedianaSalidasCortaActivo', 'MedianaSalidasCortacdt', 'MediaSalidasCortatdc']].median(axis=1, skipna=True)



# Lista de columnas calculadas
columnas_calculadas = ['DocumentoCliente',
    'MontoEntradasLarga', 'CantidadEntradasLarga', 'MediaEntradasLarga', 'MedianaEntradasLarga',
    'MontoSalidasLarga', 'CantidadSalidasLarga', 'MediaSalidasLarga', 'MedianaSalidasLarga',
    'MontoEntradasMedia', 'CantidadEntradasMedia', 'MediaEntradasMedia', 'MedianaEntradasMedia',
    'MontoSalidasMedia', 'CantidadSalidasMedia', 'MediaSalidasMedia', 'MedianaSalidasMedia',
    'MontoEntradasCorta', 'CantidadEntradasCorta', 'MediaEntradasCorta', 'MedianaEntradasCorta',
    'MontoSalidasCorta', 'CantidadSalidasCorta', 'MediaSalidasCorta', 'MedianaSalidasCorta'
]

# Filtra el DataFrame para incluir solo las columnas calculadas
consolidado_movimientos = consolidado_movimientos[columnas_calculadas]



consolidado_pj = pd.merge(consolidado_pj, consolidado_movimientos,on='DocumentoCliente',how='left')


# %% [14] FACTOR DE RIESGO JURISDICCIÓN


'''# validación catálogo jurisdicción #'''

catalogo_jurisdiccion = '''select Municipio,[Vulnerabilidad lavado de activos] as VulnerabilidadLavadoActivos
,[Vulnerabilidad terrorismo] as VulnerabilidadTerrorismo 
,[Valor de riesgo jurisdicción] as RiesgoJurisdiccion
,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank FROM [Catalogos].[dbo].[CatalogoSegmentacionJurisdiccion]'''		 

catalogo_jurisdiccion = (cx.read_sql(conn = sql_connection, query = catalogo_jurisdiccion, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])


catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].replace('BOGOTA D.C.', 'BOGOTA')
catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].replace('FLORIDA BLANCA', 'FLORIDABLANCA')
catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].replace('SANTAFE DE BOGOTA DC', 'BOGOTA')
catalogo_jurisdiccion = catalogo_jurisdiccion.sort_values(by='RiesgoJurisdiccion', ascending=False)
catalogo_jurisdiccion = (catalogo_jurisdiccion.drop_duplicates(subset='Municipio', keep='first')).rename(columns={'Municipio':'CiudadActual'})


consolidado_pj['CiudadActual'] = consolidado_pj['CiudadActual'].str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
consolidado_pj['CiudadActual'] = consolidado_pj['CiudadActual'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()
consolidado_pj['CiudadActual'] = consolidado_pj['CiudadActual'].replace('BOGOTA D.C', 'BOGOTA')



consolidado_pj = pd.merge(consolidado_pj, catalogo_jurisdiccion, on=['CiudadActual'], how='left')


consolidado_pj['VulnerabilidadLavadoActivos'] = consolidado_pj['VulnerabilidadLavadoActivos'].replace('<NA>', np.nan, regex=True)
consolidado_pj['VulnerabilidadTerrorismo'] = consolidado_pj['VulnerabilidadTerrorismo'].replace('<NA>', np.nan, regex=True)
consolidado_pj['RiesgoJurisdiccion'] = consolidado_pj['RiesgoJurisdiccion'].replace('<NA>', np.nan, regex=True)


mapeo = {'Extremo': 4,'APP': 4, 'Internet': 4 , 'CVD': 3, 'Oficina': 1, 'Concesionario': 2,
         'Banca Virtual':4,'Vinculacion Digital':4, 'Banner Aliados':4
         ,'Prospecto Aliado': 3,'Concesionarios': 2, 'Oficinas': 1,'Especializado':3
         ,'Fuerza Movil':3,
         'Fuerza móvil corretaje':2
,'Centro de Contacto telefónico':3
,'Free Line':3
,'Banner Digital':3}

# Aplica el mapeo utilizando la función map() para crear la nueva columna 'Valoconsolidado'
consolidado_pj['RiesgoCanal'] = consolidado_pj['CanalEntrada'].map(mapeo)

consolidado_pj['RiesgoCanal'] = consolidado_pj['RiesgoCanal'].fillna(4)




# Aplica la función a la columna 'A' para asignar valoconsolidado
consolidado_pj['RiesgoProducto'] = consolidado_pj.apply(riesgo_producto, axis=1)

if segmentacion == "PJ":
    consolidado_pj['TipoPersona'] = 'Juridica'
    consolidado_pj['Tipo_identificacion'] = 'NIT'
else: consolidado_pj['TipoPersona'] = 'Natural'



consolidado_pj['FechaUltimaActualizacionCore'] = consolidado_pj.apply(lambda row: row['FechaApertura'] if pd.isnull(row['FechaUltimaActualizacionCore']) or row['FechaUltimaActualizacionCore'] == "0" else row['FechaUltimaActualizacionCore'], axis=1)


## generacion alertamientos 

# %% [15] ALERTAMIENTOS


perfil_tx_salidas_ahorro = '''select *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
  FROM [Productos y transaccionalidad].[dbo].[PerfilTransaccionalAhorro]'''
perfil_tx_salidas_ahorro = (cx.read_sql(conn = sql_connection, query = perfil_tx_salidas_ahorro, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])


perfil_tx_salidas_ahorro = perfil_tx_salidas_ahorro.groupby('DocumentoCliente')[['promedio tx al mes','sd tx mes', 'promedio  monto tx mes P85','sd monto tx mes P85']].quantile(0.9).reset_index()


perfil_tx_salidas_ahorro['limitecantidadmes'] = perfil_tx_salidas_ahorro['promedio tx al mes'] + 3 * perfil_tx_salidas_ahorro['sd tx mes']
perfil_tx_salidas_ahorro['limitemontomes'] = perfil_tx_salidas_ahorro['promedio  monto tx mes P85'] + 3 * perfil_tx_salidas_ahorro['sd monto tx mes P85']


perfil_tx_salidas_ahorro=pd.merge(perfil_tx_salidas_ahorro, resultados_salidas_corta_ahorro[['DocumentoCliente','MontoSalidasCortaAhorro','CantidadSalidasCortaAhorro']],on='DocumentoCliente',how='left')

perfil_tx_salidas_ahorro['AlertaPerfiltxahorrosalidas'] = np.where(
    (perfil_tx_salidas_ahorro['MontoSalidasCortaAhorro'] > perfil_tx_salidas_ahorro['limitemontomes']) | 
    (perfil_tx_salidas_ahorro['CantidadSalidasCortaAhorro'] > perfil_tx_salidas_ahorro['limitecantidadmes']), 1, 0
)


consolidado_pj = pd.merge(consolidado_pj, perfil_tx_salidas_ahorro[['DocumentoCliente','AlertaPerfiltxahorrosalidas']],on='DocumentoCliente',how='left')



'''# Alertamiento perfil TDC #'''


perfil_tx_tdc = '''select Documento as DocumentoCliente,[Cantidad de tx promedio al mes],[SD Cantidad de tx al mes],[Monto promedio tx al mes],[SD monto tx al mes],ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
  FROM [Productos y transaccionalidad].[dbo].[PerfilTransaccionalTDC_Facturacion]'''
perfil_tx_tdc = (cx.read_sql(conn = sql_connection, query = perfil_tx_tdc, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])



perfil_tx_tdc = perfil_tx_tdc.groupby('DocumentoCliente')[['Cantidad de tx promedio al mes','SD Cantidad de tx al mes', 'Monto promedio tx al mes','SD monto tx al mes']].quantile(0.9).reset_index()


perfil_tx_tdc['limitecantidadmes'] = perfil_tx_tdc['Cantidad de tx promedio al mes'] + 3 * perfil_tx_tdc['SD Cantidad de tx al mes']
perfil_tx_tdc['limitemontomes'] = perfil_tx_tdc['Monto promedio tx al mes'] + 3 * perfil_tx_tdc['SD monto tx al mes']



perfil_tx_tdc['DocumentoCliente'] = perfil_tx_tdc['DocumentoCliente'].astype(str).str.strip()

resultados_salidas_corta_tdc['DocumentoCliente'] = resultados_salidas_corta_tdc['DocumentoCliente'].str.strip()


perfil_tx_tdc=pd.merge(perfil_tx_tdc, resultados_salidas_corta_tdc[['DocumentoCliente','MontoSalidasCortatdc','CantidadSalidasCortatdc']],on='DocumentoCliente',how='left')

perfil_tx_tdc['AlertaPerfiltxtdc'] = np.where(
    (perfil_tx_tdc['MontoSalidasCortatdc'] > perfil_tx_tdc['limitemontomes']) | 
    (perfil_tx_tdc['CantidadSalidasCortatdc'] > perfil_tx_tdc['limitecantidadmes']), 1, 0
)

perfil_tx_tdc=perfil_tx_tdc.groupby('DocumentoCliente').first().reset_index()

consolidado_pj = pd.merge(consolidado_pj, perfil_tx_tdc[['DocumentoCliente','AlertaPerfiltxtdc']],on='DocumentoCliente',how='left')


'''# Alertamiento prepagos #'''

prepagos = '''SELECT 
      [OBLIGACION]
      ,[IDENTIFICACION] as DocumentoCliente
      ,[NOMBRE_CLIENTE]
      ,[FECHA_SIG_CUOTA]
	  ,[VLR_DESEMBOLSO]
	  ,[CUO_RESTANTES]
	  ,[SALDO_CAPITAL_ACTUAL]
	  ,[VLR_CUOTA]
      ,[RECAUDO MES]
	  ,CASE 
    WHEN ([RECAUDO MES] ) > (VLR_CUOTA * 1.3) THEN 1 
    ELSE 0 
END 
AS AlertaVCRec
,ceiling(SALDO_CAPITAL_ACTUAL / [VLR_DESEMBOLSO] *100) AS Amortizado
,CASE
WHEN (VLR_DESEMBOLSO - (SALDO_CAPITAL_ACTUAL + VLR_CUOTA * CUO_RESTANTES)) < 0 THEN 1 ELSE 0 end as AlertaDSrest,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
FROM [CarteraFinaInco].[dbo].[PLANO_CLIENTE_CIERRE]
where [VLR_CUOTA] != -1 and VLR_DESEMBOLSO != 0'''
prepagos = (cx.read_sql(conn = sql_connection, query = prepagos, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])

# prepagos['DocumentoCliente'] = prepagos['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()


prepagos['DocumentoCliente'] = prepagos['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(float(x))).astype(str).str.strip()


prepagos = prepagos.groupby('DocumentoCliente')[['AlertaVCRec', 'AlertaDSrest']].sum().reset_index()

consolidado_pj = pd.merge(consolidado_pj, prepagos,on='DocumentoCliente',how='left')



# %% [16] CLIENTES ACTIVOS


list(consolidado_pj.columns)



'''# Validación última tx y si el cliente es activo #'''

validacion_saldo = '''Select CUNA1 as Nombre, DMACCT as NumeroCuenta,DMDOPN as FechaApertura,DMSTAT as Estado,DMCBAL as saldoactual,DMYBAL as saldodiaanterior,CUSSNR as Documento,DMTYPE as TipoProducto, CUEMA1 as Correo, CUCLPH as Celular,CUNA2 as Direccion
                    from openquery (DB2400_182,'select
                    ProductoAhorro.DMDOPN,
                    ProductoAhorro.DMACCT,
                    ProductoAhorro.DMCBAL,
                    ProductoAhorro.DMYBAL,
                    ProductoAhorro.DMTYPE,
                    ProductoAhorro.DMSTAT,
                    Cliente.CUSSNR,
                    Cliente.CUEMA1,
                    Cliente.CUCLPH,
                    Cliente.CUNA1,
                    Cliente.CUNA2
                    from BNKPRD01.TAP002 as ProductoAhorro
                    left join (select case when left(ltrim(rtrim(CUX1AC)),1)<>9 then right(ltrim(rtrim(CUX1AC)),length(ltrim(rtrim(CUX1AC)))-1)
                    else ltrim(rtrim(CUX1AC))
                    end as CUX1AC
                    ,CUX1CS
                    ,CUX1AP
                    ,CUXREC
                    ,CUXREL
                    ,CUX1TY
                    from BNKPRD01.CUP009
                    where CUXREL in(''SOW'',''JOF'')) as Enlace
                    on cast(ProductoAhorro.DMACCT as char(30))=cast(Enlace.CUX1AC as char(30))
                    left join BNKPRD01.CUP003 as Cliente
                    on Enlace.CUX1CS=Cliente.CUNBR ') '''


validacion_saldo = cargue_openquery(conn,validacion_saldo)

validacion_saldo = validacion_saldo.groupby('Documento').agg({'saldoactual': 'sum', 'saldodiaanterior': 'sum'}).reset_index()
validacion_saldo['Documento'] = validacion_saldo['Documento'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()
validacion_saldo['validacion_saldo'] = np.where((validacion_saldo['saldoactual'] > 0) | (validacion_saldo['saldodiaanterior'] > 0), 1, 0)

validacion_saldo = validacion_saldo[validacion_saldo['validacion_saldo']==1]


# acá



consolidado_tx = '''select DocumentoCliente, FechaTransaccion, CaracterTransaccion, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank  
FROM [Productos y transaccionalidad].[dbo].[ConsolidadoTransacciones]
where FechaTransaccion >= GETDATE()-365 '''

consolidado_tx = (cx.read_sql(conn = sql_connection, query = consolidado_tx, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])

consolidado_tx['FechaTransaccion'] = pd.to_datetime(consolidado_tx['FechaTransaccion']).dt.strftime("%Y-%m-%d")


consolidado_tx = consolidado_tx.groupby(['DocumentoCliente', 'CaracterTransaccion'])['FechaTransaccion'].max().reset_index()


consolidado_tx = (consolidado_tx.pivot(index='DocumentoCliente', columns='CaracterTransaccion', values='FechaTransaccion')).reset_index()


consolidado_tx['Entrada'] = pd.to_datetime(consolidado_tx['Entrada'])
consolidado_tx['Otros'] = pd.to_datetime(consolidado_tx['Otros'])
consolidado_tx['Salida'] = pd.to_datetime(consolidado_tx['Salida'])

# Calcula la fecha actual y la fecha hace un año
fecha_hace_un_anio = pd.to_datetime('today') - pd.DateOffset(years=1)

# Aplica una función lambda para verificar si alguna de las columnas cumple la condición
consolidado_tx['Activo'] = consolidado_tx.apply(lambda row: any((row['Entrada'] > fecha_hace_un_anio, row['Otros'] > fecha_hace_un_anio, row['Salida'] > fecha_hace_un_anio)), axis=1)


# Calcula la diferencia en días entre la fecha actual y la columna "FechaColumna"
consolidado_tx['dias_de_ultima_tx_entrada'] = (pd.to_datetime('today') - consolidado_tx['Entrada']).dt.days
consolidado_tx['dias_de_ultima_tx_salida'] = (pd.to_datetime('today') - consolidado_tx['Salida']).dt.days
consolidado_tx['dias_de_ultima_tx_otros'] = (pd.to_datetime('today') - consolidado_tx['Otros']).dt.days

consolidado_tx['Entrada'] = pd.to_datetime(consolidado_tx['Entrada']).dt.strftime("%Y-%m-%d")
consolidado_tx['Otros'] = pd.to_datetime(consolidado_tx['Otros']).dt.strftime("%Y-%m-%d")
consolidado_tx['Salida'] = pd.to_datetime(consolidado_tx['Salida']).dt.strftime("%Y-%m-%d")

consolidado_tx['tiene_saldo'] = consolidado_tx['DocumentoCliente'].isin(validacion_saldo['Documento']).astype(int)
consolidado_tx['Activo'] = np.where(consolidado_tx['tiene_saldo']==1,True,consolidado_tx['Activo'])
consolidado_tx=consolidado_tx.drop(columns={'tiene_saldo'})

consolidado_pj = pd.merge(consolidado_pj, consolidado_tx,on='DocumentoCliente',how='left')

# Aplica la función a cada fila y crea la columna 'ahorro' en el DataFrame
consolidado_pj['Ahorro'] = consolidado_pj.apply(calcular_ahorro, axis=1)


# %% [17] FACTOR DE RIESGO PRODUCTO


atributos_ahorro = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[AtributosAhorro]'''
atributos_ahorro = cx.read_sql(conn=sql_connection, query=atributos_ahorro,
                               partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

atributos_ahorro['Ahorro'] = atributos_ahorro['Ahorro'].astype(int)
consolidado_pj = pd.merge(consolidado_pj, atributos_ahorro, on='Ahorro', how='left')


consolidado_pj['MaximoAhorro'] = consolidado_pj[['producto Atributo A', 'producto Atributo B', 'producto Atributo C',
                                                 'producto Atributo D', 'producto Atributo E', 'producto Atributo F',
                                                 'producto Atributo G']].max(axis=1)


consolidado_pj['Credito'] = consolidado_pj.apply(calcular_credito, axis=1)


atributos_credito = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[AtributosCredito]'''
atributos_credito = cx.read_sql(conn=sql_connection, query=atributos_credito,
                                partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

atributos_credito['Credito'] = atributos_credito['Credito'].astype(int)
consolidado_pj = pd.merge(consolidado_pj, atributos_credito, on='Credito', how='left')


consolidado_pj['MaximoCredito'] = consolidado_pj[['Atributo credito A', 'Atributo credito B',
                                                  'Atributo credito C', 'Atributo credito D', 'Atributo credito E', 'Atributo credito F']].max(axis=1)

consolidado_pj['RIESGO PRODUCTO'] = consolidado_pj[['MaximoAhorro', 'MaximoCredito']].max(axis=1)


# ruta_completa = r'C:\Users\josgom\Desktop\BORRAR\datosPJsarlaft.xlsx'  # Usar 'r' para interpretar la cadena como una ruta cruda

# # Exportar el DataFrame a la ubicación deseada
# consolidado_pj.to_excel(ruta_completa, index=False)

# # Remover caracteres no válidos de todas las columnas
# consolidado_pjsubir = consolidado_pj.applymap(lambda x: ''.join(filter(str.isprintable, str(x))))

# # Guardar en Excel
# ruta_completa = r'C:\Users\josgom\Desktop\BORRAR\datosPJsarlaft.xlsx'
# consolidado_pjsubir.to_excel(ruta_completa, index=False)

##########################################3333


atributos_jurisdiccion = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[BaseJurisdiccion]'''
atributos_jurisdiccion = cx.read_sql(conn=sql_connection, query=atributos_jurisdiccion,
                                     partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


consolidado_pj = pd.merge(consolidado_pj, atributos_jurisdiccion, on='Departamento', how='left')


## medias informalidad


cuantiles_informalidad = consolidado_pj['Informalidad'].quantile([0.25, 0.5, 0.75])

rq_informalidad = cuantiles_informalidad.loc[0.75] - cuantiles_informalidad.loc[0.25]

MEDIAQ1jur = cuantiles_informalidad.loc[0.25] / 2


# Supongamos que tienes los valoconsolidado_pj DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_informalidad = [
    (consolidado_pj['Informalidad'] <= MEDIAQ1jur),
    (consolidado_pj['Informalidad'] > MEDIAQ1jur) & (
        consolidado_pj['Informalidad'] <= cuantiles_informalidad.loc[0.25]),
    (consolidado_pj['Informalidad'] > cuantiles_informalidad.loc[0.25]) & (
        consolidado_pj['Informalidad'] <= cuantiles_informalidad.loc[0.5]),
    (consolidado_pj['Informalidad'] > cuantiles_informalidad.loc[0.5]) & (
        consolidado_pj['Informalidad'] <= cuantiles_informalidad.loc[0.75]),
    (consolidado_pj['Informalidad'] > cuantiles_informalidad.loc[0.75])
]

# Crea una lista de valoconsolidado_pj para asignar a cada condición
valores_informalidad = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado_pj corconsolidado_pjpondientes
consolidado_pj['Nivel Riesgo informalidad Jurisdiccion'] = np.select(
    condiciones_informalidad, valores_informalidad, default=0)


## medidas desempleo


cuantiles_desempleo = consolidado_pj['Desempleo'].quantile([0.25, 0.5, 0.75])

rq_desempleo = cuantiles_desempleo.loc[0.75] - cuantiles_desempleo.loc[0.25]

MEDIAQ1desempleo = cuantiles_desempleo.loc[0.25] / 2


# Supongamos que tienes los valoconsolidado_pj DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_desempleo = [
    (consolidado_pj['Desempleo'] <= MEDIAQ1desempleo),
    (consolidado_pj['Desempleo'] > MEDIAQ1desempleo) & (
        consolidado_pj['Desempleo'] <= cuantiles_desempleo.loc[0.25]),
    (consolidado_pj['Desempleo'] > cuantiles_desempleo.loc[0.25]) & (
        consolidado_pj['Desempleo'] <= cuantiles_desempleo.loc[0.5]),
    (consolidado_pj['Desempleo'] > cuantiles_desempleo.loc[0.5]) & (
        consolidado_pj['Desempleo'] <= cuantiles_desempleo.loc[0.75]),
    (consolidado_pj['Desempleo'] > cuantiles_desempleo.loc[0.75])
]

# Crea una lista de valoconsolidado_pj para asignar a cada condición
valores_desempleo = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado_pj corconsolidado_pjpondientes
consolidado_pj['Nivel Riesgo desempleo Jurisdiccion'] = np.select(
    condiciones_desempleo, valores_desempleo, default=0)


## medidas ITD


cuantiles_itd = consolidado_pj['ITD'].quantile([0.25, 0.5, 0.75])

rq_itd = cuantiles_itd.loc[0.75] - cuantiles_itd.loc[0.25]

MEDIAQ1itd = cuantiles_itd.loc[0.25] / 2


# Supongamos que tienes los valoconsolidado_pj DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_itd = [
    (consolidado_pj['ITD'] <= MEDIAQ1itd),
    (consolidado_pj['ITD'] > MEDIAQ1itd) & (
        consolidado_pj['ITD'] <= cuantiles_itd.loc[0.25]),
    (consolidado_pj['ITD'] > cuantiles_itd.loc[0.25]) & (
        consolidado_pj['ITD'] <= cuantiles_itd.loc[0.5]),
    (consolidado_pj['ITD'] > cuantiles_itd.loc[0.5]) & (
        consolidado_pj['ITD'] <= cuantiles_itd.loc[0.75]),
    (consolidado_pj['ITD'] > cuantiles_itd.loc[0.75])
]

# Crea una lista de valoconsolidado_pj para asignar a cada condición
valores_idt = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado_pj corconsolidado_pjpondientes
consolidado_pj['Nivel Riesgo ITD Jurisdiccion'] = np.select(
    condiciones_itd, valores_idt, default=0)


## medidas PIB


cuantiles_pib = consolidado_pj['PIB departamental2'].quantile([
                                                              0.25, 0.5, 0.75])

rq_pib = cuantiles_pib.loc[0.75] - cuantiles_pib.loc[0.25]

MEDIAQ1pib = cuantiles_pib.loc[0.25] / 2


# Supongamos que tienes los valoconsolidado_pj DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_pib = [
    (consolidado_pj['PIB departamental2'] <= MEDIAQ1pib),
    (consolidado_pj['PIB departamental2'] > MEDIAQ1pib) & (
        consolidado_pj['PIB departamental2'] <= cuantiles_pib.loc[0.25]),
    (consolidado_pj['PIB departamental2'] > cuantiles_pib.loc[0.25]) & (
        consolidado_pj['PIB departamental2'] <= cuantiles_pib.loc[0.5]),
    (consolidado_pj['PIB departamental2'] > cuantiles_pib.loc[0.5]) & (
        consolidado_pj['PIB departamental2'] <= cuantiles_pib.loc[0.75]),
    (consolidado_pj['PIB departamental2'] > cuantiles_pib.loc[0.75])
]

# Crea una lista de valoconsolidado_pj para asignar a cada condición
valores_pib = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado_pj corconsolidado_pjpondientes
consolidado_pj['Nivel Riesgo PIB departamental Jurisdiccion'] = np.select(
    condiciones_pib, valores_pib, default=0)


consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] = consolidado_pj['Nivel Riesgo informalidad Jurisdiccion'] * 0.35 + consolidado_pj['Nivel Riesgo desempleo Jurisdiccion'] * \
    0.2 + consolidado_pj['Nivel Riesgo ITD Jurisdiccion'] * 0.1 + \
    consolidado_pj['Nivel Riesgo PIB departamental Jurisdiccion'] * 0.35



## segmentacion anterior

# segmentacion_anterior = f'''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
# 	    FROM [ModelosSARLAFT].[dbo].[ClienteProductoJuridico_202310_transformado]'''
# segmentacion_anterior = cx.read_sql(conn = sql_connection, query = segmentacion_anterior, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


# segmentacion_anterior = f'''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
#  	    FROM [ModelosSARLAFT].[dbo].[BasePersonanatural_SarlaftRI]'''
# consolidado_pj = cx.read_sql(conn = sql_connection, query = segmentacion_anterior, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])






maximo_jurisdiccion = consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'].max()

limite_3_jurisdiccion = maximo_jurisdiccion * 3 / 4
limite_2_jurisdiccion = maximo_jurisdiccion * 2 / 4
limite_1_jurisdiccion = maximo_jurisdiccion * 1 / 4


def asignar_valor_jurisdiccion(inicial):
    if inicial <= limite_1_jurisdiccion :
        return 1
    elif inicial <= limite_2_jurisdiccion:
        return 2
    elif inicial <= limite_3_jurisdiccion:
        return 3
    else:
        return 4

# Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] = consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'].apply(lambda x: asignar_valor_jurisdiccion(x))


# base=consolidado_pj


## prueba 

extranjeros = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
FROM [ModelosSARLAFT].[dbo].[ConsolidadoExtranjeros]'''
extranjeros = cx.read_sql(conn = sql_connection, query = extranjeros, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


consolidado_pj['extranjero'] = consolidado_pj['DocumentoCliente'].isin(extranjeros['DocumentoCliente']).astype(int)
consolidado_pj = pd.merge(consolidado_pj,extranjeros,on='DocumentoCliente',how='left')
consolidado_pj['Origen Nacional'] = np.where(consolidado_pj['Origen Nacional'].isnull(),"COLOMBIA",consolidado_pj['Origen Nacional'])



# estimacion pais 
consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] = consolidado_pj.apply(lambda row: row['Nivel de riesgo JURISDICCION NACIONAL'] if pd.isnull(row['Clasificacion_pais']) or row['Nivel de riesgo JURISDICCION NACIONAL'] > row['Clasificacion_pais'] else row['Clasificacion_pais'], axis=1)


# estimacion departamento - municipio 

consolidado_pj['depto_muni']= consolidado_pj['Departamento'] + '-' + consolidado_pj['Municipio']


depto_muni = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
FROM [ModelosSARLAFT].[dbo].[RiesgoDeptoMuni]'''
depto_muni = cx.read_sql(conn = sql_connection, query = depto_muni, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

consolidado_pj = pd.merge(consolidado_pj,depto_muni,on='depto_muni',how='left')

consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] = consolidado_pj.apply(lambda row: row['Nivel de riesgo JURISDICCION NACIONAL'] if pd.isnull(row['Clasificacion_depto_mun']) or row['Nivel de riesgo JURISDICCION NACIONAL'] > row['Clasificacion_depto_mun'] else row['Clasificacion_depto_mun'], axis=1)








# %% [18] INFORMACIÓN GLPI


###### informacion GLPI


informacion_GLPI = '''select A.NumId as DocumentoCliente,B.Ingresos,C.WECodCIIU,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
from [fabogsqlclu].[LineaProduccion].[dbo].[DB_PERSONA] A
left join [fabogsqlclu].[LineaProduccion].[dbo].[DB_ESTADO_FINANCIERO] B
on A.CodPersona = B.CodPersona
left join  [fabogsqlclu].[LineaProduccion].[dbo].[TMP_PERSONA_ICBS] C
on A.NumId = C.WENroIde'''
informacion_GLPI = cx.read_sql(conn=sql_connection, query=informacion_GLPI,
                               partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])



informacion_GLPI['DocumentoCliente'] = informacion_GLPI['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()

informacion_GLPI['Ingresos'] = informacion_GLPI['Ingresos'].fillna(0)


# Supongamos que tienes un DataFrame llamado 'informacion_GLPI' con columnas 'DocumentoCliente' y 'Ingresos'

# Elimina filas con valores nulos en 'DocumentoCliente'
informacion_GLPI = informacion_GLPI.dropna(subset=['DocumentoCliente'])


# Encuentra los índices de las filas con el valor máximo de 'Ingresos' para cada documento
indices_maximos = informacion_GLPI.groupby('DocumentoCliente')['Ingresos'].idxmax()

# Filtra el DataFrame 'informacion_GLPI' utilizando los índices de las filas máximas
informacion_GLPI = informacion_GLPI.loc[indices_maximos]


consolidado_pj = pd.merge(consolidado_pj, informacion_GLPI,on='DocumentoCliente', how='left')

consolidado_pj['MontoIngresos'] = np.where(consolidado_pj['MontoIngresos'].isnull(), consolidado_pj['Ingresos'], consolidado_pj['MontoIngresos'])

consolidado_pj['CodigoCIIU'] = np.where(consolidado_pj['CodigoCIIU'].isnull(), consolidado_pj['WECodCIIU'], consolidado_pj['CodigoCIIU'])



# %% [19] FACTOR DE RIESGO ACTIVIDAD ECONÓMICA


atributos_actividad_economica = '''SELECT [DOS DIGITOS] as CodigoCIIU,*,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[BaseActividadEconomica]'''
atributos_actividad_economica = cx.read_sql(conn = sql_connection, query = atributos_actividad_economica, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

atributos_actividad_economica['Informalidad2'] = atributos_actividad_economica['Informalidad'].str.replace('%', '')

# Reemplaza "NA" por una cadena vacía en la columna 'Informalidad (right)'
atributos_actividad_economica['Informalidad2'] = atributos_actividad_economica['Informalidad2'].str.replace('NA', '')

atributos_actividad_economica['Informalidad2'] = atributos_actividad_economica['Informalidad2'].replace('', np.nan)

# Convierte la columna 'Informalidad' a valores numéricos
atributos_actividad_economica['Informalidad2'] = atributos_actividad_economica['Informalidad2'].str.replace(',', '.', regex=True).astype(float)

atributos_actividad_economica = atributos_actividad_economica.drop(columns=['Informalidad', 'DOS DIGITOS'])



consolidado_pj['CodigoCIIU'] = consolidado_pj['CodigoCIIU'].astype(str).str[:2]


consolidado_pj = pd.merge(consolidado_pj, atributos_actividad_economica,on='CodigoCIIU', how='left')

## validacion Nivel Riesgo Efectivo
cuantiles_efectivo = consolidado_pj['Efectivo'].quantile([0.25, 0.5, 0.75])

rq_efectivo =  cuantiles_efectivo.loc[0.75] - cuantiles_efectivo.loc[0.25]

MEDIAQ1efectivo = cuantiles_efectivo.loc[0.25] / 2



# Crea una serie de condiciones
condiciones_efectivo = [
    (consolidado_pj['Efectivo'] <= MEDIAQ1efectivo),
    (consolidado_pj['Efectivo'] > MEDIAQ1efectivo) & (consolidado_pj['Efectivo'] <= cuantiles_efectivo.loc[0.25]),
    (consolidado_pj['Efectivo'] > cuantiles_efectivo.loc[0.25]) & (consolidado_pj['Efectivo'] <= cuantiles_efectivo.loc[0.5]),
    (consolidado_pj['Efectivo'] > cuantiles_efectivo.loc[0.5]) & (consolidado_pj['Efectivo'] <= cuantiles_efectivo.loc[0.75]),
    (consolidado_pj['Efectivo'] > cuantiles_efectivo.loc[0.75])
]

# Crea una lista de valoconsolidado_pj para asignar a cada condición
valores_efectivo = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado_pj corconsolidado_pjpondientes
consolidado_pj['Nivel Riesgo Efectivo'] = np.select(condiciones_efectivo, valores_efectivo, default=0)


## validacion exportaciones   Nivel Riesgo CE



cuantiles_CE = consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'].quantile([0.25, 0.5, 0.75])

rq_CE =  cuantiles_CE.loc[0.75] - cuantiles_CE.loc[0.25]

MEDIAQ1CE = cuantiles_CE.loc[0.25] / 2



# Crea una serie de condiciones
condiciones_CE = [
    (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] <= MEDIAQ1CE),
    (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] > MEDIAQ1CE) & (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] <= cuantiles_CE.loc[0.25]),
    (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] > cuantiles_CE.loc[0.25]) & (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] <= cuantiles_CE.loc[0.5]),
    (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] > cuantiles_CE.loc[0.5]) & (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] <= cuantiles_CE.loc[0.75]),
    (consolidado_pj['Comercio Exterior (Exportaciones + Importaciones)'] > cuantiles_CE.loc[0.75])
]

# Crea una lista de valoconsolidado_pj para asignar a cada condición
valores_CE = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado_pj corconsolidado_pjpondientes
consolidado_pj['Nivel Riesgo CE'] = np.select(condiciones_CE, valores_CE, default=0)





## medias informalidad AE


cuantiles_informalidad2 = consolidado_pj['Informalidad2'].quantile([0.25, 0.5, 0.75])

rq_informalidad2 =  cuantiles_informalidad2.loc[0.75] - cuantiles_informalidad2.loc[0.25]

MEDIAQ1inf2 = cuantiles_informalidad2.loc[0.25] / 2


# Crea una serie de condiciones
condiciones_informalidad2 = [
    (consolidado_pj['Informalidad2'] <= MEDIAQ1inf2),
    (consolidado_pj['Informalidad2'] > MEDIAQ1inf2) & (consolidado_pj['Informalidad2'] <= cuantiles_informalidad2.loc[0.25]),
    (consolidado_pj['Informalidad2'] > cuantiles_informalidad2.loc[0.25]) & (consolidado_pj['Informalidad2'] <= cuantiles_informalidad2.loc[0.5]),
    (consolidado_pj['Informalidad2'] > cuantiles_informalidad2.loc[0.5]) & (consolidado_pj['Informalidad2'] <= cuantiles_informalidad2.loc[0.75]),
    (consolidado_pj['Informalidad2'] > cuantiles_informalidad2.loc[0.75])
]

# Crea una lista de valoconsolidado_pj para asignar a cada condición
valores_informalidad2 = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado_pj corconsolidado_pjpondientes
consolidado_pj['Nivel Riesgo informalidad2'] = np.select(condiciones_informalidad2, valores_informalidad2, default=0)

consolidado_pj['Nivel de Riesgo Actividades Alto Riesgo'] = np.where(consolidado_pj['Actividades de alto riesgo'] =="SI",5,1)
consolidado_pj['Nivel de Riesgo Sarlaft'] = np.where(consolidado_pj['Sarlaft'] =="SI",5,1)

consolidado_pj['PonderacionAtributosAE'] = (
    consolidado_pj['Nivel Riesgo CE'] * 0.2 +
    consolidado_pj['Nivel Riesgo informalidad2'] * 0.2 +
    consolidado_pj['Nivel de Riesgo Actividades Alto Riesgo'] * 0.15 +
    consolidado_pj['Nivel de Riesgo Sarlaft'] * 0.1 +
    consolidado_pj['Nivel Riesgo Efectivo'] * 0.35
)





maximo_AE = consolidado_pj['PonderacionAtributosAE'].max()

limite_3_AE = maximo_AE * 3 / 4
limite_2_AE = maximo_AE * 2 / 4
limite_1_AE = maximo_AE * 1 / 4


def asignar_valor_AE(inicial):
    if inicial <= limite_1_AE :
        return 1
    elif inicial <= limite_2_AE:
        return 2
    elif inicial <= limite_3_AE:
        return 3
    else:
        return 4

# Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
consolidado_pj['PonderacionAtributosAE'] = consolidado_pj['PonderacionAtributosAE'].apply(lambda x: asignar_valor_AE(x))
consolidado_pj['Calificación Segmentación']=consolidado_pj['Calificación Segmentación'].fillna(4)
consolidado_pj['PonderacionAtributosAE2'] = (consolidado_pj['PonderacionAtributosAE'] + consolidado_pj['Calificación Segmentación']) / 2
consolidado_pj['PonderacionAtributosAE2'] = consolidado_pj['PonderacionAtributosAE2'].apply(lambda x: math.ceil(x))



# %% [20] ESTIMACIÓN RIESGO INHERENTE DEL CLIENTE

# estimación riesgo inherente 

consolidado_pj['Riesgo inherente del cliente']  = (
    consolidado_pj['RIESGO PRODUCTO'] * 0.2 +
    consolidado_pj['PonderacionAtributosAE'] * 0.35 +
    consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] * 0.2 +
    consolidado_pj['RiesgoCanal'] * 0.25
)


maximo_RI = consolidado_pj['Riesgo inherente del cliente'].max()

limite_3_RI = maximo_RI * 3 / 4
limite_2_RI = maximo_RI * 2 / 4
limite_1_RI = maximo_RI * 1 / 4


def asignar_valor_RI(inicial):
    if inicial <= limite_1_RI :
        return 1
    elif inicial <= limite_2_RI:
        return 2
    elif inicial <= limite_3_RI:
        return 3
    else:
        return 4

# Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
consolidado_pj['Riesgo inherente del cliente_categoria'] = consolidado_pj['Riesgo inherente del cliente'].apply(lambda x: asignar_valor_RI(x))



##################### 

columnas_a_contar = [
    'CDT', 'Ahorro de la red', 'Cuenta corriente', 'FlexiDigital',
    'Nomina Finandina', 'Otros ahorros', 'TDC Digital', 'TDC Física',
    'Crédito hipotecario', 'Crédito libre inversión', 'Libranza',
    'Crédito vehículo', 'Leasing vehículo', 'Maquina agrícola',
    'Plan mayor', 'Redescuentos', 'Cartera vendida', 'Castigados',
    'Otros activos','Riesgo inherente del cliente'
]



conteos = (pd.DataFrame({'Columna': consolidado_pj[columnas_a_contar].columns, 'Conteo': consolidado_pj[columnas_a_contar].apply(lambda col: (col > 0).sum())}).reset_index(drop=True)).T
conteos.columns = conteos.iloc[0]
conteos = conteos.iloc[1:]
conteos.columns = ['np' + col for col in conteos.columns]



# Supongamos que 'consolidado_pj' es tu DataFrame y 'columnas_a_contar' es la lista de columnas que deseas contar y sumar 'Riesgo inherente del cliente' si cumplen con la condición

# Crear un diccionario para almacenar los resultados
resultados = {}

# Iterar sobre las columnas y calcular la suma de 'Riesgo inherente del cliente' para cada una
for columna in columnas_a_contar:
    suma_de_riesgo = consolidado_pj[consolidado_pj[columna] > 0]['Riesgo inherente del cliente'].sum()
    resultados[f'i_p_{columna}'] = suma_de_riesgo

# Crear un DataFrame a partir del diccionario de resultados
sumas_riesgo = pd.DataFrame(resultados, index=[0])


consolidado_pj['RipCDT'] = np.where(consolidado_pj['CDT']>0, (sumas_riesgo['i_p_CDT'].iloc[0]  / conteos['npCDT'].iloc[0]),0)
consolidado_pj['RipAhorro de la red'] = np.where(consolidado_pj['Ahorro de la red']>0, (sumas_riesgo['i_p_Ahorro de la red'].iloc[0]  / conteos['npAhorro de la red'].iloc[0]),0)
consolidado_pj['RipCuenta corriente'] = np.where(consolidado_pj['Cuenta corriente']>0, (sumas_riesgo['i_p_Cuenta corriente'].iloc[0]  / conteos['npCuenta corriente'].iloc[0]),0)
consolidado_pj['RipFlexiDigital'] = np.where(consolidado_pj['FlexiDigital']>0, (sumas_riesgo['i_p_FlexiDigital'].iloc[0]  / conteos['npFlexiDigital'].iloc[0]),0)
consolidado_pj['RipNomina Finandina'] = np.where(consolidado_pj['Nomina Finandina']>0, (sumas_riesgo['i_p_Nomina Finandina'].iloc[0]  / conteos['npNomina Finandina'].iloc[0]),0)
consolidado_pj['RipOtros ahorros'] = np.where(consolidado_pj['Otros ahorros']>0, (sumas_riesgo['i_p_Otros ahorros'].iloc[0]  / conteos['npOtros ahorros'].iloc[0]),0)
consolidado_pj['RipTDC Digital'] = np.where(consolidado_pj['TDC Digital']>0, (sumas_riesgo['i_p_TDC Digital'].iloc[0]  / conteos['npTDC Digital'].iloc[0]),0)
consolidado_pj['RipTDC Física'] = np.where(consolidado_pj['TDC Física']>0, (sumas_riesgo['i_p_TDC Física'].iloc[0]  / conteos['npTDC Física'].iloc[0]),0)
consolidado_pj['RipCrédito hipotecario'] = np.where(consolidado_pj['Crédito hipotecario']>0, (sumas_riesgo['i_p_Crédito hipotecario'].iloc[0]  / conteos['npCrédito hipotecario'].iloc[0]),0)
consolidado_pj['RipCrédito libre inversión'] = np.where(consolidado_pj['Crédito libre inversión']>0, (sumas_riesgo['i_p_Crédito libre inversión'].iloc[0]  / conteos['npCrédito libre inversión'].iloc[0]),0)
consolidado_pj['RipLibranza'] = np.where(consolidado_pj['Libranza']>0, (sumas_riesgo['i_p_Libranza'].iloc[0]  / conteos['npLibranza'].iloc[0]),0)
consolidado_pj['RipCrédito vehículo'] = np.where(consolidado_pj['Crédito vehículo']>0, (sumas_riesgo['i_p_Crédito vehículo'].iloc[0]  / conteos['npCrédito vehículo'].iloc[0]),0)
consolidado_pj['RipLeasing vehículo'] = np.where(consolidado_pj['Leasing vehículo']>0, (sumas_riesgo['i_p_Leasing vehículo'].iloc[0]  / conteos['npLeasing vehículo'].iloc[0]),0)
consolidado_pj['RipMaquina agrícola'] = np.where(consolidado_pj['Maquina agrícola']>0, (sumas_riesgo['i_p_Maquina agrícola'].iloc[0]  / conteos['npMaquina agrícola'].iloc[0]),0)
consolidado_pj['RipPlan mayor'] = np.where(consolidado_pj['Plan mayor']>0, (sumas_riesgo['i_p_Plan mayor'].iloc[0]  / conteos['npPlan mayor'].iloc[0]),0)
consolidado_pj['RipRedescuentos'] = np.where(consolidado_pj['Redescuentos']>0, (sumas_riesgo['i_p_Redescuentos'].iloc[0]  / conteos['npRedescuentos'].iloc[0]),0)
consolidado_pj['RipCartera vendida'] = np.where(consolidado_pj['Cartera vendida']>0, (sumas_riesgo['i_p_Cartera vendida'].iloc[0]  / conteos['npCartera vendida'].iloc[0]),0)
consolidado_pj['RipCastigados'] = np.where(consolidado_pj['Castigados']>0, (sumas_riesgo['i_p_Castigados'].iloc[0]  / conteos['npCastigados'].iloc[0]),0)
consolidado_pj['RipOtros activos'] = np.where(consolidado_pj['Otros activos']>0, (sumas_riesgo['i_p_Otros activos'].iloc[0]  / conteos['npOtros activos'].iloc[0]),0)


# %% [21] ESTIMACIÓN RIESGO INHERENTE COMPUESTO POR PRODUCTO


# Calcular la suma de las columnas 'Rip' para cada registro y crear una nueva columna 'Suma_Rip'
consolidado_pj['Riesgo inherente compuesto por producto'] = consolidado_pj[[col for col in consolidado_pj.columns if col.startswith('Rip')]].sum(axis=1)



maximo_producto = consolidado_pj['Riesgo inherente compuesto por producto'].max()

limite_3_producto = maximo_producto * 3 / 4
limite_2_producto = maximo_producto * 2 / 4
limite_1_producto = maximo_producto * 1 / 4


def asignar_valor_producto(inicial):
    if inicial <= limite_1_producto :
        return 1
    elif inicial <= limite_2_producto:
        return 2
    elif inicial <= limite_3_producto:
        return 3
    else:
        return 4

# Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
consolidado_pj['Riesgo inherente compuesto por producto'] = consolidado_pj['Riesgo inherente compuesto por producto'].apply(lambda x: asignar_valor_producto(x))



consolidado_pj['Riesgo inherente del cliente']  = (
    consolidado_pj['Riesgo inherente compuesto por producto'] * 0.2 +
    consolidado_pj['PonderacionAtributosAE'] * 0.35 +
    consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] * 0.2 +
    consolidado_pj['RiesgoCanal'] * 0.25
)




# Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
consolidado_pj['Riesgo inherente del cliente_categoria'] = consolidado_pj['Riesgo inherente del cliente'].apply(lambda x: asignar_valor_RI(x))




# %%  ESTIMACIÓN RIESGO INHERENTE RESIDUAL


crucevigia = '''SELECT Documento,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[CRUCEVIGIA]'''
crucevigia = cx.read_sql(conn = sql_connection, query = crucevigia, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])
crucevigia['Documento']=crucevigia['Documento'].fillna(0).astype('int64').astype(str).str.strip()
crucevigia = crucevigia.groupby('Documento').first().reset_index()




consolidado_pj['Riesgo inherente compuesto por producto_residual'] = np.where((consolidado_pj['DocumentoCliente'].isin(crucevigia['Documento'])) & (consolidado_pj['Riesgo inherente compuesto por producto'] >= 2), consolidado_pj['Riesgo inherente compuesto por producto']-1, consolidado_pj['Riesgo inherente compuesto por producto'])
consolidado_pj['PonderacionAtributosAE_residual'] = np.where((consolidado_pj['DocumentoCliente'].isin(crucevigia['Documento'])) & (consolidado_pj['PonderacionAtributosAE'] >= 2), consolidado_pj['PonderacionAtributosAE']-1, consolidado_pj['PonderacionAtributosAE'])
consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL_residual'] = np.where((consolidado_pj['DocumentoCliente'].isin(crucevigia['Documento'])) & (consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] >= 2), consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL']-1, consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'])
consolidado_pj['RiesgoCanal_residual'] = np.where((consolidado_pj['DocumentoCliente'].isin(crucevigia['Documento'])) & (consolidado_pj['RiesgoCanal'] >= 2), consolidado_pj['RiesgoCanal']-1, consolidado_pj['RiesgoCanal'])



consolidado_pj['Riesgo inherente del cliente_residual']  = (
    consolidado_pj['Riesgo inherente compuesto por producto_residual'] * 0.2 +
    consolidado_pj['PonderacionAtributosAE_residual'] * 0.35 +
    consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL_residual'] * 0.2 +
    consolidado_pj['RiesgoCanal_residual'] * 0.25
)


# Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
consolidado_pj['Riesgo inherente del cliente_categoria_residual'] = consolidado_pj['Riesgo inherente del cliente_residual'].apply(lambda x: asignar_valor_RI(x))






# %% [22] ESTIMACIÓN PARÁMETRO DE STURGES CLUTERIZACIÓN

parametro_sturges_pj = math.ceil(math.log(consolidado_pj.shape[0]) / math.log(2))


# Llenar los valores faltantes en la columna "MontoEntradasLarga" con ceros
consolidado_pj['MontoEntradasLarga'].fillna(0, inplace=True)
consolidado_pj['MontoEntradasMedia'].fillna(0, inplace=True)
consolidado_pj['MontoEntradasCorta'].fillna(0, inplace=True)

consolidado_pj['MontoSalidasLarga'].fillna(0, inplace=True)
consolidado_pj['MontoSalidasMedia'].fillna(0, inplace=True)
consolidado_pj['MontoSalidasCorta'].fillna(0, inplace=True)



# Número de clusters que deseas crear
n_clusters = parametro_sturges_pj 

# Crear y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
consolidado_pj['Clasificacion MontoEntradasLarga'] = kmeans.fit_predict(consolidado_pj[['MontoEntradasLarga']])
consolidado_pj['Clasificacion MontoEntradasMedia'] = kmeans.fit_predict(consolidado_pj[['MontoEntradasMedia']])
consolidado_pj['Clasificacion MontoEntradasCorta'] = kmeans.fit_predict(consolidado_pj[['MontoEntradasCorta']])
consolidado_pj['Clasificacion MontoSalidasLarga'] = kmeans.fit_predict(consolidado_pj[['MontoSalidasLarga']])
consolidado_pj['Clasificacion MontoSalidasMedia'] = kmeans.fit_predict(consolidado_pj[['MontoSalidasMedia']])
consolidado_pj['Clasificacion MontoSalidasCorta'] = kmeans.fit_predict(consolidado_pj[['MontoSalidasCorta']])



# entradas_columnas = [ 'Clasificacion MontoEntradasLarga','Clasificacion MontoEntradasMedia','Clasificacion MontoEntradasCorta']

# consolidado_pj['Entradas'] = consolidado_pj[entradas_columnas].max(axis=1)

# salidas_columnas = ['Clasificacion MontoSalidasLarga','Clasificacion MontoSalidasMedia','Clasificacion MontoSalidasCorta']
# salidas_columnas = ['Clasificacion MontoSalidasCorta']

# consolidado_pj['Salidas'] = consolidado_pj[salidas_columnas].max(axis=1)


# tx_columnas = ['Entradas','Salidas']

tx_columnas = ['Clasificacion MontoEntradasCorta','Clasificacion MontoSalidasCorta']


consolidado_pj['Comportamiento transaccional'] = consolidado_pj[tx_columnas].max(axis=1)


maximo_transaccionalidad = consolidado_pj['Comportamiento transaccional'].max()

limite_3_tx = maximo_transaccionalidad * 3 / 4
limite_2_tx = maximo_transaccionalidad * 2 / 4
limite_1_tx = maximo_transaccionalidad * 1 / 4


def asignar_valor_tx(inicial):
    if inicial <= limite_1_tx :
        return 1
    elif inicial <= limite_2_tx:
        return 2
    elif inicial <= limite_3_tx:
        return 3
    else:
        return 4

# Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
consolidado_pj['Comportamiento transaccional_categoria'] = consolidado_pj['Comportamiento transaccional'].apply(lambda x: asignar_valor_tx(x))





# %% [23] DEFINICIÓN DEL CUADRANTE ASOCIADO AL CLIENTE


# def cuadrante(row):
#     RI = row['Riesgo inherente del cliente']
#     TX = row['Comportamiento transaccional']
    
#     linea1 = int(parametro_sturges_pj / 3)
#     linea2 = linea1 * 2
#     li_ri= 1.5
#     lm_ri = 2.5
    
#     if RI <= li_ri:
#         if TX <= linea1:
#             return 1
#         elif TX < linea2:
#             return 4
#         else:
#             return 7
#     elif li_ri < RI <= lm_ri:
#         if TX <= linea1:
#             return 2
#         elif TX < linea2:
#             return 5
#         else:
#             return 8
#     else:
#         if TX <= linea1:
#             return 3
#         elif TX < linea2:
#             return 6
#         else:
#             return 9


def cuadrante(row):
    RI = row['Riesgo inherente del cliente']
    TX = row['Comportamiento transaccional']
    
    linea1 = int(parametro_sturges_pj / 4)
    linea2 = linea1 * 2
    linea3 = linea1 * 3
    li_ri= 1
    lm_ri = 2
    lm_tres = 3
    
    if RI <= li_ri and TX <= linea1:
        return 1
    elif RI <= li_ri and TX <= linea2:
        return 5
    elif RI <= li_ri and TX <= linea3:
        return 9
    elif RI <= li_ri and TX > linea3:
        return 13
    
    elif RI <= lm_ri and TX <= linea1:
        return 2
    elif RI <= lm_ri and TX <= linea2:
        return 6
    elif RI <= lm_ri and TX <= linea3:
        return 10
    elif RI <= lm_ri and TX > linea3:
        return 14
    
    elif RI <= lm_tres and TX <= linea1:
        return 3
    elif RI <= lm_tres and TX <= linea2:
        return 7
    elif RI <= lm_tres and TX <= linea3:
        return 11
    elif RI <= lm_tres and TX > linea3:
        return 15
    
    elif RI > lm_tres and TX <= linea1:
        return 4
    elif RI > lm_tres and TX <= linea2:
        return 8
    elif RI > lm_tres and TX <= linea3:
        return 12
    elif RI > lm_tres and TX > linea3:
        return 16
    
    
    # if RI <= li_ri:
    #     if TX <= linea1:
    #         return 1
    #     elif TX < linea2:
    #         return 5
    #     elif TX < linea3:
    #         return 9
    #     else:
    #         return 13
    # elif li_ri < RI <= lm_ri:
    #     if TX <= linea1:
    #         return 2
    #     elif TX < linea2:
    #         return 6
    #     elif TX < linea3:
    #         return 10
    #     else:
    #         return 14
    # else:
    #     if TX <= linea1:
    #         return 4
    #     elif TX < linea2:
    #         return 8
    #     elif TX < linea3:
    #         return 12
    #     else:
    #         return 16


# def cuadrante(row):
#     RI = row['Riesgo inherente del cliente']
#     TX = row['Comportamiento transaccional']

#     linea1 = int(parametro_sturges_pj / 4)
#     linea2 = linea1 * 2
#     linea3 = linea1 * 3

#     cuadrante = 0

#     if RI <= 1:
#         cuadrante += 0
#     elif RI <= 2:
#         cuadrante += 4
#     elif RI <= 3:
#         cuadrante += 8
#     elif RI > 3:
#         cuadrante += 12

#     if TX <= linea1:
#         cuadrante += 1
#     elif TX <= linea2:
#         cuadrante += 2
#     elif TX <= linea3:
#         cuadrante += 3
#     elif TX > linea3:
#         cuadrante += 4

#     return cuadrante


# Aplicar la función a cada fila del DataFrame "consolidado_pj"
consolidado_pj['Cuadrante'] = consolidado_pj.apply(cuadrante, axis=1)



# jose=consolidado_pj[['Riesgo inherente del cliente','Comportamiento transaccional','Riesgo inherente del cliente_categoria','Comportamiento transaccional_categoria','Cuadrante']]

# jose['Cuadrante'].value_counts()

# alertamiento si cambió de cuadrante con respecto a la segmentación anterior


from datetime import datetime
from datetime import timedelta

# Obtener la fecha actual
fecha_actual = datetime.now()
mes_anterior=(datetime.now()-timedelta(days=30)).month
# Obtener el año y el mes
anio = fecha_actual.year


if mes_anterior == 1:
    mes_anterior = "12"
    anio = anio - 1
elif mes_anterior == 2:
    mes_anterior = "01"
elif mes_anterior == 3:
    mes_anterior = "02"
elif mes_anterior == 4:
    mes_anterior = "03"
if mes_anterior == 5:
    mes_anterior = "04"
elif mes_anterior == 6:
    mes_anterior = "05"
elif mes_anterior == 7:
    mes_anterior = "06"
elif mes_anterior == 8:
    mes_anterior = "07"
elif mes_anterior == 9:
    mes_anterior = "08"
elif mes_anterior == 10:
    mes_anterior = "09"
elif mes_anterior == 11:
    mes_anterior = "10"
elif mes_anterior ==12:
    mes_anterior = "11"

# anio=2023
# mes_anterior=11


if segmentacion == "PJ":
    valor=str('ClienteProductoJuridico_')+str(anio)+str(mes_anterior)+str('_transformado')
    
else:
    valor=str('ClienteProductoNatural_')+str(anio)+str(mes_anterior)+str('_transformado')


segmentacion_anterior = f'''SELECT DocumentoCliente,Cuadrante as cuadranteanterior,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[{valor}]'''
segmentacion_anterior = cx.read_sql(conn = sql_connection, query = segmentacion_anterior, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

consolidado_pj = pd.merge(consolidado_pj,segmentacion_anterior,on='DocumentoCliente',how='left')

consolidado_pj['Cuadrante'] = consolidado_pj['Cuadrante'].fillna(0)
consolidado_pj['cuadranteanterior'] = consolidado_pj['cuadranteanterior'].fillna(0)

# Realizar la comparación después de rellenar los valores nulos
consolidado_pj['Alertamiento_cambio_cuadrante'] = np.where(consolidado_pj['Cuadrante'] != consolidado_pj['cuadranteanterior'], 1, 0)

########################## validacion fraudes 


# %% [24] VALIDACIÓN INFORMACIÓN DE FRAUDES AUDITORÍA


# Configurar la cadena de conexión (como se mencionó en respuestas anteriores)

connection_string = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};Integrated Security={integrated_security}'

# Intentar establecer la conexión
try:
    conn = pyodbc.connect(connection_string)
    print('Conexión exitosa a SQL Server')
    
    # Consulta SQL que deseas ejecutar
    query = "SELECT id_cliente as DocumentoCliente, tipologia FROM [AUDITORIA_COMPARTIDA].[PV].[Bd_Fraudes]"
    
    # Ejecutar la consulta y almacenar los resultados en un DataFrame
    auditoria = pd.read_sql(query, conn)
    
    # Imprimir los resultados (opcional)
    # print(df)
    
    # No olvides cerrar la conexión cuando hayas terminado
    conn.close()

except Exception as e:
    print(f'Error al conectar a SQL Server: {str(e)}')



auditoria['DocumentoCliente'] = auditoria['DocumentoCliente'].apply(lambda x: '{:.0f}'.format(x)).astype(str).str.strip()


auditoria['tipologia']=auditoria['tipologia'].str.replace('[^a-zA-Z ]', '', regex=True).str.upper().str.strip()







# Supongamos que tienes un DataFrame 'auditoria' con una columna 'tipologia'

# Lista de correcciones a realizar
correcciones = [
    ('SOSPECHA POR INCUMPLIMIENTO EN PAGO', 'SOSPECHA POR INCUMPLIMIENTO EN PAGOS'),
    ('DOCUMENTACIN FALSA EN RADICACIN', 'DOCUMENTACION FALSA EN RADICACION'),
    ('DOCUMENTACION FALSA EN RADICACIN', 'DOCUMENTACION FALSA EN RADICACION'),
    ('SUPLANTACIN','SUPLANTACION')
]

# Realiza las correcciones en un bucle
for antigua, nueva in correcciones:
    auditoria['tipologia'] = np.where(auditoria['tipologia'].str.contains(antigua, case=False, na=False), nueva, auditoria['tipologia'])



# Utiliza la función pivot_table para pivotear el DataFrame
auditoria = auditoria.pivot_table(auditoria, index='DocumentoCliente', columns='tipologia', aggfunc='size', fill_value=0).reset_index()


# Lista de las columnas en las que deseas verificar si algún valor es mayor que 0
columnas_verificar = [
    'DOCUMENTACION FALSA',
      'SUPLANTACION',
      'SOSPECHA POR INCUMPLIMIENTO EN PAGOS',
      'EMPRESA FACHADA',
      'DOCUMENTACION FALSA EN RADICACION',
      'PREVENCION',
      'DOCUMENTACION FALSA INCOCREDITO'
]


# columnas_verificar=[[auditoria.columns]]

# # Verificar si al menos una columna es mayor que 0 para cada fila
# auditoria['ValidacionFraudes'] = auditoria[columnas_verificar].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)

# columnas_verificar = auditoria.columns.tolist()

# # Verificar si al menos una columna es mayor que 0 para cada fila
# auditoria['ValidacionFraudes'] = auditoria[columnas_verificar].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)





import pandas as pd

# Assuming 'auditoria' contains a mix of numeric and string columns
# Specify the columns to convert to numeric (excluding 'Documento' if it exists)
columns_to_convert = [col for col in columnas_verificar if col != 'DocumentoCliente' and col in auditoria.columns]

# Convert selected columns to numeric (errors='coerce' will convert non-numeric values to NaN)
auditoria[columns_to_convert] = auditoria[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Check if at least one value is greater than 0 for each row
auditoria['ValidacionFraudes'] = auditoria[columns_to_convert].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)





consolidado_pj = pd.merge(consolidado_pj,auditoria,on='DocumentoCliente',how='left')



# reporte de PEP 

# %% [25] VALIDACIÓN DE PERSONAS EXPUESTAS PUBLICAMENTE

server_name_vigia = 'FABOGLOGDB\MSSQLPRLOGS,53978'
database_name_vigia = 'VIGIAV2'

connection_string_vigia = f'DRIVER={{SQL Server}};SERVER={server_name_vigia};DATABASE={database_name_vigia};Integrated Security={integrated_security}'

# Intentar establecer la conexión
try:
    conn_vigia = pyodbc.connect(connection_string_vigia)
    print('Conexión exitosa a SQL Server')
    
    # Consulta SQL que deseas ejecutar
    query = "SELECT distinct [NUMIDE] as DocumentoCliente  FROM [VIGIAV2].[VIGIAV2].[VSDNLIST] where TIPOLISTA ='PERSONAS POLITICAMENTE EXPUESTAS'"
    
    # Ejecutar la consulta y almacenar los resultados en un DataFrame
    pep = pd.read_sql(query, conn_vigia)
    
    # Imprimir los resultados (opcional)
    # print(df)
    
    # No olvides cerrar la conexión cuando hayas terminado
    conn_vigia.close()

except Exception as e:
    print(f'Error al conectar a SQL Server: {str(e)}')




# pep = '''SELECT NUMIDE as DocumentoCliente,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
#   FROM [ModelosSARLAFT].[dbo].[BASEPEP]'''
# pep = cx.read_sql(conn = sql_connection, query = pep, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

import numpy as np

consolidado_pj['pep'] = consolidado_pj['DocumentoCliente'].isin(pep['DocumentoCliente']).astype(int)
# consolidado_pj['pep']=0

consolidado_pj['FechaUltimaActualizacionCore'] = pd.to_datetime(consolidado_pj['FechaUltimaActualizacionCore'])
consolidado_pj['dias_de_ultima_actualizacion'] = (pd.to_datetime('today') - consolidado_pj['FechaUltimaActualizacionCore']).dt.days


# if segmentacion == "PJ":
#     consolidado_pj['Requiere_actualizacion'] = np.where(consolidado_pj['dias_de_ultima_actualizacion']<=(365*3),'No','Si')
# else:
#     consolidado_pj['Requiere_actualizacion'] = np.where(consolidado_pj['dias_de_ultima_actualizacion']<=(365),'No','Si')




consolidado_pj['Requiere_actualizacion'] = np.where(
    (consolidado_pj['Riesgo inherente del cliente_categoria'] >= 3) & 
    (consolidado_pj['dias_de_ultima_actualizacion'] <= 365),
    'Si',
    'No'
)


consolidado_pj['Requiere_actualizacion'] = np.where(
    (consolidado_pj['Riesgo inherente del cliente_categoria'] == 2) & 
    (consolidado_pj['dias_de_ultima_actualizacion'] <= 365*2),
    'Si',
    consolidado_pj['Requiere_actualizacion']
)

consolidado_pj['Requiere_actualizacion'] = np.where(
    (consolidado_pj['Riesgo inherente del cliente_categoria'] == 1) & 
    (consolidado_pj['dias_de_ultima_actualizacion'] <= 365*3),
    'Si',
    consolidado_pj['Requiere_actualizacion']
)


consolidado_pj['Requiere_actualizacion'] = np.where(
    (consolidado_pj['Riesgo inherente del cliente_categoria'] == 1) & 
    (consolidado_pj['dias_de_ultima_actualizacion'] <= 365),
    'Si',
    consolidado_pj['Requiere_actualizacion']
)





# validación extranjeros 

# %% [26] VALIDACIÓN DE EXTRANJEROS


# extranjeros = '''SELECT [DocumentoCliente],ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
# FROM [ModelosSARLAFT].[dbo].[ConsolidadoExtranjeros]'''
# extranjeros = cx.read_sql(conn = sql_connection, query = extranjeros, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

# consolidado_pj['extranjero'] = consolidado_pj['DocumentoCliente'].isin(extranjeros['DocumentoCliente']).astype(int)


# ## prueba 

# extranjeros = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
# FROM [ModelosSARLAFT].[dbo].[ConsolidadoExtranjeros]'''
# extranjeros = cx.read_sql(conn = sql_connection, query = extranjeros, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])
# consolidado_pj['extranjero'] = consolidado_pj['DocumentoCliente'].isin(extranjeros['DocumentoCliente']).astype(int)
# consolidado_pj = pd.merge(consolidado_pj,extranjeros,on='DocumentoCliente',how='left')


# # estimacion pais 
# consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] = consolidado_pj.apply(lambda row: row['Nivel de riesgo JURISDICCION NACIONAL'] if pd.isnull(row['Clasificacion_pais']) or row['Nivel de riesgo JURISDICCION NACIONAL'] > row['Clasificacion_pais'] else row['Clasificacion_pais'], axis=1)


# # estimacion departamento - municipio 

# consolidado_pj['depto_muni']= consolidado_pj['Departamento'] + '-' + consolidado_pj['Municipio']


# depto_muni = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
# FROM [ModelosSARLAFT].[dbo].[RiesgoDeptoMuni]'''
# depto_muni = cx.read_sql(conn = sql_connection, query = depto_muni, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])


# consolidado_pj = pd.merge(consolidado_pj,depto_muni,on='depto_muni',how='left')

# consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] = consolidado_pj.apply(lambda row: row['Nivel de riesgo JURISDICCION NACIONAL'] if pd.isnull(row['NIVEL RIESGO FINAL']) or row['Nivel de riesgo JURISDICCION NACIONAL'] > row['NIVEL RIESGO FINAL'] else row['NIVEL RIESGO FINAL'], axis=1)



# %% [20] ESTIMACIÓN RIESGO INHERENTE DEL CLIENTE

# # estimación riesgo inherente 

# consolidado_pj['Riesgo inherente del cliente']  = (
#     consolidado_pj['Riesgo inherente compuesto por producto'] * 0.2 +
#     consolidado_pj['PonderacionAtributosAE'] * 0.35 +
#     consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL'] * 0.2 +
#     consolidado_pj['RiesgoCanal'] * 0.25
# )




# maximo_RI = consolidado_pj['Riesgo inherente del cliente'].max()

# limite_3_RI = maximo_RI * 3 / 4
# limite_2_RI = maximo_RI * 2 / 4
# limite_1_RI = maximo_RI * 1 / 4


# def asignar_valor_RI(inicial):
#     if inicial <= limite_1_RI :
#         return 1
#     elif inicial <= limite_2_RI:
#         return 2
#     elif inicial <= limite_3_RI:
#         return 3
#     else:
#         return 4

# # Aplicar la función a la columna 'inicial' para crear una nueva columna 'nueva_columna'
# consolidado_pj['Riesgo inherente del cliente'] = consolidado_pj['Riesgo inherente del cliente'].apply(lambda x: asignar_valor_RI(x))


# Supongamos que tienes un DataFrame llamado df

# Lista de las columnas que deseas mantener
columnas_deseadas = [
    'DocumentoCliente',
    'Tipo_identificacion',
    'Ahorro de la red',
    'CDT',
    'Cartera vendida',
    'Castigados',
    'Crédito hipotecario',
    'Crédito libre inversión',
    'Crédito vehículo',
    'Cuenta corriente',
    'FlexiDigital',
    'Leasing vehículo',
    'Libranza',
    'Maquina agrícola',
    'Nomina Finandina',
    'Otros activos',
    'Otros ahorros',
    'Plan mayor',
    'Redescuentos',
    'TDC Digital',
    'TDC Física',
    'FechaApertura',
    'NombreCliente',
    'DireccionActual',
    'Direccionlargo',
    'DireccionActual_valida',
    'Celular',
    'Celularlargo',
    'Celular_valido',
    'FechaVinculacion',
    'MontoIngresos',
    'Egresos',
    'Pasivos',
    'Correo',
    'Correo_valido',
    'FechaUltimaActualizacionCore',
    'Origen Nacional',
    'CiudadActual',
    'Departamento',
    'Municipio',
    'TipoPersona',
    'CodigoCIIU',
    'ActividadEconomica',
    'Grupo general actividad economica',    
    'seccion',
    'ciiu_2',
    'ciiu_4',
    'AE_DANE',  
    'DescripcionSucursal',
    'CanalEntrada',
    'MontoEntradasLarga',
    'CantidadEntradasLarga',
    'MediaEntradasLarga',
    'MedianaEntradasLarga',
    'MontoSalidasLarga',
    'CantidadSalidasLarga',
    'MediaSalidasLarga',
    'MedianaSalidasLarga',
    'MontoEntradasMedia',
    'CantidadEntradasMedia',
    'MediaEntradasMedia',
    'MedianaEntradasMedia',
    'MontoSalidasMedia',
    'CantidadSalidasMedia',
    'MediaSalidasMedia',
    'MedianaSalidasMedia',
    'MontoEntradasCorta',
    'CantidadEntradasCorta',
    'MediaEntradasCorta',
    'MedianaEntradasCorta',
    'MontoSalidasCorta',
    'CantidadSalidasCorta',
    'MediaSalidasCorta',
    'MedianaSalidasCorta',
    'VulnerabilidadLavadoActivos',
    'VulnerabilidadTerrorismo',
    'RiesgoJurisdiccion',
    'RiesgoCanal',
    'RiesgoProducto',
    'AlertaPerfiltxahorrosalidas',
    'AlertaPerfiltxtdc',
    'AlertaVCRec',
    'AlertaDSrest',
    'Entrada',
    'Otros',
    'Salida',
    'Activo',
    'dias_de_ultima_tx_entrada',
    'dias_de_ultima_tx_salida',
    'dias_de_ultima_tx_otros',
    'RIESGO PRODUCTO',
    'Nivel de riesgo JURISDICCION NACIONAL',
    'PonderacionAtributosAE',
    'Riesgo inherente del cliente',
    'Riesgo inherente del cliente_categoria',
    'Riesgo inherente compuesto por producto',
    'Comportamiento transaccional',
    'Cuadrante',
    'DOCUMENTACION FALSA EN RADICACION',
    # 'DOCUMENTACION FALSA',
    'DOCUMENTACION FALSA INCOCREDITO',
    'EMPRESA FACHADA',
    'PREVENCION',
    'SOSPECHA POR INCUMPLIMIENTO EN PAGOS',
    'SUPLANTACION',
    'ValidacionFraudes',
    'pep',
    'extranjero',
    'Incomercio',
    'Alertamiento_cambio_cuadrante',
    'dias_de_ultima_actualizacion',
    'Requiere_actualizacion',
    'Riesgo inherente del cliente_residual',
    'Riesgo inherente compuesto por producto_residual',
    'PonderacionAtributosAE_residual',
    'Nivel de riesgo JURISDICCION NACIONAL_residual',
    'RiesgoCanal_residual',
    'Riesgo inherente del cliente_categoria_residual'
    
]





# consolidado_pj['Riesgo inherente del cliente_categoria'].value_counts()


# Filtra el DataFrame para incluir solo las columnas deseadas
consolidado_pj = consolidado_pj[columnas_deseadas]

columnas_fraude = ['DOCUMENTACION FALSA EN RADICACION'# , 'DOCUMENTACION FALSA'
                   , 'DOCUMENTACION FALSA INCOCREDITO',
            'EMPRESA FACHADA', 'PREVENCION', 'SOSPECHA POR INCUMPLIMIENTO EN PAGOS', 'SUPLANTACION']

# Itera a través de las columnas y actualiza los valores en el DataFrame original
for columna in columnas_fraude:
    consolidado_pj[columna] = (consolidado_pj[columna] > 0).astype(int)



columnas_a_reemplazar3 = ["Nivel de riesgo JURISDICCION NACIONAL", "PonderacionAtributosAE", "Riesgo inherente compuesto por producto", "RiesgoCanal"]
valor_a_insertar = 4

for columna in columnas_a_reemplazar3:
    consolidado_pj[columna].fillna(valor_a_insertar, inplace=True)


# Crea un imputador que reemplace los NaN con la media de la columna
imputer = SimpleImputer(strategy='mean')

# Aplica la imputación solo a la columna 'ingresos'
consolidado_pj['MontoIngresos'] = imputer.fit_transform(consolidado_pj[['MontoIngresos']])


# Grupos autónomos de riesgo y digitalización


# consolidado_pj['Grupo actividad riesgo']=np.where(consolidado_pj['Riesgo actividad económica'].isna(),3,consolidado_pj['Riesgo actividad económica'])

## optimización de memoria de cara al proceso de clusterización


dataframes_a_conservar = ["consolidado_pj",'df']

# Obtener una lista de todas las variables en el espacio de nombres actual
variables = list(globals())

# Eliminar los DataFrames que no están en la lista de conservar
for variable in variables:
    if isinstance(globals()[variable], pd.DataFrame) and variable not in dataframes_a_conservar:
        del globals()[variable]



consolidado_pj = consolidado_pj.groupby('DocumentoCliente').first().reset_index()





# ### Datos de agrupacion

columnas_agrupacion_general=['MontoEntradasCorta','MontoSalidasCorta','MontoIngresos']
columnas_agrupacion_cluster_1=['MontoEntradasCorta']
columnas_agrupacion_cluster_2=['MontoSalidasCorta']
columnas_agrupacion_cluster_3=['MontoIngresos']



# Escalamiento

# Normalización

datos_cliente_activo_escalados=consolidado_pj.copy(deep=True)
scaler_normal=QuantileTransformer(n_quantiles=500)
datos_cliente_activo_escalados[columnas_agrupacion_general]=scaler_normal.fit_transform(datos_cliente_activo_escalados[columnas_agrupacion_general])


# ### Formato de datos

datos_cliente_activo_final=consolidado_pj.copy(deep=True)


# ### Clusters #1

numero_clusters_1=6
k_vecindarios_1=KMeans(n_clusters=numero_clusters_1,random_state=42)
k_vecindarios_ajuste_1=k_vecindarios_1.fit(datos_cliente_activo_escalados[columnas_agrupacion_cluster_1])
k_vecindarios_prediccion_1=k_vecindarios_ajuste_1.predict(datos_cliente_activo_escalados[columnas_agrupacion_cluster_1])+1
datos_cliente_activo_escalados['Cluster 1']=k_vecindarios_prediccion_1
datos_cliente_activo_final['Cluster 1']=k_vecindarios_prediccion_1


# ### Clusters #2

numero_clusters_2=6
k_vecindarios_2=KMeans(n_clusters=numero_clusters_2,random_state=42)
k_vecindarios_ajuste_2=k_vecindarios_2.fit(datos_cliente_activo_escalados[columnas_agrupacion_cluster_2])
k_vecindarios_prediccion_2=k_vecindarios_ajuste_2.predict(datos_cliente_activo_escalados[columnas_agrupacion_cluster_2])+1
datos_cliente_activo_escalados['Cluster 2']=k_vecindarios_prediccion_2
datos_cliente_activo_final['Cluster 2']=k_vecindarios_prediccion_2


# ### Clusters #3

numero_clusters_3=6
k_vecindarios_3=KMeans(n_clusters=numero_clusters_3,random_state=42)
k_vecindarios_ajuste_3=k_vecindarios_3.fit(datos_cliente_activo_escalados[columnas_agrupacion_cluster_3])
k_vecindarios_prediccion_3=k_vecindarios_ajuste_3.predict(datos_cliente_activo_escalados[columnas_agrupacion_cluster_3])+1
datos_cliente_activo_escalados['Cluster 3']=k_vecindarios_prediccion_3
datos_cliente_activo_final['Cluster 3']=k_vecindarios_prediccion_3



# ## Creación de grupos de control

# %% [27] CREACIÓN DE GRUPOS DE CONTROL

# ### Rangos de grupos

clusters_entrada_corto=sorted(np.unique(datos_cliente_activo_final['Cluster 1']))
clusters_salida_corto=sorted(np.unique(datos_cliente_activo_final['Cluster 2']))
clusters_financiero=sorted(np.unique(datos_cliente_activo_final['Cluster 3']))

rangos_transaccional_entrada_corto=[]
for numero_cluster in clusters_entrada_corto:
    rangos_transaccional_entrada_corto.append(round(datos_cliente_activo_final.loc[datos_cliente_activo_final['Cluster 1']==numero_cluster]['MontoEntradasCorta'].mean()))

rangos_transaccional_salida_corto=[]
for numero_cluster in clusters_salida_corto:
    rangos_transaccional_salida_corto.append(round(datos_cliente_activo_final.loc[datos_cliente_activo_final['Cluster 2']==numero_cluster]['MontoSalidasCorta'].mean()))

rangos_financiero=[]
for numero_cluster in clusters_financiero:
    rangos_financiero.append(round(datos_cliente_activo_final.loc[datos_cliente_activo_final['Cluster 3']==numero_cluster]['MontoIngresos'].mean()))

rangos_transaccional_entradas_corto=sorted(rangos_transaccional_entrada_corto)
rangos_transaccional_salidas_corto=sorted(rangos_transaccional_salida_corto)
rangos_financiero=sorted(rangos_financiero)


# ### Condiciones y etiquetas

condiciones_transaccionalidad_entradas_corto=[]

for i in range(len(rangos_transaccional_entradas_corto)):
    if i<len(rangos_transaccional_entradas_corto)-1:
        condiciones_transaccionalidad_entradas_corto.append((datos_cliente_activo_final['MontoEntradasCorta']>=rangos_transaccional_entradas_corto[i])&(datos_cliente_activo_final['MontoEntradasCorta']<rangos_transaccional_entradas_corto[i+1]))
    else:
        condiciones_transaccionalidad_entradas_corto.append((datos_cliente_activo_final['MontoEntradasCorta']>=rangos_transaccional_entradas_corto[i]))

condiciones_transaccionalidad_salidas_corto=[]

for i in range(len(rangos_transaccional_salidas_corto)):
    if i<len(rangos_transaccional_salidas_corto)-1:
        condiciones_transaccionalidad_salidas_corto.append((datos_cliente_activo_final['MontoSalidasCorta']>=rangos_transaccional_salidas_corto[i])&(datos_cliente_activo_final['MontoSalidasCorta']<rangos_transaccional_salidas_corto[i+1]))
    else:
        condiciones_transaccionalidad_salidas_corto.append((datos_cliente_activo_final['MontoSalidasCorta']>=rangos_transaccional_salidas_corto[i]))

condiciones_financieras=[]

for i in range(len(rangos_financiero)):
    if i<len(rangos_financiero)-1:
        condiciones_financieras.append((datos_cliente_activo_final['MontoIngresos']>=rangos_financiero[i])&(datos_cliente_activo_final['MontoIngresos']<rangos_financiero[i+1]))
    else:
        condiciones_financieras.append((datos_cliente_activo_final['MontoIngresos']>=rangos_financiero[i]))


etiquetas_transaccionalidad_entradas=list(range(1,len(condiciones_transaccionalidad_entradas_corto)+1))
etiquetas_transaccionalidad_salidas=list(range(1,len(condiciones_transaccionalidad_salidas_corto)+1))
etiquetas_financieras=list(range(1,len(rangos_financiero)+1))


# ### Generación de grupos de control

datos_cliente_activo_final['Grupo transaccional entrada']=np.select(condiciones_transaccionalidad_entradas_corto,etiquetas_transaccionalidad_entradas,default=1)
datos_cliente_activo_final['Grupo transaccional salida']=np.select(condiciones_transaccionalidad_salidas_corto,etiquetas_transaccionalidad_salidas,default=1)
datos_cliente_activo_final['Grupo financiero']=np.select(condiciones_financieras,etiquetas_financieras,default=1)


# ### Grupos de riesgo consolidados

# datos_cliente_activo_final['Grupo riesgo cliente consolidado']=(datos_cliente_activo_final['Grupo actividad riesgo']+datos_cliente_activo_final['Grupo transaccional entrada']+datos_cliente_activo_final['Grupo transaccional salida']+datos_cliente_activo_final['Grupo financiero'])/4
# datos_cliente_activo_final['Grupo riesgo jurisdicción consolidado']=(datos_cliente_activo_final['Grupo jurisdicción riesgo']+datos_cliente_activo_final['Grupo transaccional entrada']+datos_cliente_activo_final['Grupo transaccional salida']+datos_cliente_activo_final['Grupo financiero'])/4
# datos_cliente_activo_final['Grupo riesgo canal consolidado']=(datos_cliente_activo_final['Grupo canal riesgo']+datos_cliente_activo_final['Grupo transaccional entrada']+datos_cliente_activo_final['Grupo transaccional salida']+datos_cliente_activo_final['Grupo financiero'])/4
# datos_cliente_activo_final['Grupo riesgo producto consolidado']=(datos_cliente_activo_final['Grupo producto riesgo']+datos_cliente_activo_final['Grupo transaccional entrada']+datos_cliente_activo_final['Grupo transaccional salida']+datos_cliente_activo_final['Grupo financiero'])/4


# datos_cliente_activo_final['Grupo riesgo consolidado total']=(datos_cliente_activo_final['Grupo riesgo cliente consolidado']+datos_cliente_activo_final['Grupo riesgo jurisdicción consolidado']+datos_cliente_activo_final['Grupo riesgo canal consolidado']+datos_cliente_activo_final['Grupo riesgo producto consolidado'])/4



# def etiquetar_valor(valor):
#     if valor <= 1.26:
#         return 'Bajo'
#     elif valor <= 4.3:
#         return 'Medio'
#     else:
#         return 'Alto'


# datos_cliente_activo_final['Categorias Grupo riesgo consolidado total'] =  datos_cliente_activo_final['Grupo riesgo consolidado total'].apply(etiquetar_valor)


# ## Métricas y estadística por grupo

# ### Métricas de las clusterizaciones individual

# %% [28] MÉTRICAS ESTADÍSTICAS MODELO DE APRENDIZAJE NO SUPERVISADO


silhouette_score_kmeans_1=silhouette_score(datos_cliente_activo_escalados[columnas_agrupacion_cluster_1],k_vecindarios_prediccion_1)
dunn_score_kmeans_1=davies_bouldin_score(datos_cliente_activo_escalados[columnas_agrupacion_cluster_1],k_vecindarios_prediccion_1)

silhouette_score_kmeans_2=silhouette_score(datos_cliente_activo_escalados[columnas_agrupacion_cluster_2],k_vecindarios_prediccion_2)
dunn_score_kmeans_2=davies_bouldin_score(datos_cliente_activo_escalados[columnas_agrupacion_cluster_2],k_vecindarios_prediccion_2)

silhouette_score_kmeans_3=silhouette_score(datos_cliente_activo_escalados[columnas_agrupacion_cluster_3],k_vecindarios_prediccion_3)
dunn_score_kmeans_3=davies_bouldin_score(datos_cliente_activo_escalados[columnas_agrupacion_cluster_3],k_vecindarios_prediccion_3)

diccionario_metricas_cluster_1={'Coeficiente silueta':[silhouette_score_kmeans_1],'Coeficiente Davis':[dunn_score_kmeans_1],'Cluster':1}
diccionario_metricas_cluster_2={'Coeficiente silueta':[silhouette_score_kmeans_2],'Coeficiente Davis':[dunn_score_kmeans_2],'Cluster':2}
diccionario_metricas_cluster_3={'Coeficiente silueta':[silhouette_score_kmeans_3],'Coeficiente Davis':[dunn_score_kmeans_3],'Cluster':3}

datos_metricas_cluster_1=pd.DataFrame(data=diccionario_metricas_cluster_1)
datos_metricas_cluster_2=pd.DataFrame(data=diccionario_metricas_cluster_2)
datos_metricas_cluster_3=pd.DataFrame(data=diccionario_metricas_cluster_3)

datos_metricas_cluster_individual=pd.concat([datos_metricas_cluster_1,datos_metricas_cluster_2,datos_metricas_cluster_3]).reset_index(drop=True)





# %% [29] ANÁLISIS EXPLORATORIO GRUPOS DE CONTROL


# ### Estadística de grupos de control

# datos_cliente_activo_final.loc[datos_cliente_activo_final['Grupo riesgo canal consolidado'].isna()]

# estadistica_grupo_entradas=datos_cliente_activo_final.groupby(by='Grupo transaccional entrada').agg(conteo_grupo=('Documento','size'),minimo_entradas=('Entradas corto','min'),minimo_salidas=('Salidas corto','min'),minimo_ingresos=('Ingresos','min'),maximo_entradas=('Entradas corto','max'),maximo_salidas=('Salidas corto','max'),maximo_ingresos=('Ingresos','max'),media_entradas=('Entradas corto','mean'),media_salidas=('Salidas corto','mean'),media_ingresos=('Ingresos','mean'),mediana_entradas=('Entradas corto','median'),mediana_salidas=('Salidas corto','median'),mediana_ingresos=('Ingresos','median'),desviacion_entradas=('Entradas corto','std'),desviacion_salidas=('Salidas corto','std'),desviacion_ingresos=('Ingresos','std')).reset_index().rename(columns={'Grupo transaccional entrada':'Grupo específico'}).astype(np.int64)





estadistica_grupo_entradas = datos_cliente_activo_final.groupby(by='Grupo transaccional entrada').agg(
    conteo_grupo=('DocumentoCliente', 'size'),
    minimo_entradas=('MontoEntradasCorta', 'min'),
    minimo_salidas=('MontoSalidasCorta', 'min'),
    minimo_ingresos=('MontoIngresos', 'min'),
    maximo_entradas=('MontoEntradasCorta', 'max'),
    maximo_salidas=('MontoSalidasCorta', 'max'),
    maximo_ingresos=('MontoIngresos', 'max'),
    media_entradas=('MontoEntradasCorta', 'mean'),
    media_salidas=('MontoSalidasCorta', 'mean'),
    media_ingresos=('MontoIngresos', 'mean'),
    mediana_entradas=('MontoEntradasCorta', 'median'),
    mediana_salidas=('MontoSalidasCorta', 'median'),
    mediana_ingresos=('MontoIngresos', 'median'),
    desviacion_entradas=('MontoEntradasCorta', 'std'),
    desviacion_salidas=('MontoSalidasCorta', 'std'),
    desviacion_ingresos=('MontoIngresos', 'std')
).reset_index().rename(columns={'Grupo transaccional entrada':'Grupo específico'})






# estadistica_grupo_salidas=datos_cliente_activo_final.groupby(by='Grupo transaccional salida').agg(conteo_grupo=('Documento','size'),minimo_entradas=('Entradas corto','min'),minimo_salidas=('Salidas corto','min'),minimo_ingresos=('Ingresos','min'),maximo_entradas=('Entradas corto','max'),maximo_salidas=('Salidas corto','max'),maximo_ingresos=('Ingresos','max'),media_entradas=('Entradas corto','mean'),media_salidas=('Salidas corto','mean'),media_ingresos=('Ingresos','mean'),mediana_entradas=('Entradas corto','median'),mediana_salidas=('Salidas corto','median'),mediana_ingresos=('Ingresos','median'),desviacion_entradas=('Entradas corto','std'),desviacion_salidas=('Salidas corto','std'),desviacion_ingresos=('Ingresos','std')).reset_index().rename(columns={'Grupo transaccional salida':'Grupo específico'}).astype(np.int64)


estadistica_grupo_salidas = datos_cliente_activo_final.groupby(by='Grupo transaccional salida').agg(
    conteo_grupo=('DocumentoCliente', 'size'),
    minimo_entradas=('MontoEntradasCorta', 'min'),
    minimo_salidas=('MontoSalidasCorta', 'min'),
    minimo_ingresos=('MontoIngresos', 'min'),
    maximo_entradas=('MontoEntradasCorta', 'max'),
    maximo_salidas=('MontoSalidasCorta', 'max'),
    maximo_ingresos=('MontoIngresos', 'max'),
    media_entradas=('MontoEntradasCorta', 'mean'),
    media_salidas=('MontoSalidasCorta', 'mean'),
    media_ingresos=('MontoIngresos', 'mean'),
    mediana_entradas=('MontoEntradasCorta', 'median'),
    mediana_salidas=('MontoSalidasCorta', 'median'),
    mediana_ingresos=('MontoIngresos', 'median'),
    desviacion_entradas=('MontoEntradasCorta', 'std'),
    desviacion_salidas=('MontoSalidasCorta', 'std'),
    desviacion_ingresos=('MontoIngresos', 'std')
).reset_index().rename(columns={'Grupo transaccional salida':'Grupo específico'})




estadistica_grupo_financiero=datos_cliente_activo_final.groupby(by='Grupo financiero').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index().rename(columns={'Grupo financiero':'Grupo específico'}).astype(np.int64)

estadistica_grupo_actividad=datos_cliente_activo_final.round().groupby(by='Riesgo inherente del cliente').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index()
estadistica_grupo_jurisdiccion=datos_cliente_activo_final.round().groupby(by='Nivel de riesgo JURISDICCION NACIONAL').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index().rename(columns={'Grupo riesgo jurisdicción consolidado':'Grupo específico'})


estadistica_grupo_canal=datos_cliente_activo_final.round().groupby(by='RiesgoCanal').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index()
estadistica_grupo_producto=datos_cliente_activo_final.round().groupby(by='Riesgo inherente compuesto por producto').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std'))

estadistica_grupo_entradas['Grupo']='Grupo transaccional entradas'
estadistica_grupo_salidas['Grupo']='Grupo transaccional salidas'
estadistica_grupo_financiero['Grupo']='Grupo financiero'
estadistica_grupo_actividad['Grupo']='Riesgo inherente del cliente'
estadistica_grupo_jurisdiccion['Grupo']='Nivel de riesgo JURISDICCION NACIONAL'
estadistica_grupo_canal['Grupo']='RiesgoCanal'
estadistica_grupo_producto['Grupo']='Riesgo inherente compuesto por producto'


estadistica_grupos_individuales=pd.concat([estadistica_grupo_entradas,estadistica_grupo_salidas,estadistica_grupo_financiero,estadistica_grupo_actividad,estadistica_grupo_jurisdiccion,estadistica_grupo_canal,estadistica_grupo_producto]).reset_index(drop=True)


# ### Estadística del grupo final

estadistica_grupo_final=datos_cliente_activo_final.round(1).groupby(by='Riesgo inherente del cliente').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index()


# %% [30] GENERACIÓN DE ALERTAMIENTOS 

# ### Rangos de alertamiento

desviaciones_alertamiento_cliente_entradas=[]
desviaciones_alertamiento_jurisdiccion_entradas=[]
desviaciones_alertamiento_canal_entradas=[]
desviaciones_alertamiento_producto_entradas=[]
medias_alertamiento_cliente_entradas=[]
medias_alertamiento_jurisdiccion_entradas=[]
medias_alertamiento_canal_entradas=[]
medias_alertamiento_producto_entradas=[]
desviaciones_alertamiento_cliente_salidas=[]
desviaciones_alertamiento_jurisdiccion_salidas=[]
desviaciones_alertamiento_canal_salidas=[]
desviaciones_alertamiento_producto_salidas=[]
medias_alertamiento_cliente_salidas=[]
medias_alertamiento_jurisdiccion_salidas=[]
medias_alertamiento_canal_salidas=[]
medias_alertamiento_producto_salidas=[]
desviaciones_alertamiento_cliente_ingresos=[]
desviaciones_alertamiento_jurisdiccion_ingresos=[]
desviaciones_alertamiento_canal_ingresos=[]
desviaciones_alertamiento_producto_ingresos=[]
medias_alertamiento_cliente_ingresos=[]
medias_alertamiento_jurisdiccion_ingresos=[]
medias_alertamiento_canal_ingresos=[]
medias_alertamiento_producto_ingresos=[]

estadistica_grupo_financiero=datos_cliente_activo_final.groupby(by='Grupo financiero').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index().rename(columns={'Grupo financiero':'Grupo específico'}).astype(np.int64)

estadistica_grupo_actividad=datos_cliente_activo_final.round().groupby(by='Riesgo inherente del cliente').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index()
estadistica_grupo_jurisdiccion=datos_cliente_activo_final.round().groupby(by='Nivel de riesgo JURISDICCION NACIONAL').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index().rename(columns={'Grupo riesgo jurisdicción consolidado':'Grupo específico'})


estadistica_grupo_canal=datos_cliente_activo_final.round().groupby(by='RiesgoCanal').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std')).reset_index()
estadistica_grupo_producto=datos_cliente_activo_final.round().groupby(by='Riesgo inherente compuesto por producto').agg(conteo_grupo=('DocumentoCliente','size'),minimo_entradas=('MontoEntradasCorta','min'),minimo_salidas=('MontoSalidasCorta','min'),minimo_MontoIngresos=('MontoIngresos','min'),maximo_entradas=('MontoEntradasCorta','max'),maximo_salidas=('MontoSalidasCorta','max'),maximo_MontoIngresos=('MontoIngresos','max'),media_entradas=('MontoEntradasCorta','mean'),media_salidas=('MontoSalidasCorta','mean'),media_MontoIngresos=('MontoIngresos','mean'),mediana_entradas=('MontoEntradasCorta','median'),mediana_salidas=('MontoSalidasCorta','median'),mediana_MontoIngresos=('MontoIngresos','median'),desviacion_entradas=('MontoEntradasCorta','std'),desviacion_salidas=('MontoSalidasCorta','std'),desviacion_MontoIngresos=('MontoIngresos','std'))






for i in range(1,numero_clusters_1):
    desviaciones_alertamiento_cliente_entradas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1))]['MontoEntradasCorta']),3))
    desviaciones_alertamiento_jurisdiccion_entradas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1))]['MontoEntradasCorta']),3))
    desviaciones_alertamiento_canal_entradas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1))]['MontoEntradasCorta']),3))
    desviaciones_alertamiento_producto_entradas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1))]['MontoEntradasCorta']),3))    
    medias_alertamiento_cliente_entradas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1))]['MontoEntradasCorta']),3))
    medias_alertamiento_jurisdiccion_entradas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1))]['MontoEntradasCorta']),3))
    medias_alertamiento_canal_entradas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1))]['MontoEntradasCorta']),3))
    medias_alertamiento_producto_entradas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1))]['MontoEntradasCorta']),3))
for i in range(1,numero_clusters_2):
    desviaciones_alertamiento_cliente_salidas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1))]['MontoSalidasCorta']),3))
    desviaciones_alertamiento_jurisdiccion_salidas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1))]['MontoSalidasCorta']),3))
    desviaciones_alertamiento_canal_salidas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1))]['MontoSalidasCorta']),3))
    desviaciones_alertamiento_producto_salidas.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1))]['MontoSalidasCorta']),3))
    medias_alertamiento_cliente_salidas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1))]['MontoSalidasCorta']),3))
    medias_alertamiento_jurisdiccion_salidas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1))]['MontoSalidasCorta']),3))
    medias_alertamiento_canal_salidas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1))]['MontoSalidasCorta']),3))
    medias_alertamiento_producto_salidas.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1))]['MontoSalidasCorta']),3))
for i in range(1,numero_clusters_3):
    desviaciones_alertamiento_cliente_ingresos.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1))]['MontoIngresos']),3))    
    desviaciones_alertamiento_jurisdiccion_ingresos.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1))]['MontoIngresos']),3))    
    desviaciones_alertamiento_canal_ingresos.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1))]['MontoIngresos']),3))    
    desviaciones_alertamiento_producto_ingresos.append(round(3*np.std(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1))]['MontoIngresos']),3))      
    medias_alertamiento_cliente_ingresos.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1))]['MontoIngresos']),3))    
    medias_alertamiento_jurisdiccion_ingresos.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1))]['MontoIngresos']),3))    
    medias_alertamiento_canal_ingresos.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1))]['MontoIngresos']),3))    
    medias_alertamiento_producto_ingresos.append(round(np.mean(datos_cliente_activo_final.loc[(datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1))]['MontoIngresos']),3))    


# ### Generación de alertas

# #### Alertas por grupo

datos_cliente_activo_final['Alertamiento cliente de entradas']=0
datos_cliente_activo_final['Alertamiento cliente de salidas']=0
datos_cliente_activo_final['Alertamiento cliente de ingresos']=0
for i,j in zip(range(1,numero_clusters_1),range(len(medias_alertamiento_cliente_entradas))):
    datos_cliente_activo_final['Alertamiento cliente de entradas']=np.where((datos_cliente_activo_final['Alertamiento cliente de entradas']!=0)|(((datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1)))&((datos_cliente_activo_final['MontoEntradasCorta']>(medias_alertamiento_cliente_entradas[j]+desviaciones_alertamiento_cliente_entradas[j]))|(datos_cliente_activo_final['MontoEntradasCorta']<(medias_alertamiento_cliente_entradas[j]-desviaciones_alertamiento_cliente_entradas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento cliente de salidas']=np.where((datos_cliente_activo_final['Alertamiento cliente de salidas']!=0)|(((datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1)))&((datos_cliente_activo_final['MontoSalidasCorta']>(medias_alertamiento_cliente_salidas[j]+desviaciones_alertamiento_cliente_salidas[j]))|(datos_cliente_activo_final['MontoSalidasCorta']<(medias_alertamiento_cliente_salidas[j]-desviaciones_alertamiento_cliente_salidas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento cliente de ingresos']=np.where((datos_cliente_activo_final['Alertamiento cliente de ingresos']!=0)|(((datos_cliente_activo_final['Riesgo inherente del cliente']>=i)&(datos_cliente_activo_final['Riesgo inherente del cliente']<(i+1)))&((datos_cliente_activo_final['MontoIngresos']>(medias_alertamiento_cliente_ingresos[j]+desviaciones_alertamiento_cliente_ingresos[j]))|(datos_cliente_activo_final['MontoIngresos']<(medias_alertamiento_cliente_ingresos[j]-desviaciones_alertamiento_cliente_ingresos[j])))),1,0)
    
datos_cliente_activo_final['Alertamiento jurisdicción de entradas']=0
datos_cliente_activo_final['Alertamiento jurisdicción de salidas']=0
datos_cliente_activo_final['Alertamiento jurisdicción de ingresos']=0
for i,j in zip(range(1,numero_clusters_1),range(len(medias_alertamiento_cliente_entradas))):
    datos_cliente_activo_final['Alertamiento jurisdicción de entradas']=np.where((datos_cliente_activo_final['Alertamiento jurisdicción de entradas']!=0)|(((datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1)))&((datos_cliente_activo_final['MontoEntradasCorta']>(medias_alertamiento_cliente_entradas[j]+desviaciones_alertamiento_cliente_entradas[j]))|(datos_cliente_activo_final['MontoEntradasCorta']<(medias_alertamiento_cliente_entradas[j]-desviaciones_alertamiento_cliente_entradas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento jurisdicción de salidas']=np.where((datos_cliente_activo_final['Alertamiento jurisdicción de salidas']!=0)|(((datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1)))&((datos_cliente_activo_final['MontoSalidasCorta']>(medias_alertamiento_cliente_salidas[j]+desviaciones_alertamiento_cliente_salidas[j]))|(datos_cliente_activo_final['MontoSalidasCorta']<(medias_alertamiento_cliente_salidas[j]-desviaciones_alertamiento_cliente_salidas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento jurisdicción de ingresos']=np.where((datos_cliente_activo_final['Alertamiento jurisdicción de ingresos']!=0)|(((datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']>=i)&(datos_cliente_activo_final['Nivel de riesgo JURISDICCION NACIONAL']<(i+1)))&((datos_cliente_activo_final['MontoIngresos']>(medias_alertamiento_cliente_ingresos[j]+desviaciones_alertamiento_cliente_ingresos[j]))|(datos_cliente_activo_final['MontoIngresos']<(medias_alertamiento_cliente_ingresos[j]-desviaciones_alertamiento_cliente_ingresos[j])))),1,0)
    
datos_cliente_activo_final['Alertamiento canal de entradas']=0
datos_cliente_activo_final['Alertamiento canal de salidas']=0
datos_cliente_activo_final['Alertamiento canal de ingresos']=0
for i,j in zip(range(1,numero_clusters_1),range(len(medias_alertamiento_cliente_entradas))):
    datos_cliente_activo_final['Alertamiento canal de entradas']=np.where((datos_cliente_activo_final['Alertamiento canal de entradas']!=0)|(((datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1)))&((datos_cliente_activo_final['RiesgoCanal']>(medias_alertamiento_cliente_entradas[j]+desviaciones_alertamiento_cliente_entradas[j]))|(datos_cliente_activo_final['MontoEntradasCorta']<(medias_alertamiento_cliente_entradas[j]-desviaciones_alertamiento_cliente_entradas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento canal de salidas']=np.where((datos_cliente_activo_final['Alertamiento canal de salidas']!=0)|(((datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1)))&((datos_cliente_activo_final['RiesgoCanal']>(medias_alertamiento_cliente_salidas[j]+desviaciones_alertamiento_cliente_salidas[j]))|(datos_cliente_activo_final['MontoSalidasCorta']<(medias_alertamiento_cliente_salidas[j]-desviaciones_alertamiento_cliente_salidas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento canal de ingresos']=np.where((datos_cliente_activo_final['Alertamiento canal de ingresos']!=0)|(((datos_cliente_activo_final['RiesgoCanal']>=i)&(datos_cliente_activo_final['RiesgoCanal']<(i+1)))&((datos_cliente_activo_final['RiesgoCanal']>(medias_alertamiento_cliente_ingresos[j]+desviaciones_alertamiento_cliente_ingresos[j]))|(datos_cliente_activo_final['MontoIngresos']<(medias_alertamiento_cliente_ingresos[j]-desviaciones_alertamiento_cliente_ingresos[j])))),1,0)
    
datos_cliente_activo_final['Alertamiento producto de entradas']=0
datos_cliente_activo_final['Alertamiento producto de salidas']=0
datos_cliente_activo_final['Alertamiento producto de ingresos']=0
for i,j in zip(range(1,numero_clusters_1),range(len(medias_alertamiento_cliente_entradas))):
    datos_cliente_activo_final['Alertamiento producto de entradas']=np.where((datos_cliente_activo_final['Alertamiento producto de entradas']!=0)|(((datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1)))&((datos_cliente_activo_final['MontoEntradasCorta']>(medias_alertamiento_cliente_entradas[j]+desviaciones_alertamiento_cliente_entradas[j]))|(datos_cliente_activo_final['MontoEntradasCorta']<(medias_alertamiento_cliente_entradas[j]-desviaciones_alertamiento_cliente_entradas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento producto de salidas']=np.where((datos_cliente_activo_final['Alertamiento producto de salidas']!=0)|(((datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1)))&((datos_cliente_activo_final['MontoSalidasCorta']>(medias_alertamiento_cliente_salidas[j]+desviaciones_alertamiento_cliente_salidas[j]))|(datos_cliente_activo_final['MontoSalidasCorta']<(medias_alertamiento_cliente_salidas[j]-desviaciones_alertamiento_cliente_salidas[j])))),1,0)
    datos_cliente_activo_final['Alertamiento producto de ingresos']=np.where((datos_cliente_activo_final['Alertamiento producto de ingresos']!=0)|(((datos_cliente_activo_final['Riesgo inherente compuesto por producto']>=i)&(datos_cliente_activo_final['Riesgo inherente compuesto por producto']<(i+1)))&((datos_cliente_activo_final['MontoIngresos']>(medias_alertamiento_cliente_ingresos[j]+desviaciones_alertamiento_cliente_ingresos[j]))|(datos_cliente_activo_final['MontoIngresos']<(medias_alertamiento_cliente_ingresos[j]-desviaciones_alertamiento_cliente_ingresos[j])))),1,0)


# #### Alertas consolidadas por tipo

datos_cliente_activo_final['Puntaje de alertamiento entradas consolidado']=datos_cliente_activo_final['Alertamiento cliente de entradas']+datos_cliente_activo_final['Alertamiento jurisdicción de entradas']+datos_cliente_activo_final['Alertamiento canal de entradas']+datos_cliente_activo_final['Alertamiento producto de entradas']
datos_cliente_activo_final['Puntaje de alertamiento salidas consolidado']=datos_cliente_activo_final['Alertamiento cliente de salidas']+datos_cliente_activo_final['Alertamiento jurisdicción de salidas']+datos_cliente_activo_final['Alertamiento canal de salidas']+datos_cliente_activo_final['Alertamiento producto de salidas']
datos_cliente_activo_final['Puntaje de alertamiento ingresos consolidado']=datos_cliente_activo_final['Alertamiento cliente de ingresos']+datos_cliente_activo_final['Alertamiento jurisdicción de ingresos']+datos_cliente_activo_final['Alertamiento canal de ingresos']+datos_cliente_activo_final['Alertamiento producto de ingresos']


# #### Alertas totales

datos_cliente_activo_final['Puntaje de alertamiento total']=(datos_cliente_activo_final['Puntaje de alertamiento entradas consolidado']+datos_cliente_activo_final['Puntaje de alertamiento salidas consolidado']+datos_cliente_activo_final['Puntaje de alertamiento ingresos consolidado'])/3


datos_cliente_activo_final['Correo'] = datos_cliente_activo_final['Correo'].str.replace(r'[\x00-\x1F\x7F]', '', regex=True)

# Lista de columnas en las que se desea reemplazar los valores faltantes con 0
columnas_a_reemplazar = [
    'AlertaPerfiltxahorrosalidas',
    'Activo',
    'AlertaPerfiltxtdc',
    'AlertaVCRec',
    'AlertaDSrest',
    'DOCUMENTACION FALSA EN RADICACION',
    # 'DOCUMENTACION FALSA',
    'DOCUMENTACION FALSA INCOCREDITO',
    'EMPRESA FACHADA',
    'PREVENCION',
    'SOSPECHA POR INCUMPLIMIENTO EN PAGOS',
    'SUPLANTACION',
    'ValidacionFraudes'
]

# Itera a través de las columnas y reemplaza los valores faltantes por 0
for columna in columnas_a_reemplazar:
    consolidado_pj[columna] = consolidado_pj[columna].fillna(0).astype(int)




# %% [31] GENERACIÓN DE REPORTE INCREMENTAL DE CADA SEGMENTACIÓN SQL SERVER


# Supongamos que tienes el diccionario 'reporte'
reporte = {
    'Clientes': consolidado_pj.shape[0],
    'Activos': sum(consolidado_pj['Activo']),
    'Ingreso_Promedio': np.mean(consolidado_pj['MontoIngresos']),
    'Riesgo_inherente_minimo': min(consolidado_pj['Riesgo inherente del cliente']),
    'Riesgo_inherente_promedio': np.mean(consolidado_pj['Riesgo inherente del cliente']),
    'Riesgo_inherente_maximo': max(consolidado_pj['Riesgo inherente del cliente']),
    'Departamento_mas_respresentativo': f"{consolidado_pj['Departamento'].mode().iloc[0]} {consolidado_pj['Departamento'].value_counts().max() / len(consolidado_pj) * 100:.0f}%",
    'Validacion_de_Fraude': sum(consolidado_pj['ValidacionFraudes']),
    'Riesgo_Entidad': sum(consolidado_pj['Riesgo inherente compuesto por producto'])/consolidado_pj.shape[0],
    'pep': sum(consolidado_pj['pep']),
    'extranjeros': sum(consolidado_pj['extranjero']),
    'Incomercio': sum(consolidado_pj['Incomercio']),
    'Coeficiente_silueta_cluster_1': silhouette_score_kmeans_1,
    'Coeficiente_silueta_cluster_2': silhouette_score_kmeans_2,
    'Coeficiente_silueta_cluster_3': silhouette_score_kmeans_3,
    'Coeficiente_Davis_cluster_1': dunn_score_kmeans_1,
    'Coeficiente_Davis_cluster_2': dunn_score_kmeans_1,
    'Coeficiente_Davis_cluster_3': dunn_score_kmeans_1,
    'Ejecucion': datetime.now().strftime("%Y-%m-%d")
}



#%% Generación reporte de alertamientos






reporte_alertas = {
    'Alertamientos_canal': consolidado_pj[(consolidado_pj['RiesgoCanal']==3) | (consolidado_pj['RiesgoCanal']==4)].shape[0],
    'Alertamientos_jurisdiccion':  consolidado_pj[(consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL']==3) | (consolidado_pj['Nivel de riesgo JURISDICCION NACIONAL']==4)].shape[0],
    'Alertamientos_actividad_economica':  consolidado_pj[(consolidado_pj['PonderacionAtributosAE']==3) | (consolidado_pj['PonderacionAtributosAE']==4)].shape[0],
    'Alertamientos_producto':  consolidado_pj[(consolidado_pj['Riesgo inherente compuesto por producto']==3) | (consolidado_pj['Riesgo inherente compuesto por producto']==4)].shape[0],
    'Alertamientos_perfil_transaccional':  consolidado_pj[consolidado_pj['AlertaPerfiltxahorrosalidas'] == 1].shape[0],
    'Alertamientos_cambio_cudrante':  consolidado_pj[consolidado_pj['Alertamiento_cambio_cuadrante'] == 1].shape[0],
    

    
    'Ejecucion': datetime.now().strftime("%Y-%m-%d")
}




# Convertir el diccionario 'reporte' en un DataFrame
reporte = pd.DataFrame([reporte])
reporte_alertas = pd.DataFrame([reporte_alertas])





##### acaaaaaaa

if segmentacion == "PJ":
    '''
    #####################################################################################################
                                          PERSONA JURÍDICA   
    #####################################################################################################
    '''
    consolidado_pj.to_sql('BasePersonajuridica_SarlaftRI',con=config_db[1],if_exists='replace',index=False,schema='dbo')
    
    
    reporte.to_sql('ReportePersonaJuridicaSarlaft',con=config_db[1],if_exists='append',index=False,schema='dbo')
    reporte_alertas.to_sql('ReporteAlertasPersonaJuridicaSarlaft',con=config_db[1],if_exists='append',index=False,schema='dbo')
    
    
    config_db_fraude=config_db_trusted(con_trusted='yes',
                        con_driver='ODBC Driver 17 for SQL Server',
                        con_server='FABOGRIESGO\RIESGODB',
                        con_databaseName='ModelosSarlaft')
    
    
    consolidado_pj.to_sql((str(df[(df.nummonth ==str(dt.datetime.now().month))].iloc[0, 6])+str("_transformado")).replace("dbo.", ""),
                                                con=config_db_fraude[1],
                                                if_exists='replace',
                                                index=False,
                                                schema='dbo')
    
    
    
    excel = consolidado_pj.drop(columns=['NombreCliente', 'DireccionActual', 'Correo', 'ActividadEconomica', 'Grupo general actividad economica', 'DescripcionSucursal'])
    
    
    
    # with pd.ExcelWriter(str(df[(df.nummonth ==str( dt.datetime.now().month))].iloc[0, 9])+str('/Juridica.xlsx')) as escritor:
    #     excel.to_excel(escritor,sheet_name='Base',index=False)
    #     estadistica_grupo_final.to_excel(escritor,sheet_name='EstadisticaFinal',index=False)
    #     estadistica_grupos_individuales.to_excel(escritor,sheet_name='EstadisticaGrupo',index=False)
    #     datos_metricas_cluster_individual.to_excel(escritor,sheet_name='MetricaGrupo',index=False)
    
else:
    '''
    #####################################################################################################
                                          PERSONA NATURAL   
    #####################################################################################################
    '''
    consolidado_pj.to_sql('BasePersonanatural_SarlaftRI',
                                      con=config_db[1],
                                      if_exists='replace',
                                      index=False,
                                      schema='dbo')
    
    
    reporte.to_sql('ReportePersonaNaturalSarlaft',
                                      con=config_db[1],
                                      if_exists='append',
                                      index=False,
                                      schema='dbo')
    
    reporte_alertas.to_sql('ReporteAlertasPersonaNaturalSarlaft',con=config_db[1],if_exists='append',index=False,schema='dbo')
    
    config_db_fraude=config_db_trusted(con_trusted='yes',
                        con_driver='ODBC Driver 17 for SQL Server',
                        con_server='FABOGRIESGO\RIESGODB',
                        con_databaseName='ModelosSarlaft')
    
    
    
    # config_db_fraude=config_db_trusted(con_trusted='yes',
    #                     con_driver='ODBC Driver 17 for SQL Server',
    #                     con_server='FABOGRIESGO\RIESGODB',
    #                     con_databaseName='Productos y Transaccionalidad')
    
    consolidado_pj.to_sql((str(df[(df.nummonth ==str(dt.datetime.now().month))].iloc[0, 7])+str("_transformado")).replace("dbo.", ""),
                                                con=config_db_fraude[1],
                                                if_exists='replace',
                                                index=False,
                                                schema='dbo')
    
    
    excel = consolidado_pj.drop(columns=['NombreCliente', 'DireccionActual', 'Correo', 'ActividadEconomica', 'Grupo general actividad economica', 'DescripcionSucursal'])
    
    
    
    # with pd.ExcelWriter(str(df[(df.nummonth ==str( dt.datetime.now().month))].iloc[0, 9])+str('/Natural.xlsx')) as escritor:
    #     excel.to_excel(escritor,sheet_name='Base',index=False)
    #     estadistica_grupo_final.to_excel(escritor,sheet_name='EstadisticaFinal',index=False)
    #     estadistica_grupos_individuales.to_excel(escritor,sheet_name='EstadisticaGrupo',index=False)
    #     datos_metricas_cluster_individual.to_excel(escritor,sheet_name='MetricaGrupo',index=False)  




    #-------------------------------------------------------------------------------------------------#
    #                                     ENVÍO DE RESULTADOS                                         #
    #-------------------------------------------------------------------------------------------------#
    
    # %% [32] ENTREGA DE RESULTADOS
    
    
    # skipped your comments for readability
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    
    me = "jose.gomezv@bancofinandina.com"
    my_password = r"{}".format(str(lineas[2].strip()))
    
    # you=["jose.gomezv@bancofinandina.com"]
    
    you=["jose.gomezv@bancofinandina.com" ,"unidaddecumplimiento@bancofinandina.com","fernando.segura@bancofinandina.com","olga.garzon@bancofinandina.com"]
    # you=["jose.gomezv@bancofinandina.com" ,"unidaddecumplimiento@bancofinandina.com"]
    # you=["jose.gomezv@bancofinandina.com"]
    
    # you=["jose.gomezv@bancofinandina.com"]
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "RESULTADOS SEGMENTACIÓN SARLAFT OPTIMIZADA"
    msg['From'] = me
    msg['To'] = ",".join(you)
    
    html = """\
    <html>
      <head></head>
      <body>
        <p> 📊 📈 Buen Día, el presente correo presenta los detalles de la ejecución para la segmentación de SARLAFT Natural - Jurídica 📊 📈<br>
    """" ✅ RESULTADOS OBTENIDOS : 🕓 " +str(fecha_actual)+ """<br>
    """" ✅ Los resultados del proceso de segmentación Sarlaft fue realizada con éxito :" """<br>
    """" ✅ SEGMENTACIÓN PERSONA NATURAL : 🆙 "  """<br>
    """" ✅ REPORTE PERSONA NATURAL 🆙 " """<br>
    """" ✅ REPORTE PERSONA NATURAL ALERTAMIENTOS 🆙 " """<br>
    """" ✅ SEGMENTACIÓN PERSONA JURIDICA 🆙 " """<br>
    """" ✅ REPORTE PERSONA JURIDICA 🆙 " """<br>
    """" ✅ REPORTE PERSONA JURIDICA ALERTAMIENTOS 🆙 " """<br>
    """" ✅ TABLERO DE CONTROL ACTUALIZADO 🆙 " """<br>
        </p>
      </body>
    </html>
    """
    part2 = MIMEText(html, 'html')
    
    msg.attach(part2)
    
    # Send the message via gmail's regular server, over SSL - passwords are being sent, afterall
    s = smtplib.SMTP_SSL('smtp.gmail.com')
    # uncomment if interested in the actual smtp conversation
    # s.set_debuglevel(1)
    # do the smtp auth; sends ehlo if it hasn't been sent already
    s.login(me, my_password)
    
    s.sendmail(me,you, msg.as_string())
    s.quit()





