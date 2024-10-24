# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:10:25 2023

@author: josgom
"""

import base64



## cargar base de datos 


import numpy as np
import connectorx as cx
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
pd.options.mode.chained_assignment=None

# Formulas para la ejecución

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


# Extracción de datos
# Inicialización de logger
logger=logger_config('Alerta Fraude')


# Configuración base de datos de extracción

config_db_riesgo=config_db_trusted(con_trusted='yes',
                    con_driver='ODBC Driver 17 for SQL Server',
                    con_server='FABOGRIESGO\RIESGODB',
                    con_databaseName='InsumosAS400')



config_db_fraude=config_db_trusted(con_trusted='yes',
                    con_driver='ODBC Driver 17 for SQL Server',
                    con_server='FABOGRIESGO\RIESGODB',
                    con_databaseName='AlertasFraude')

config_db_alerta=config_db_trusted(con_trusted='yes',
                                   con_driver='ODBC Driver 17 for SQL Server',
                                   con_server='FABOGRIESGO\RIESGODB',
                                   con_databaseName='AlertasFraude')


# Carga de datos demográficos consolidados  # 1

query_datos_demograficos='''SELECT
*
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[AlertasFraude].[dbo].[visaborrar]')
visa=load_data(query_datos_demograficos,config_db_riesgo[1],model_logger=logger)


def encriptar_numero_telefono(numero_telefono, clave):
    mensaje_codificado = numero_telefono.encode('utf-8')
    clave_codificada = clave.encode('utf-8')
    longitud_clave = len(clave_codificada)
    mensaje_encriptado = bytearray()
    for i in range(len(mensaje_codificado)):
        valor = (mensaje_codificado[i] + clave_codificada[i % longitud_clave]) % 256
        mensaje_encriptado.append(valor)
    mensaje_encriptado_base64 = base64.b64encode(mensaje_encriptado).decode('utf-8')
    return mensaje_encriptado_base64

def desencriptar_numero_telefono(numero_telefono_encriptado, clave):
    mensaje_encriptado = base64.b64decode(numero_telefono_encriptado)
    clave_codificada = clave.encode('utf-8')
    longitud_clave = len(clave_codificada)
    mensaje_decodificado = bytearray()
    for i in range(len(mensaje_encriptado)):
        valor = (mensaje_encriptado[i] - clave_codificada[i % longitud_clave]) % 256
        mensaje_decodificado.append(valor)
    numero_telefono_decodificado = mensaje_decodificado.decode('utf-8')
    return numero_telefono_decodificado


clave = 'CDD2023*#'
visa['HPANENCRIPTADO'] = visa['HPAN'].apply(lambda x: encriptar_numero_telefono(x, clave))
visa['HPANDESENCRIPTADO'] = visa['HPANENCRIPTADO'].apply(lambda x: desencriptar_numero_telefono(x, clave))


# visa.to_sql('CONSULTAVISA',
#                                         con=config_db_fraude[1],
#                                         if_exists='replace',
#                                         index=False,
#                                         schema='dbo')



visa = visa.drop(columns=['HPAN', 'HPANDESENCRIPTADO'])
ruta = r'C:\Users\josgom\Desktop\BasesVisa\consultavisaCDfacturacionjunioEE.csv'



# ruta = r'\\192.168.60.149\desarrollo y ciencia de datos\RIESGO\facturacionvisaCD.csv'

visa.to_csv(ruta, sep='¬', index=False)

aaa= visa.tail()

# visa.to_sql('CONSULTAVISA',
#                                         con=config_db_fraude[1],
#                                         if_exists='replace',
#                                         index=False,
#                                         schema='dbo')








import pandas as pd

ruta_archivo = r'C:\Users\josgom\Desktop\BasesVisa\consultavisaCDfacturacionjunioEE.csv'

df = pd.read_csv(ruta_archivo, delimiter='¬', encoding='utf-8')

aaaa= df.head(100)

import pandas as pd

ruta_archivo = r'rutadelarchivo'

df = pd.read_csv(ruta_archivo, delimiter='¬', encoding='utf-8')

df.head(100)









### encriptacion debito 


ruta_inicial = r'C:\Users\josgom\Desktop\BasesVisa\basedebito.xlsx'

ruta_con_dobles_barras = ruta_inicial.replace('\\', '\\\\')
ruta_con_dobles_barras = ruta_con_dobles_barras.replace("\\", "/")

dfdebito = pd.read_excel(ruta_con_dobles_barras)


dfdebito['NumeroTarjeta']=dfdebito['NumeroTarjeta'].astype(str).str.strip()

dfdebito['HPANENCRIPTADO'] = dfdebito['NumeroTarjeta'].apply(lambda x: encriptar_numero_telefono(x, clave))

dfdebito = dfdebito.drop(columns=['NumeroTarjeta'])

dfdebito.columns
