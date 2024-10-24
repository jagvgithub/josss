''' # Created on Tue Nov 21 14:56:00 2023 @author: josgom Ciencia de datos # '''

'''# ALERTAMIENTOS DE PREGUNTAS DE NEGOCIO  #'''



## temporizador 
import time 
startTime0 = time.time()


# %% LIBRERIAS NECESARIAS

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
import connectorx as cx
pd.options.mode.chained_assignment=None


#%% CREDENCIALES


# Ruta del archivo txt
ruta_archivo = 'C:/Users/josgom/Desktop/Credenciales.txt'

# 1. Abrir el archivo txt en modo lectura
with open(ruta_archivo, 'r') as archivo:
    # 2. Leer el contenido del archivo
    lineas = archivo.readlines()
# 3. Procesar los datos (opcional)


#%% CLIENTES VIP

# exlusión de clientes 

ruta_inicial_vip = r'C:\Users\josgom\Desktop\NOBORRAR\vip.xlsx'
ruta_con_dobles_barras_vip = ruta_inicial_vip.replace('\\', '\\\\')
ruta_con_dobles_barras_vip = ruta_con_dobles_barras_vip.replace("\\", "/")
vip = pd.read_excel(str(ruta_con_dobles_barras_vip))
# vip.to_sql(name='vip', con=engine_r, if_exists='replace', index=False, schema='dbo')
vip = vip['DocumentoCliente'].tolist()


# % conexion nuevo servidor de fraude



# import pyodbc
# import pandas as pd
# # Establecer los detalles de la conexión
# server = '192.168.118.73'
# database = 'PreguntasNegocio'
# username = 'josgom'
# password = 'Fraude0505'
# # Cadena de conexión
# conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
# # Establecer la conexión
# conn = pyodbc.connect(conn_str)
# # Consulta SQL para cargar la base de datos
# query = '''SELECT * FROM [PreguntasNegocio].[dbo].[Personas]'''
# # Leer los resultados de la consulta en un DataFrame
# df = pd.read_sql_query(query, conn)
# # Cerrar la conexión
# conn.close()
# # Mostrar el DataFrame
# print(df)

# Establecer los detalles de la conexión
server_r = '192.168.118.73'
database_r = 'PreguntasNegocio'
username_r = 'AlerFaud'
password_r = 'ikBdbged2xy7iG0WcEix'
driver_r = 'ODBC Driver 17 for SQL Server'

# Cadena de conexión compatible con SQLAlchemy
conn_str_r = f'mssql+pyodbc://{username_r}:{password_r}@{server_r}/{database_r}?driver={driver_r}'

# Crear un motor SQLAlchemy
engine_r = create_engine(conn_str_r)

# Guardar el DataFrame en la tabla AI0001
# table_name = 'AI0001'
# df.to_sql(name='AI0001', con=engine_r, if_exists='replace', index=False, schema='dbo')


# %% BASE DE RECONOCER


SQL_SERVER_RIESGO= "fabogriesgo:49505"
SQL_DB_RIESGO = "AlertasFraude"
sql_connection = f"mssql://{SQL_SERVER_RIESGO}/{SQL_DB_RIESGO}?trusted_connection=true"


reconocer='''Select NumeroId as Documento,Celular1 as TelefonoReconocer,Email1 as EmailReconocer,cierre as CierreReconocer
,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from fabogcubox.[Finandina_cartera].dbo.[Reconocerhistorico]'''

reconocer = (cx.read_sql(conn = sql_connection, query = reconocer, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])

reconocer['Documento'] = reconocer['Documento'].astype('int64').astype(str).str.strip()
reconocer = reconocer.sort_values('CierreReconocer', ascending=False)
reconocer = reconocer.groupby('Documento')
reconocer = reconocer.first().reset_index()


#%% CONEXIONES NECESARIAS

fecha_actual=datetime.now().date()

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


#%% FUNCIONES NECESARIAS



def agregar_cero(valor):
        if len(valor) == 5:
            return '0' + valor
        else:
            return valor

def agregar_20(valor):
    if len(valor) == 6:
        return valor[:4] + '20' + valor[4:]
    else:
        return valor

def convertir_a_fecha(cadena):
    return pd.to_datetime(cadena, format='%d%m%Y').strftime('%Y/%m/%d')



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



#%% ALERTAS DE INFORMACIÓN

#%%% AI0001

'''# Cuentas flexi que presentan coincidencia en celular o correo o dirección  #'''

cuentas_flexi = '''Select CUNA1 as Nombre, DMACCT as NumeroCuenta,DMDOPN as FechaApertura,DMSTAT as Estado,DMCBAL as saldoactual,DMYBAL as saldodiaanterior,CUSSNR as Documento,DMTYPE as TipoProducto, CUEMA1 as Correo, CUCLPH as Celular,CUNA2 as Direccion
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
                    on Enlace.CUX1CS=Cliente.CUNBR
                    where ProductoAhorro.DMTYPE in (34,23)')
                '''

# where ProductoAhorro.DMSTAT = 1
cuentas_flexi = cargue_openquery(conn, cuentas_flexi)

cuentas_flexi['NumeroCuenta'] = cuentas_flexi['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()
cuentas_flexi['Documento'] = cuentas_flexi['Documento'].fillna(0).astype('int64').astype(str).str.strip()
cuentas_flexi['Celular'] = cuentas_flexi['Celular'].fillna(0).astype('int64').astype(str).str.strip()
cuentas_flexi['FechaApertura'] = cuentas_flexi['FechaApertura'].astype(int).astype(str).apply(agregar_cero)
cuentas_flexi['FechaApertura'] = cuentas_flexi['FechaApertura'].apply(agregar_20)
cuentas_flexi['FechaApertura'] = cuentas_flexi['FechaApertura'].apply(convertir_a_fecha)
cuentas_flexi['FechaApertura'] = pd.to_datetime(cuentas_flexi['FechaApertura'])

cuentas_flexi = cuentas_flexi.groupby('Documento').last().reset_index()

countscel = cuentas_flexi['Celular'].value_counts()
cuentas_flexi['ocurrenciascelular'] = cuentas_flexi['Celular'].map(countscel)
countsemail = cuentas_flexi['Correo'].value_counts()
cuentas_flexi['ocurrenciascorreo'] = cuentas_flexi['Correo'].map(countsemail)
countsdir = cuentas_flexi['Direccion'].value_counts()
cuentas_flexi['ocurrenciasdireccion'] = cuentas_flexi['Direccion'].map(countsdir)
mapeo = {'1': 'Activa','2':'Depurada','3':'Inactiva','4':'Cancelada','5':'Inactiva_(Auditoría)','6':'Activa_abierta_hoy'}
# Aplica el mapeo utilizando la función map() para crear la nueva columna 'Valoconsolidado'
cuentas_flexi['Estado_Descripcion'] = cuentas_flexi['Estado'].map(mapeo)


# jose =cuentas_flexi[cuentas_flexi['Documento']=='1020738171']


AI0001= pd.merge(cuentas_flexi[cuentas_flexi['TipoProducto']==34],reconocer,on=['Documento'],how='left')
AI0001 = AI0001[~AI0001['Documento'].isin(vip)]
AI0001['Fecha_ejecucion'] = fecha_actual
AI0001 = AI0001.drop_duplicates()
AI0001 = AI0001[AI0001['Documento']!='0']
AI0001 = AI0001[(AI0001['ocurrenciascelular']>2) | (AI0001['ocurrenciascorreo']>2) | (AI0001['ocurrenciasdireccion']>2)]
AI0001 = AI0001.groupby('Documento').first().reset_index()

# AI0001.to_sql('AI0001',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')


AI0001.to_sql(name='AI0001', con=engine_r, if_exists='append', index=False, schema='dbo')

print('1. Alertamiento AI0001 actualizado con éxito')



#%%% AI0002


''' # Clientes con producto TDC que tienen condición de titular y amparado   # '''

producto_tdc = '''select NUMDOC as Documento,CUENTA as Cuenta,PAN,PANANT,FECALTA,CALPART as Condicion from openquery(DB2400_182,' select base.NUMDOC
,ContratoTDC.Cuenta
,TarjetaTDC.PAN
,TarjetaTDC.PANANT
,ContratoTDC.FECALTA
,ContratoTDC.CALPART
from INTTARCRE.SATBENEFI as ContratoTDC 
join INTTARCRE.SATTARJET as TarjetaTDC
on ContratoTDC.CUENTA   =TarjetaTDC.CUENTA and
ContratoTDC.NUMBENCTA=TarjetaTDC.NUMBENCTA
left join INTTARCRE.SATCTATAR as TarjetaClienteTDC
on ContratoTDC.CUENTA=TarjetaClienteTDC.CUENTA
left join INTTARCRE.SATDACOPE as base
on ContratoTDC.IDENTCLI=base.IDENTCLI
where ContratoTDC.FECBAJA =''0001-01-01'' 
AND TarjetaTDC.FECBAJA =''0001-01-01'' 
AND TarjetaClienteTDC.INDBLQOPE =''N'' 
AND TarjetaTDC.INDSITTAR = 5' )'''

# where ProductoAhorro.DMSTAT = 1

producto_tdc = cargue_openquery(conn, producto_tdc)
producto_tdc['FECALTA'] = pd.to_datetime(producto_tdc['FECALTA'])
producto_tdc['PANANT'] = producto_tdc['PANANT'].str.strip()



AI0002 = producto_tdc
AI0002['Documento'] = AI0002['Documento'].str.strip()
AI0002['doble_titularidad'] = AI0002.groupby('Documento')['Condicion'].transform('nunique')
AI0002['validacion'] = np.where((AI0002['doble_titularidad']==2)&(AI0002['PANANT'].str.len()>0),'error','bueno')
errores=AI0002[AI0002['validacion']!='bueno']
AI0002=AI0002[AI0002['validacion']=='bueno']
AI0002=AI0002[~AI0002['Documento'].isin(errores['Documento'])]
AI0002['doble_titularidad'] = AI0002.groupby('Documento')['Condicion'].transform('nunique')
AI0002['tarjetas'] = AI0002.groupby('Documento')['PAN'].transform('nunique')
AI0002=AI0002[(AI0002['doble_titularidad']>1)]
AI0002= pd.merge(AI0002,reconocer,on=['Documento'],how='left')
AI0002 = AI0002[~AI0002['Documento'].isin(vip)]
AI0002['Fecha_ejecucion'] = fecha_actual
AI0002 = AI0002.drop_duplicates()
AI0002 = AI0002[AI0002['Documento']!='0']
AI0002=AI0002.drop(columns='validacion')


# AI0002.to_sql('AI0002',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AI0002.to_sql(name='AI0002', con=engine_r, if_exists='append', index=False, schema='dbo')

print('2. Alertamiento AI0002 actualizado con éxito')

# %%% AI0003

''' # Clientes con apertura de productos en los últimos 30 días y realizan actualizacion de datos, celular o correo dentro de las siguientes 24 horas a la apertura #'''

datos_mensajeria_limpio='''SELECT 
CAST(Documento AS VARCHAR(255)) AS Documento
,Nombre
,Fecha as 'Fecha de evento'
,Mensaje
,Estrategia as 'Evento'
,IdEstado as 'Estado'
FROM {tabla_datos}
 where
(Estrategia='ActualizacionDatos' OR
                           Estrategia='SMS418' OR
                           Estrategia='SMS419') AND
                           IdTipoMensaje=2 AND
 Fecha >= CONVERT(date, DATEADD(day, -30, GETDATE()))
 '''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_limpio=load_data(datos_mensajeria_limpio,config_db_riesgo[1],model_logger=logger)


# productos aperturados 

productos_aperturados_activo='''Select Documento
	  ,NumeroCuenta
	  ,convert(DATE,dateadd (day,right(fechaapertura,3)-1,left(fechaapertura,4)+'0101'),112) as FechaApertura
	  ,Producto
	  ,Estado
from openquery (DB2400_182,'SELECT cussnr as Documento
								  ,lnnote as NumeroCuenta
								  ,lnntdtj as fechaapertura
								  ,cflnme as Producto
								  ,CUX1AP as Estado
							FROM bnkprd01.lnp003 a
							INNER JOIN (SELECT CUX1CS,CUX1AC, CUX1AP
										FROM bnkprd01.cup009
										WHERE CUX1AP in (50,51)
										AND CUXREL IN (''SOW'',''JAF'',''JOF'')) b ON a.lnnote = b.CUX1AC
							INNER JOIN (SELECT CUNBR,CUNA1,CUSSNR,CUEMA1,CUEMA2,CUCLPH
										FROM bnkprd01.cup003) c ON c.CUNBR = b.CUX1CS
							INNER JOIN bnkprd01.cfp503 d ON a.lntype = d.cftyp ')'''
productos_aperturados_activo =  cargue_openquery(conn, productos_aperturados_activo)

productos_aperturados_activo['NumeroCuenta'] = productos_aperturados_activo['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_activo['Documento'] = productos_aperturados_activo['Documento'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_activo['FechaApertura'] = pd.to_datetime(productos_aperturados_activo['FechaApertura'])
productos_aperturados_activo = productos_aperturados_activo[productos_aperturados_activo['FechaApertura']> datetime.now() - timedelta(days=30)]



## ahorro 

productos_aperturados_ahorro_mes = '''SELECT DMACCT as NumeroCuenta,DMDOPN as FechaApertura,DMSTAT as Estado,CUSSNR as Documento,DMTYPE as Producto
                    from openquery (DB2400_182,'select
                    ProductoAhorro.DMDOPN,
                    ProductoAhorro.DMACCT,
                    ProductoAhorro.DMTYPE,
                    ProductoAhorro.DMSTAT,
                    Cliente.CUSSNR
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
                    on Enlace.CUX1CS=Cliente.CUNBR ')'''

# where ProductoAhorro.DMSTAT = 1

productos_aperturados_ahorro_mes = cargue_openquery(conn, productos_aperturados_ahorro_mes)


productos_aperturados_ahorro_mes['NumeroCuenta'] = productos_aperturados_ahorro_mes['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_ahorro_mes['Documento'] = productos_aperturados_ahorro_mes['Documento'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_ahorro_mes['FechaApertura'] = productos_aperturados_ahorro_mes['FechaApertura'].astype(int).astype(str).apply(agregar_cero)
productos_aperturados_ahorro_mes['FechaApertura'] = productos_aperturados_ahorro_mes['FechaApertura'].apply(agregar_20)
productos_aperturados_ahorro_mes['FechaApertura'] = productos_aperturados_ahorro_mes['FechaApertura'].apply(convertir_a_fecha)
productos_aperturados_ahorro_mes['FechaApertura'] = pd.to_datetime(productos_aperturados_ahorro_mes['FechaApertura'])
productos_aperturados_ahorro_mes = productos_aperturados_ahorro_mes[productos_aperturados_ahorro_mes['FechaApertura']> datetime.now() - timedelta(days=30)]

##  consolidado de productos aperturados ultimo mes 
consolidado_productos_aperturados = pd.concat([productos_aperturados_activo,productos_aperturados_ahorro_mes],axis=0)




import pandas as pd

def tiene_actualizacion_proxima(documento, consolidado_productos_aperturados, datos_mensajeria_limpio):
    filtro = datos_mensajeria_limpio[
        (datos_mensajeria_limpio['Documento'] == documento) &
        (datos_mensajeria_limpio['Fecha de evento'] > consolidado_productos_aperturados.loc[consolidado_productos_aperturados['Documento'] == documento, 'FechaApertura'].values[0]) &
        (datos_mensajeria_limpio['Fecha de evento'] <= (consolidado_productos_aperturados.loc[consolidado_productos_aperturados['Documento'] == documento, 'FechaApertura'].values[0] + pd.Timedelta(hours=24)))
    ]
    
    # Obtener categorías únicas de la columna "Evento" y contar su frecuencia
    eventos_unicos = datos_mensajeria_limpio[datos_mensajeria_limpio['Documento'] == documento]['Evento'].value_counts()
    
    # Crear una lista de pares de categoría y cantidad
    categoria_cantidad = [f'{categoria},{cantidad}' for categoria, cantidad in eventos_unicos.items()]
    
    # Convertir la lista en una cadena separada por comas
    eventos_unicos_str = ', '.join(categoria_cantidad)
    
    consolidado_productos_aperturados.loc[consolidado_productos_aperturados['Documento'] == documento, 'eventos_unicos'] = eventos_unicos_str
    
    return not filtro.empty

# Aplica la función a cada registro del DataFrame consolidado_productos_aperturados
consolidado_productos_aperturados['tiene_actualizacion_proxima'] = (consolidado_productos_aperturados.apply(
    lambda row: tiene_actualizacion_proxima(row['Documento'], consolidado_productos_aperturados, datos_mensajeria_limpio),
    axis=1
)).astype(int)



consolidado_productos_aperturados_solo_ahorro = consolidado_productos_aperturados[consolidado_productos_aperturados['Producto'].str.len().isna()]



AI0003=consolidado_productos_aperturados_solo_ahorro[consolidado_productos_aperturados_solo_ahorro['tiene_actualizacion_proxima']>0]
AI0003= pd.merge(AI0003,reconocer,on=['Documento'],how='left')
AI0003 = AI0003[~AI0003['Documento'].isin(vip)]
AI0003['Fecha_ejecucion'] = fecha_actual
AI0003 = AI0003.drop_duplicates()
AI0003 = AI0003[AI0003['Documento']!='0']

# AI0003.to_sql('AI0003',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AI0003.to_sql(name='AI0003', con=engine_r, if_exists='replace', index=False, schema='dbo')


print('3. Alertamiento AI0003 actualizado con éxito')




#%%% AI0004

''' #  Alertar clientes que en 72 horas registre nuevo dispositivo y recupere usuario # '''

datos_mensajeria_limpio72='''SELECT 
*
FROM {tabla_datos}
 where
 Estrategia in ('RegistroEquipoApp','RecuperaUsuario','GestionClaves') AND
 Fecha >= CONVERT(date, DATEADD(day, -3, GETDATE()))
 '''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_limpio72=load_data(datos_mensajeria_limpio72,config_db_riesgo[1],model_logger=logger)


datos_mensajeria_limpio72['Documento']=datos_mensajeria_limpio72['Documento'].fillna(0).astype('int64').astype(str).str.strip()
datos_mensajeria_limpio72=datos_mensajeria_limpio72.sort_values(by=['Documento','Fecha']).reset_index(drop=True)
datos_mensajeria_limpio72['Fecha'] = pd.to_datetime(datos_mensajeria_limpio72['Fecha'].dt.strftime('%Y-%m-%d %H:%M:%S'))


registroequipo = datos_mensajeria_limpio72[datos_mensajeria_limpio72['Estrategia'] == 'RegistroEquipoApp']
registroequipo['equipos_registrados']  = registroequipo.groupby('Documento')['Mensaje'].transform('count')
registroequipo['fecha_inicial_registro_equipo'] = registroequipo.groupby('Documento')['Fecha'].transform('min')
registroequipo['fecha_final_registro_equipo'] = registroequipo.groupby('Documento')['Fecha'].transform('max')

registroequipo= registroequipo[['Documento', 'equipos_registrados','fecha_inicial_registro_equipo','fecha_final_registro_equipo']]
recuperusuario = datos_mensajeria_limpio72[datos_mensajeria_limpio72['Estrategia'] == 'RecuperaUsuario']
recuperusuario['Recuperacion_usuario']  = recuperusuario.groupby('Documento')['Mensaje'].transform('count')
recuperusuario['fecha_inicial_recuperacion_usuario'] = recuperusuario.groupby('Documento')['Fecha'].transform('min')
recuperusuario['fecha_final_recuperacion_usuario'] = recuperusuario.groupby('Documento')['Fecha'].transform('max')


gestion_claves = datos_mensajeria_limpio72[datos_mensajeria_limpio72['Estrategia'] == 'GestionClaves']
gestion_claves['gestion_claves']  = gestion_claves.groupby('Documento')['Mensaje'].transform('count')
gestion_claves['fecha_inicial_gestion_claves'] = gestion_claves.groupby('Documento')['Fecha'].transform('min')
gestion_claves['fecha_final_gestion_claves'] = gestion_claves.groupby('Documento')['Fecha'].transform('max')


recuperusuario= recuperusuario[['Documento', 'Recuperacion_usuario','fecha_inicial_recuperacion_usuario','fecha_final_recuperacion_usuario']]

alertarecuperacionusuarioRegistroequipo = pd.merge(registroequipo,recuperusuario,on='Documento',how='inner')

gestion_claves= gestion_claves[['Documento', 'gestion_claves','fecha_inicial_gestion_claves','fecha_final_gestion_claves']]

alertarecuperacionusuarioRegistroequipo = pd.merge(alertarecuperacionusuarioRegistroequipo,gestion_claves,on='Documento',how='inner')

alertarecuperacionusuarioRegistroequipo = alertarecuperacionusuarioRegistroequipo.drop_duplicates()

AI0004= pd.merge(alertarecuperacionusuarioRegistroequipo,reconocer,on=['Documento'],how='left')
AI0004 = AI0004[~AI0004['Documento'].isin(vip)]
AI0004 = AI0004.drop_duplicates()
AI0004 = AI0004[AI0004['Documento']!='0']

# AI0004.to_sql('AI0004',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AI0004.to_sql(name='AI0004', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('4. Alertamiento AI0004 actualizado con éxito')



#%%% AI0005

datos_mensajeria_90='''SELECT 
CAST(Documento AS VARCHAR(255)) AS Documento
,Nombre
,Fecha as 'Fecha de evento'
,Mensaje
,Estrategia as 'Evento'
,IdEstado as 'Estado'
FROM {tabla_datos}
where
(Estrategia='ActualizacionDatos' OR
                            Estrategia='SMS418' OR
                            Estrategia='SMS419') AND
                            IdTipoMensaje=2 AND
Fecha >= CONVERT(date, DATEADD(day, -1, GETDATE()))
'''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_90=load_data(datos_mensajeria_90,config_db_riesgo[1],model_logger=logger)



resultados = (datos_mensajeria_90.groupby(['Documento', 'Evento'])['Fecha de evento'].agg(['min', 'max']).reset_index()).rename(columns={'min': 'Fecha_minima', 'max': 'Fecha_maxima'})


import pandas as pd
from datetime import datetime

resultados['dias'] = (resultados['Fecha_maxima'] - resultados['Fecha_minima']).dt.days
resultados['Rango_tiempo']=str('[')+resultados['Fecha_minima'].astype(str)+str('   ')+resultados['Fecha_maxima'].astype(str)+str(']')+str(' ')+resultados['dias'].astype(str)+str(' ')+str('dias')


resultados = (resultados.pivot(index='Documento', columns='Evento', values='Rango_tiempo')).reset_index()

actualizaciones = (datos_mensajeria_90[['Documento','Evento']].pivot_table(index='Documento', columns='Evento', aggfunc='size', fill_value=0)).reset_index()

actualizaciones['Total_actualizaciones'] = actualizaciones[[col for col in actualizaciones.columns if col != 'Documento']].apply(pd.to_numeric, errors='coerce').sum(axis=1)


actualizaciones = pd.merge(actualizaciones, resultados,on='Documento', how='left', suffixes=('', '_tiempos'))


# datos_mensajeria_90 = datos_mensajeria_90.groupby(['Documento']).agg({
#     'Documento': 'count',  # Contar la cantidad de registros         
#     'Fecha de evento': ['min', 'max']  # Encontrar el valor mínimo y máximo de la columna Fecha
# })

# # Cambiar el nombre de las columnas resultantes
# datos_mensajeria_90.columns = ['Actualizaciones','Fecha_minima', 'Fecha_maxima']

# datos_mensajeria_90=datos_mensajeria_90.reset_index()


ultima_tx = '''SELECT MCNROCTA as NumeroCuenta,ultimatx
FROM OPENQUERY(DB2400_182,
   'SELECT MCNROCTA, MAX(MCFHRPTA) AS ultimatx
    FROM BNKPRD01.EXTMOVCAJ
    GROUP BY MCNROCTA'
)
where ultimatx <= getdate()-30'''
                

# where ProductoAhorro.DMSTAT = 1
ultima_tx = cargue_openquery(conn, ultima_tx)

ultima_tx['NumeroCuenta']=ultima_tx['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()


data = pd.merge(cuentas_flexi,ultima_tx,on='NumeroCuenta',how='inner')
data = pd.merge(data,actualizaciones,on='Documento',how='inner')

AI0005= pd.merge(data,reconocer,on=['Documento'],how='left')
AI0005 = AI0005[~AI0005['Documento'].isin(vip)]
AI0005['Fecha_ejecucion'] = fecha_actual
AI0005 = AI0005.drop_duplicates()
AI0005 = AI0005[AI0005['Documento']!='0']

# AI0005.to_sql('AI0005',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AI0005.to_sql(name='AI0005', con=engine_r, if_exists='append', index=False, schema='dbo')

print('5. Alertamiento AI0005 actualizado con éxito')



#%%%AI0006


''' # Clientes que hagan registro de dispositivo diferentes más de 2 veces en 30 dias # '''

query_mensajeriames='''SELECT 
CAST(Documento AS VARCHAR(255)) AS Documento
,Nombre
,Fecha
,Mensaje
,Estrategia as 'Evento'
,IdEstado as 'Estado'
FROM {tabla_datos}
where
Estrategia in ('RegistroEquipoApp') AND
Fecha >= CONVERT(date, DATEADD(day, -30, GETDATE()))
'''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_limpiomes=load_data(query_mensajeriames,config_db_riesgo[1],model_logger=logger)


# Filtrar el dataframe usando la función loc() de pandas y las condiciones de fecha y Mensaje
# df_filtrado = datos_mensajeria_limpiomes.loc[(datos_mensajeria_limpiomes['Mensaje'].str.contains('Banco Finandina te notifica la actualizacion de tu No. de Celular.'))]
df_filtrado=datos_mensajeria_limpiomes

df_filtrado['Registro_equipo']  = datos_mensajeria_limpiomes.groupby('Documento')['Mensaje'].transform('count')
df_filtrado['fecha_inicial_registro_equipo'] = df_filtrado.groupby('Documento')['Fecha'].transform('min')
df_filtrado['fecha_final_registro_equipo'] = df_filtrado.groupby('Documento')['Fecha'].transform('max')
df_filtrado= df_filtrado[['Documento', 'Registro_equipo','fecha_inicial_registro_equipo','fecha_final_registro_equipo']]
df_filtrado_ocurrencias =df_filtrado[df_filtrado['Registro_equipo']>=2]
df_filtrado_ocurrencias = df_filtrado_ocurrencias.drop_duplicates()

AI0006= pd.merge(df_filtrado_ocurrencias,reconocer,on=['Documento'],how='left')
AI0006 = df_filtrado_ocurrencias[~df_filtrado_ocurrencias['Documento'].isin(vip)]
AI0006['FechaAlerta']=fecha_actual
AI0006 = AI0006.drop_duplicates()
AI0006 = AI0006[AI0006['Documento']!='0']

# AI0006.to_sql('AI0006',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AI0006.to_sql(name='AI0006', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('6. Alertamiento AI0006 actualizado con éxito')

# %%% AI0007

''' # Cuentas con correo asociado a correos sospechosos con dominio protonmail, mail o con la base fraude y suplantación # '''


server_name_auditoria = 'FABOGSQL01\\AUDITORIA,52715'
database_name_auditoria = 'AUDITORIA_COMPARTIDA'
integrated_security_auditoria = 'yes'  # Para autenticación de Windows

# Configurar la cadena de conexión (como se mencionó en respuestas anteriores)

connection_string = f'DRIVER={{SQL Server}};SERVER={server_name_auditoria};DATABASE={database_name_auditoria};Integrated Security={integrated_security_auditoria}'

# Intentar establecer la conexión
try:
    conn = pyodbc.connect(connection_string)
    print('Conexión exitosa a SQL Server')
    
    # Consulta SQL que deseas ejecutar
    query = "SELECT * FROM [AUDITORIA_COMPARTIDA].[PV].[Bd_Fraudes]"
    
    # Ejecutar la consulta y almacenar los resultados en un DataFrame
    auditoria = pd.read_sql(query, conn)
    
    # Imprimir los resultados (opcional)
    # print(df)
    
    # No olvides cerrar la conexión cuando hayas terminado
    # conn.close()

except Exception as e:
    print(f'Error al conectar a SQL Server: {str(e)}')



auditoria['ID_Cliente'] = auditoria['ID_Cliente'].fillna(0).astype('int64').astype(str).str.strip()




datos_demograficos = '''select cussnr as Documento,cuema1 as Correo,cuclph as Celular from openquery(DB2400_182,'select cussnr,cuema1,cuclph from BNKPRD01.cup003 ')'''

# where ProductoAhorro.DMSTAT = 1

datos_demograficos = cargue_openquery(conn, datos_demograficos)
datos_demograficos['Correo']=datos_demograficos['Correo'].str.upper().str.strip()
datos_demograficos['Celular']=datos_demograficos['Celular'].fillna(0).astype('int64').astype(str).str.strip()
datos_demograficos['Documento']=datos_demograficos['Documento'].fillna(0).astype('int64').astype(str).str.strip()
conteo_correos = datos_demograficos['Correo'].value_counts()
datos_demograficos['ocurrencias_correo'] = datos_demograficos['Correo'].map(conteo_correos)
conteo_celular = datos_demograficos['Celular'].value_counts()
datos_demograficos['ocurrencias_celular'] = datos_demograficos['Celular'].map(conteo_celular)
patron_correo = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
# Aplicar la validación a la columna 'Correo' y crear una columna booleana 'Correo_Valido'
datos_demograficos['Correo_Valido_estructura'] = (datos_demograficos['Correo'].str.match(patron_correo)).astype(int)
datos_demograficos['Validacion_protonmail'] = np.where(datos_demograficos['Correo'].str.contains('@PROTONMAIL'), 1, 0)
datos_demograficos['Validacion_mail'] = np.where(datos_demograficos['Correo'].str.contains('@MAIL'), 1, 0)
datos_demograficos['Validacion_fraude'] = np.where(datos_demograficos['Documento'].isin(auditoria['ID_Cliente']), 1, 0)
alerta_correo_riesgo=datos_demograficos.loc[(datos_demograficos['Validacion_protonmail'] == 1) | (datos_demograficos['Validacion_mail'] == 1) | (datos_demograficos['Validacion_fraude'] == 1)]
alerta_correo_riesgo = alerta_correo_riesgo[(alerta_correo_riesgo['Correo'] != '') & (alerta_correo_riesgo['Celular'] != '0') & (alerta_correo_riesgo['Documento'] != '0')]

                                                                                    
AI0007= pd.merge(alerta_correo_riesgo,reconocer,on=['Documento'],how='left')
AI0007 = AI0007[~AI0007['Documento'].isin(vip)]
AI0007['Fecha_ejecucion'] = fecha_actual
AI0007 = AI0007.drop_duplicates()
AI0007 = AI0007[AI0007['Documento']!='0']

# AI0007.to_sql('AI0007',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AI0007.to_sql(name='AI0007', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('7. Alertamiento AI0007 actualizado con éxito')


# %%% AI0008

''' # Clientes que realizan en un mes más de dos recuperaciones de contraseña es un mes # '''


datos_mensajeria_claves_30='''SELECT 
Documento,Fecha,Estrategia
FROM {tabla_datos}
where Estrategia = 'GestionClaves'
and IdTipoMensaje = 2 AND
Fecha >= CONVERT(date, DATEADD(day, -30, GETDATE()))
'''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_claves_30=load_data(datos_mensajeria_claves_30,config_db_riesgo[1],model_logger=logger)


datos_mensajeria_claves_30 = datos_mensajeria_claves_30.groupby(['Documento']).agg({
    'Documento': 'count',  # Contar la cantidad de registros         
    'Fecha': ['min', 'max']  # Encontrar el valor mínimo y máximo de la columna Fecha
})

# Cambiar el nombre de las columnas resultantes
datos_mensajeria_claves_30.columns = ['Actualizaciones','Fecha_minima', 'Fecha_maxima']
datos_mensajeria_claves_30=datos_mensajeria_claves_30.reset_index()
datos_mensajeria_claves_30['dias'] = (datos_mensajeria_claves_30['Fecha_maxima'] - datos_mensajeria_claves_30['Fecha_minima']).dt.days
datos_mensajeria_claves_30= datos_mensajeria_claves_30[datos_mensajeria_claves_30['Actualizaciones']>2]

AI0008= pd.merge(datos_mensajeria_claves_30,reconocer,on=['Documento'],how='left')
AI0008 = AI0008[~AI0008['Documento'].isin(vip)]
AI0008['Fecha_ejecucion'] = fecha_actual
AI0008 = AI0008.drop_duplicates()
AI0008 = AI0008[AI0008['Documento']!='0']
# AI0008.to_sql('AI0008',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AI0008.to_sql(name='AI0008', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('8. Alertamiento AI0008 actualizado con éxito')

#%%% AI0009

''' # Actualizacion de correo o celular en los ultimos 30 dias y generar marca si se actualizan ambos en una ventana de tiempo de 4 horas además de validar la información con la base de fraudes #'''

datos_mensajeria_celular_correo_30='''SELECT 
Documento,Fecha,Estrategia
FROM {tabla_datos}
where Estrategia in('SMS418','SMS419')
and IdTipoMensaje = 2 AND
Fecha >= CONVERT(date, DATEADD(day, -30, GETDATE()))
'''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_celular_correo_30=load_data(datos_mensajeria_celular_correo_30,config_db_riesgo[1],model_logger=logger)

import pandas as pd
from datetime import datetime, timedelta

# Ordena el DataFrame por documento y fecha
datos_mensajeria_celular_correo_30 = datos_mensajeria_celular_correo_30.sort_values(by=['Documento', 'Fecha'])

# Inicializa una lista para almacenar las marcas de validación
marcas = []

# Define una ventana de tiempo de 4 horas
ventana_de_tiempo = timedelta(hours=4)

# Itera a través de los registros para validar las actualizaciones
for index, row in datos_mensajeria_celular_correo_30.iterrows():
    documento = row['Documento']
    fecha = row['Fecha']
    estrategia = row['Estrategia']

    # Filtra el DataFrame para registros del mismo documento y estrategia dentro de la ventana de tiempo
    ventana = datos_mensajeria_celular_correo_30[(datos_mensajeria_celular_correo_30['Documento'] == documento) & (datos_mensajeria_celular_correo_30['Estrategia'] == estrategia) & (datos_mensajeria_celular_correo_30['Fecha'] >= fecha - ventana_de_tiempo) & (datos_mensajeria_celular_correo_30['Fecha'] <= fecha)]
    
    # Si hay al menos dos registros dentro de la ventana (el registro actual y al menos uno más),
    # se marca como actualización válida; de lo contrario, no se marca.
    if len(ventana) > 1:
        marcas.append('Si')
    else:
        marcas.append('No')

# Agrega la columna "Marca" al DataFrame
datos_mensajeria_celular_correo_30['Ambos_cambios_4_horas'] = marcas



# validacion de fraudes 

datos_mensajeria_celular_correo_30['validacion_fraudes'] = datos_mensajeria_celular_correo_30['Documento'].isin(auditoria['ID_Cliente']).astype(int)

datos_mensajeria_celular_correo_30 = datos_mensajeria_celular_correo_30[(datos_mensajeria_celular_correo_30['Documento']!='0') & ((datos_mensajeria_celular_correo_30['validacion_fraudes']==1) & (datos_mensajeria_celular_correo_30['Ambos_cambios_4_horas']=='Si'))  ]
datos_mensajeria_celular_correo_30 = datos_mensajeria_celular_correo_30.drop_duplicates()

# validar que cumpla las condiciones, registros duplicados revisar bien 
AI0009= pd.merge(datos_mensajeria_celular_correo_30,reconocer,on=['Documento'],how='left')
AI0009 = AI0009[~AI0009['Documento'].isin(vip)]
AI0009['Fecha_ejecucion'] = fecha_actual
AI0009 = AI0009.drop_duplicates()
AI0009 = AI0009[AI0009['Documento']!='0']

# AI0009.to_sql('AI0009',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')
AI0009.to_sql(name='AI0009', con=engine_r, if_exists='append', index=False, schema='dbo')

print('9. Alertamiento AI0009 actualizado con éxito')

#%%% AI0010

''' # alertar más de 3 clientes aperturando un producto desde la misma IP en un período de 72 horas #  '''

datos_log_creacion_limpio = '''SELECT 
FechaRegistro as 'Fecha de registro'
,Documento as 'Documento'
,Sesion as 'Sesión'
,Ip as 'IP'
,NumeroPaso as 'Paso'
,Mensaje as 'Mensaje'
,DescripcionPaso as 'Descripción de paso'
,DescripcionTipo as 'Descripción de tipo'
,DescripcionNivel as 'Descripción de nivel'
,DescripcionError as 'Descripción de error' 
,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
FROM [FABOGRIESGO\RIESGODB].[LogTransaccional].[dbo].[RegistroAhorroCreado]
WHERE  [FechaRegistro] >=  DATEADD(dd,DATEDIFF(dd,3,GETDATE()),0) and [FechaRegistro] <=  DATEADD(dd,DATEDIFF(dd,0,GETDATE()),0)'''
datos_log_creacion_limpio = cx.read_sql(conn = sql_connection, query = datos_log_creacion_limpio, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])





datos_log_creacion_limpio['Documento']=datos_log_creacion_limpio['Documento'].astype(str).str.strip()
datos_log_creacion_limpio['Documento']=pd.to_numeric(datos_log_creacion_limpio['Documento'],errors='coerce',downcast='integer')
datos_log_creacion_limpio=datos_log_creacion_limpio.dropna(subset=['Documento'])
datos_log_creacion_limpio['Documento']=datos_log_creacion_limpio['Documento'].astype(np.int64).astype(str)
datos_log_creacion_limpio['Fecha de registro']=pd.to_datetime(datos_log_creacion_limpio['Fecha de registro'])
datos_log_creacion_limpio=datos_log_creacion_limpio.sort_values(by=['Fecha de registro']).reset_index(drop=True)


datos_registro_creacion=datos_log_creacion_limpio[['Documento','Fecha de registro','Sesión','IP','Mensaje','Descripción de tipo']]
datos_registro_creacion=datos_registro_creacion[datos_registro_creacion['IP']!='']
datos_registro_creacion_conteo = datos_registro_creacion['IP'].value_counts()
datos_registro_creacion['ocurrencias_IP'] = datos_registro_creacion['IP'].map(datos_registro_creacion_conteo)
datos_registro_creacion=datos_registro_creacion.loc[datos_registro_creacion['ocurrencias_IP']>3]


AI0010= pd.merge(datos_registro_creacion,reconocer,on=['Documento'],how='left')
AI0010 = AI0010[~AI0010['Documento'].isin(vip)]
AI0010['Fecha_ejecucion'] = fecha_actual
AI0010 = AI0010.drop_duplicates()
AI0010 = AI0010[AI0010['Documento']!='0']

# AI0010.to_sql('AI0010',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AI0010.to_sql(name='AI0010', con=engine_r, if_exists='append', index=False, schema='dbo')



print('10. Alertamiento AI0010 actualizado con éxito')


#%%% AI0011


try:
    ''' # Número de intentos fallidos de vinculación a cuentas de ahorro asociados a una misma cedula en un plazo menor a 72 horas # '''
    
    intentos_vinculacion = '''SELECT 
    NumeroIdentificacion as 'Documento'
    ,CodigoSolicitud as 'Código de solicitud'
    ,FechaInicialSolicitud as 'Fecha inicial'
    ,FechaFinalizacionSolicitud as 'Fecha final'
    ,Nombres as 'Nombre'
    ,PrimerApellido as 'Apellido'
    ,NumeroCelular as 'Celular'
    ,NumeroCuenta as 'Número de cuenta'
    ,TipoProceso as 'Estado de proceso'
    ,DescripcionProceso as 'Descripción de proceso'
    ,ClienteCreado as 'Cliente creado con éxito'
    ,CuentaCreada as 'Cuenta creada con éxito'
    ,CDTCreado as 'CDT creado con éxito'
    ,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
    FROM [FABOGRIESGO\RIESGODB].[LogTransaccional].[dbo].[LogCreacionCliente]
    WHERE FechaInicialSolicitud >= GETDATE()-3 '''
    intentos_vinculacion = cx.read_sql(conn = sql_connection, query = intentos_vinculacion, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])
    
    intentos_vinculacion['Documento']=intentos_vinculacion['Documento'].astype(str).str.strip()
    intentos_vinculacion['Documento']=pd.to_numeric(intentos_vinculacion['Documento'],errors='coerce',downcast='integer')
    intentos_vinculacion=intentos_vinculacion.dropna(subset=['Documento'])
    intentos_vinculacion['Documento']=intentos_vinculacion['Documento'].astype(np.int64).astype(str)
    intentos_vinculacion['Fecha inicial']=pd.to_datetime(intentos_vinculacion['Fecha inicial'])
    intentos_vinculacion['Fecha final']=pd.to_datetime(intentos_vinculacion['Fecha final'])
    intentos_vinculacion=intentos_vinculacion.sort_values(by=['Documento','Fecha inicial']).reset_index(drop=True)
    
    
    datos_intentos_vinculacion=intentos_vinculacion.drop_duplicates(subset=['Código de solicitud'])
    
    datos_intentos_vinculacion['Delta transaccional']=datos_intentos_vinculacion.groupby(by=['Documento'])[['Fecha inicial']].diff()
    datos_intentos_vinculacion['Intento vinculación reciente']=np.where(datos_intentos_vinculacion['Delta transaccional']<=pd.Timedelta(hours=72),True,False)
    
    alerta_intentos_vinculacion=datos_intentos_vinculacion.loc[(datos_intentos_vinculacion['Intento vinculación reciente']==True)|((datos_intentos_vinculacion['Intento vinculación reciente']==True).shift(-1))].reset_index(drop=True)
    alerta_intentos_vinculacion['Delta transaccional']=alerta_intentos_vinculacion['Delta transaccional'].astype(str)
    
    AI0011= pd.merge(alerta_intentos_vinculacion,reconocer,on=['Documento'],how='left')
    AI0011 = AI0011[~AI0011['Documento'].isin(vip)]
    AI0011['Fecha_ejecucion'] = fecha_actual
    AI0011 = AI0011.drop_duplicates()
    AI0011 = AI0011[AI0011['Documento']!='0']
    
    # AI0011.to_sql('AI0011',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')
    AI0011.to_sql(name='AI0011', con=engine_r, if_exists='replace', index=False, schema='dbo')

except Exception as e:

    AI0011 = pd.DataFrame(columns=['Documento'])


print('11. Alertamiento AI0011 actualizado con éxito')

#%%% AI0012

''' # Clientes distintos que se conectan desde el mismo dispositivo seguro en un tiempo de 72 horas # '''

datos_disposito_vinculado = '''SELECT 
*
,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
FROM [FABOGREPORTS].[AppFinandina].[Core].[SecureDevice]
WHERE [ChangeDate] >= DATEADD(dd,DATEDIFF(dd,3,GETDATE()),0) AND [ChangeDate] <  DATEADD(dd,DATEDIFF(dd,0,GETDATE()),0)'''
datos_disposito_vinculado = cx.read_sql(conn = sql_connection, query = datos_disposito_vinculado, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

datos_disposito_vinculado_limpio=datos_disposito_vinculado.rename(columns={'RegisterDate':'Fecha de registro'
                                                                          ,'Identification':'Documento'
                                                                          ,'Latitude':'Latitud'
                                                                          ,'Longitude':'Longitud'
                                                                          ,'ChangeDate':'Fecha de cambio'
                                                                          ,'identification':'Documento'
                                                                          ,'Hash':'Identificador dispositivo'})
datos_disposito_vinculado_limpio['Documento']=datos_disposito_vinculado_limpio['Documento'].astype(str).str.strip()
datos_disposito_vinculado_limpio['Documento']=pd.to_numeric(datos_disposito_vinculado_limpio['Documento'],errors='coerce',downcast='integer')
datos_disposito_vinculado_limpio=datos_disposito_vinculado_limpio.dropna(subset=['Documento'])
datos_disposito_vinculado_limpio['Documento']=datos_disposito_vinculado_limpio['Documento'].astype(np.int64).astype(str)
datos_disposito_vinculado_limpio['Fecha de registro']=pd.to_datetime(datos_disposito_vinculado_limpio['Fecha de registro'],format='%Y-%m-%d')
datos_disposito_vinculado_limpio=datos_disposito_vinculado_limpio.sort_values(by=['Fecha de registro']).reset_index(drop=True)


datos_dispositivo_vinculado_disponible=datos_disposito_vinculado_limpio.dropna(subset=['Fingerprint']).reset_index(drop=True)



datos_dispositivo_vinculado_disponible['Fingerprint'] = datos_dispositivo_vinculado_disponible['Fingerprint'].astype(str)

# Now apply json.loads()
datos_dispositivo_vinculado_disponible['Fingerprint'] = datos_dispositivo_vinculado_disponible['Fingerprint'].apply(lambda x: json.loads(x))



# datos_dispositivo_vinculado_disponible['Fingerprint']=datos_dispositivo_vinculado_disponible['Fingerprint'].apply(lambda x: json.loads(x))
datos_dispositivo_descompuesto=pd.concat([datos_dispositivo_vinculado_disponible.drop(['Fingerprint'],axis=1),pd.DataFrame(datos_dispositivo_vinculado_disponible['Fingerprint'].tolist())],axis=1)




try:
    datos_dispositivo_vinculado_especifico = datos_dispositivo_descompuesto[['Documento','IP','SO','DeviceType','Latitud','Longitud','Fecha de registro','Fecha de cambio','Identificador dispositivo','androidId','platform','brand','id','manufacturer','model','identifierForVendor','utsname.machine:']]
    datos_dispositivo_vinculado_especifico['Id único dispositivo'] = datos_dispositivo_vinculado_especifico['androidId'].fillna(datos_dispositivo_vinculado_especifico['identifierForVendor'])
    datos_dispositivo_vinculado_especifico['Marca'] = datos_dispositivo_vinculado_especifico['brand'].fillna(datos_dispositivo_vinculado_especifico['model'])
    datos_dispositivo_vinculado_especifico['Fabricante'] = datos_dispositivo_vinculado_especifico['manufacturer'].fillna('Apple')
    datos_dispositivo_vinculado_especifico['Modelo'] = datos_dispositivo_vinculado_especifico['model'].fillna(datos_dispositivo_vinculado_especifico['utsname.machine:'])
    datos_dispositivo_vinculado_especifico['Plataforma'] = datos_dispositivo_vinculado_especifico['platform'].fillna('iOS')
    datos_dispositivo_vinculado_detalle = datos_dispositivo_vinculado_especifico[['Documento','IP','SO','DeviceType','Latitud','Longitud','Fecha de registro','Fecha de cambio','Id único dispositivo','Marca','Fabricante','Modelo','Plataforma']]
    datos_cantidad_clientes_dispositivos = pd.merge(datos_dispositivo_vinculado_detalle, pd.DataFrame(datos_dispositivo_vinculado_detalle.groupby(by=['Id único dispositivo']).size(), columns=['Cantidad de usuarios asociados']).reset_index(), on=['Id único dispositivo'], how='inner')
    alerta_cantidad_dispositivos_vinculados = datos_cantidad_clientes_dispositivos.loc[datos_cantidad_clientes_dispositivos['Cantidad de usuarios asociados'] > 1]
except Exception as e:
    alerta_cantidad_dispositivos_vinculados = pd.DataFrame(columns=['Documento'])


# datos_dispositivo_vinculado_especifico=datos_dispositivo_descompuesto[['Documento','IP','SO','DeviceType','Latitud','Longitud','Fecha de registro','Fecha de cambio','Identificador dispositivo','androidId','platform','brand','id','manufacturer','model','identifierForVendor','utsname.machine:','systemVersion']]
# datos_dispositivo_vinculado_especifico['Id único dispositivo']=datos_dispositivo_vinculado_especifico['androidId'].fillna(datos_dispositivo_vinculado_especifico['identifierForVendor'])
# datos_dispositivo_vinculado_especifico['Marca']=datos_dispositivo_vinculado_especifico['brand'].fillna(datos_dispositivo_vinculado_especifico['model'])
# datos_dispositivo_vinculado_especifico['Fabricante']=datos_dispositivo_vinculado_especifico['manufacturer'].fillna('Apple')
# datos_dispositivo_vinculado_especifico['Modelo']=datos_dispositivo_vinculado_especifico['model'].fillna(datos_dispositivo_vinculado_especifico['utsname.machine:'])
# datos_dispositivo_vinculado_especifico['Plataforma']=datos_dispositivo_vinculado_especifico['platform'].fillna('iOS')
# datos_dispositivo_vinculado_detalle=datos_dispositivo_vinculado_especifico[['Documento','IP','SO','DeviceType','Latitud','Longitud','Fecha de registro','Fecha de cambio','Id único dispositivo','Marca','Fabricante','Modelo','Plataforma']]
# datos_cantidad_clientes_dispositivos=pd.merge(datos_dispositivo_vinculado_detalle,pd.DataFrame(datos_dispositivo_vinculado_detalle.groupby(by=['Id único dispositivo']).size(),columns=['Cantidad de usuarios asociados']).reset_index(),on=['Id único dispositivo'],how='inner')
# alerta_cantidad_dispositivos_vinculados=datos_cantidad_clientes_dispositivos.loc[datos_cantidad_clientes_dispositivos['Cantidad de usuarios asociados']>1]


alerta_cantidad_dispositivos_vinculados = alerta_cantidad_dispositivos_vinculados[~alerta_cantidad_dispositivos_vinculados['Documento'].isin(vip)]

AI0012= pd.merge(alerta_cantidad_dispositivos_vinculados,reconocer,on=['Documento'],how='left')
AI0012 = AI0012[~AI0012['Documento'].isin(vip)]
AI0012['Fecha_ejecucion'] = fecha_actual
AI0012 = AI0012.drop_duplicates()
AI0012 = AI0012[AI0012['Documento']!='0']

# AI0012.to_sql('AI0012',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')


AI0012.to_sql(name='AI0012', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('12. Alertamiento AI0012 actualizado con éxito')



# %%%  AI0013


''' # Cliente apertura flexidigital con mas de tres intentos de evidente en 72 horas # '''

datos_log_evidente_limpio = '''SELECT 
NumeroDocumento as 'Documento'
,Response as 'Respuesta'
,FechaRegistro as 'Fecha de registro'
,TipoConsultaEvidente as 'Tipo de consulta'
,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
FROM [FABOGSQLCLU].[Usuarios].[dbo].[HistoricoEvidente]
WHERE [FechaRegistro] >=  DATEADD(dd,DATEDIFF(dd,3,GETDATE()),0) and [FechaRegistro] <  DATEADD(dd,DATEDIFF(dd,0,GETDATE()),0) '''
datos_log_evidente_limpio = cx.read_sql(conn = sql_connection, query = datos_log_evidente_limpio, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])



datos_log_evidente_limpio['Documento']=datos_log_evidente_limpio['Documento'].astype(str).str.strip()
datos_log_evidente_limpio['Documento']=pd.to_numeric(datos_log_evidente_limpio['Documento'],errors='coerce',downcast='integer')
datos_log_evidente_limpio=datos_log_evidente_limpio.dropna(subset=['Documento'])
datos_log_evidente_limpio['Documento']=datos_log_evidente_limpio['Documento'].astype(np.int64).astype(str)
datos_log_evidente_limpio['Fecha de registro']=pd.to_datetime(datos_log_evidente_limpio['Fecha de registro'])


datos_log_evidente_limpio['Respuesta'] = datos_log_evidente_limpio['Respuesta'].apply(lambda x: {k:v for k,v in (subString.split(':') for subString in x[1:-1].split(','))} if isinstance(x, str) else {})

# datos_log_evidente_limpio['Respuesta']=datos_log_evidente_limpio['Respuesta'].apply(lambda x: dict(subString.split(':') for subString in x[1:-1].split(',')))

datos_log_evidente_limpio=datos_log_evidente_limpio.sort_values(by=['Fecha de registro']).reset_index(drop=True)

datos_evidente_descompuesto=pd.concat([datos_log_evidente_limpio.drop(['Respuesta'],axis=1),pd.DataFrame(datos_log_evidente_limpio['Respuesta'].tolist())],axis=1)
# datos_evidente_descompuesto = datos_evidente_descompuesto.rename(columns={0: 'Respuesta'})


datos_evidente_descompuesto = datos_evidente_descompuesto.groupby(['Documento']).agg({
    'Documento': 'count',  # Contar la cantidad de registros         
    'Fecha de registro': ['min', 'max']  # Encontrar el valor mínimo y máximo de la columna Fecha
})

# Cambiar el nombre de las columnas resultantes
datos_evidente_descompuesto.columns = ['Intentos_evidente','Fecha_minima', 'Fecha_maxima']
datos_evidente_descompuesto=datos_evidente_descompuesto.reset_index()
data=pd.merge(cuentas_flexi,datos_evidente_descompuesto,on='Documento',how='inner')



AI0013= pd.merge(data,reconocer,on=['Documento'],how='left')
AI0013 = AI0013[~AI0013['Documento'].isin(vip)]
AI0013['Fecha_ejecucion'] = fecha_actual
AI0013 = AI0013.drop_duplicates()
AI0013 = AI0013[AI0013['Documento']!='0']

# AI0013.to_sql('AI0013',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AI0013.to_sql(name='AI0013', con=engine_r, if_exists='append', index=False, schema='dbo')

print('13. Alertamiento AI0013 actualizado con éxito')


#%%% AI0014

''' # Clientes con productos aperturados en el ultimo mes y que en la ultima semana hayan realizado enrolamiento # '''


productos_aperturados_activo='''Select Documento
	  ,NumeroCuenta
	  ,convert(DATE,dateadd (day,right(fechaapertura,3)-1,left(fechaapertura,4)+'0101'),112) as FechaApertura
	  ,Producto
	  ,Estado
from openquery (DB2400_182,'SELECT cussnr as Documento
								  ,lnnote as NumeroCuenta
								  ,lnntdtj as fechaapertura
								 
								  ,cflnme as Producto
								  ,CUX1AP as Estado
							FROM bnkprd01.lnp003 a
							INNER JOIN (SELECT CUX1CS,CUX1AC, CUX1AP
										FROM bnkprd01.cup009
										WHERE CUX1AP in (50,51)
										AND CUXREL IN (''SOW'',''JAF'',''JOF'')) b ON a.lnnote = b.CUX1AC
							INNER JOIN (SELECT CUNBR,CUNA1,CUSSNR,CUEMA1,CUEMA2,CUCLPH
										FROM bnkprd01.cup003) c ON c.CUNBR = b.CUX1CS
							INNER JOIN bnkprd01.cfp503 d ON a.lntype = d.cftyp ')'''
productos_aperturados_activo =  cargue_openquery(conn, productos_aperturados_activo)

productos_aperturados_activo['NumeroCuenta'] = productos_aperturados_activo['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_activo['Documento'] = productos_aperturados_activo['Documento'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_activo['FechaApertura'] = pd.to_datetime(productos_aperturados_activo['FechaApertura'])
productos_aperturados_activo = productos_aperturados_activo[productos_aperturados_activo['FechaApertura']> datetime.now() - timedelta(days=30)]



## ahorro 



productos_aperturados_ahorro_mes = '''SELECT DMACCT as NumeroCuenta,DMDOPN as FechaApertura,DMSTAT as Estado,CUSSNR as Documento,DMTYPE as Producto
                    from openquery (DB2400_182,'select
                    ProductoAhorro.DMDOPN,
                    ProductoAhorro.DMACCT,
                    ProductoAhorro.DMTYPE,
                    ProductoAhorro.DMSTAT,
                    Cliente.CUSSNR
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
                    on Enlace.CUX1CS=Cliente.CUNBR ')'''

# where ProductoAhorro.DMSTAT = 1

productos_aperturados_ahorro_mes = cargue_openquery(conn, productos_aperturados_ahorro_mes)


productos_aperturados_ahorro_mes['NumeroCuenta'] = productos_aperturados_ahorro_mes['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_ahorro_mes['Documento'] = productos_aperturados_ahorro_mes['Documento'].fillna(0).astype('int64').astype(str).str.strip()
productos_aperturados_ahorro_mes['FechaApertura'] = productos_aperturados_ahorro_mes['FechaApertura'].astype(int).astype(str).apply(agregar_cero)
productos_aperturados_ahorro_mes['FechaApertura'] = productos_aperturados_ahorro_mes['FechaApertura'].apply(agregar_20)
productos_aperturados_ahorro_mes['FechaApertura'] = productos_aperturados_ahorro_mes['FechaApertura'].apply(convertir_a_fecha)
productos_aperturados_ahorro_mes['FechaApertura'] = pd.to_datetime(productos_aperturados_ahorro_mes['FechaApertura'])
productos_aperturados_ahorro_mes = productos_aperturados_ahorro_mes[productos_aperturados_ahorro_mes['FechaApertura']> datetime.now() - timedelta(days=30)]

##  consolidado de productos aperturados ultimo mes 
consolidado_productos_aperturados = pd.concat([productos_aperturados_activo,productos_aperturados_ahorro_mes],axis=0)


import pandas as pd

# Supongamos que ya tienes un DataFrame llamado df con columnas 'Documento' y 'Fecha'

# Encuentra la fecha máxima para cada documento
ultimo_producto_aperturado = consolidado_productos_aperturados.groupby('Documento')['FechaApertura'].max().reset_index()

# Combina el DataFrame original con la fecha máxima por documento
ultimo_producto_aperturado = pd.merge(consolidado_productos_aperturados, ultimo_producto_aperturado, on=['Documento', 'FechaApertura'], how='inner')


enrolamiento = '''SELECT 
TipoDocumento as 'Tipo de documento'
,NumDocumento as 'Documento'
,Estado as 'Estadoenrolamiento'
,Nombre as 'Nombre'
,FechaEnrolamiento as 'Fecha de enrolamiento' 
FROM [FABOGSQLCLU].[Usuarios].[dbo].[Usuarios]
where FechaEnrolamiento >= getdate()-7'''
                

# where ProductoAhorro.DMSTAT = 1
enrolamiento = cargue_openquery(conn, enrolamiento)


alerta = pd.merge(ultimo_producto_aperturado,enrolamiento,on='Documento',how='inner')


alerta['Apertura_enrolamiento'] = np.where(alerta['FechaApertura']<=alerta['Fecha de enrolamiento'],1,0)

alerta['DiferenciaEnDias'] = (alerta['Fecha de enrolamiento'] - alerta['FechaApertura']).dt.days


alerta = alerta[alerta['Apertura_enrolamiento']==1]

AI0014= pd.merge(alerta,reconocer,on=['Documento'],how='left')


AI0014 = AI0014[~AI0014['Documento'].isin(vip)]
AI0014['Fecha_ejecucion'] = fecha_actual
AI0014 = AI0014.drop_duplicates()
AI0014 = AI0014[AI0014['Documento']!='0']
# AI0014.to_sql('AI0014',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AI0014.to_sql(name='AI0014', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('14. Alertamiento AI0014 actualizado con éxito')

#%%% AI0015

cuentas_flexi_ayer = '''Select CUNA1 as Nombre, DMACCT as NumeroCuenta,DMDOPN as FechaApertura,DMSTAT as Estado,DMCBAL as saldoactual,DMYBAL as saldodiaanterior,CUSSNR as Documento,DMTYPE as TipoProducto, CUEMA1 as Correo, CUCLPH as Celular,CUNA2 as Direccion
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
                    on Enlace.CUX1CS=Cliente.CUNBR
                    where ProductoAhorro.DMTYPE in (34,23)')
                '''

# where ProductoAhorro.DMSTAT = 1
cuentas_flexi_ayer = cargue_openquery(conn, cuentas_flexi_ayer)

cuentas_flexi_ayer['NumeroCuenta'] = cuentas_flexi_ayer['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()
cuentas_flexi_ayer['Documento'] = cuentas_flexi_ayer['Documento'].fillna(0).astype('int64').astype(str).str.strip()
cuentas_flexi_ayer['Celular'] = cuentas_flexi_ayer['Celular'].fillna(0).astype('int64').astype(str).str.strip()
cuentas_flexi_ayer['FechaApertura'] = cuentas_flexi_ayer['FechaApertura'].astype(int).astype(str).apply(agregar_cero)
cuentas_flexi_ayer['FechaApertura'] = cuentas_flexi_ayer['FechaApertura'].apply(agregar_20)
cuentas_flexi_ayer['FechaApertura'] = cuentas_flexi_ayer['FechaApertura'].apply(convertir_a_fecha)
cuentas_flexi_ayer['FechaApertura'] = pd.to_datetime(cuentas_flexi_ayer['FechaApertura'])
cuentas_flexi_ayer['Estado_Descripcion'] = cuentas_flexi_ayer['Estado'].map(mapeo)


conteo_cuentas = (cuentas_flexi_ayer.pivot_table(index='Documento',columns='Estado_Descripcion', aggfunc='size', fill_value=0)).reset_index()
cuentas_flexi_ayer = cuentas_flexi_ayer[cuentas_flexi_ayer['FechaApertura']>=(datetime.now() - timedelta(days=1))]

AI0015 = pd.merge(cuentas_flexi_ayer,conteo_cuentas,on='Documento',how='inner')
AI0015 = AI0015[(AI0015['Inactiva'] > 0) | (AI0015['Cancelada'] > 0) | (AI0015['Inactiva_(Auditoría)'] > 0)]
AI0015 = AI0015[~AI0015['Documento'].isin(vip)]
AI0015= pd.merge(AI0015,reconocer,on=['Documento'],how='left')
AI0015['Fecha_ejecucion'] = fecha_actual
AI0015 = AI0015[AI0015['Documento']!='0']



fraudes_llamada='''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank FROM [AlertasFraude].[dbo].[BloqueosLlamada]'''
fraudes_llamada = (cx.read_sql(conn = sql_connection, query = fraudes_llamada, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])

# fraudes_llamada['DOCUMENTO'] = fraudes_llamada['DOCUMENTO'].fillna(0).astype('int64').astype(str).str.strip() no sirve

AI0015 = AI0015[AI0015['Documento'].isin(fraudes_llamada['DOCUMENTO'])]
# AI0015.to_sql('AI0015',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AI0015.to_sql(name='AI0015', con=engine_r, if_exists='append', index=False, schema='dbo')

print('15. Alertamiento AI0015 actualizado con éxito')

#%%ALERTAS DE SERVICIO
#%%% AS0001 

''' # Cliente con mas de una tarjeta reexpedida en un dia y mas de 3 tarjetas reexpedidas en un mes # '''


import pandas as pd
from datetime import datetime, timedelta


reexpediciones_tdc = producto_tdc[(producto_tdc['PANANT']!='') & (producto_tdc['FECALTA'] > datetime.now() - timedelta(days=30))]

# reexpediciones_tdc = producto_tdc[(producto_tdc['PANANT']!='')]
reexpediciones_tdc['reexpediciones_totales'] = reexpediciones_tdc.groupby('Documento')['Documento'].transform('count')

# Filtra los registros del último mes y agrega una nueva columna 'Registros_ultimo_mes'
ultimo_mes = reexpediciones_tdc[reexpediciones_tdc['FECALTA'] >= datetime.today() - pd.DateOffset(months=1)]
ultimo_mes['reexpediciones_ultimo_mes'] = ultimo_mes.groupby('Documento')['Documento'].transform('count')

reexpediciones_tdc = reexpediciones_tdc.merge(ultimo_mes[['Documento','reexpediciones_ultimo_mes']], on='Documento', how='left')
# Llena los valores nulos en 'Registros_ultimo_mes' con 0
reexpediciones_tdc['reexpediciones_ultimo_mes'].fillna(0, inplace=True)
conteo = reexpediciones_tdc.groupby(['FECALTA', 'Documento']).size().reset_index(name='Cantidad')
# Crear una nueva columna llamada 'Mas_de_tres' que indica si hay más de tres registros
reexpediciones_tdc['Mas_de_tres_en_un_dia'] = (reexpediciones_tdc['Documento'].apply(lambda x: x in conteo[conteo['Documento'] == x]['Cantidad'].values > 2)).astype(int)
# reexpediciones_tdc['Dias_transcurridos'] = (datetime.now() - reexpediciones_tdc['FECALTA']).dt.days
reexpediciones_tdc = reexpediciones_tdc[(reexpediciones_tdc['Mas_de_tres_en_un_dia']>0)|(reexpediciones_tdc['reexpediciones_ultimo_mes']>3) |(reexpediciones_tdc['reexpediciones_totales']>3)]


AS0001= pd.merge(reexpediciones_tdc,reconocer,on=['Documento'],how='left')
AS0001 = AS0001[~AS0001['Documento'].isin(vip)]

AS0001['Fecha_ejecucion'] = fecha_actual

AS0001 = AS0001.drop_duplicates()
AS0001 = AS0001[AS0001['Documento']!='0']

# AS0001.to_sql('AS0001',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AS0001.to_sql(name='AS0001', con=engine_r, if_exists='append', index=False, schema='dbo')

print('15. Alertamiento AS0001 actualizado con éxito')


#%%% AS0002

''' # Cliente con mas de una tarjeta reexpedida en un dia ó mas de 3 tarjetas reexpedidas en un mes  # '''


datos_tarjeta_debito_limpio='''SELECT  TANROIDE as Documento, TANROTRJ as HPAN, TATIPNOV as CondicionTarjeta, TAFECPRC as FechaCondicion,
  TANROCTA as NumeroCuenta,TAFECEXP as FechaExpiracion,TASUBTIP as TipoTarjeta,TAFECMOD as FechaEstado, TACODEST as EstadoTarjeta
      FROM OPENQUERY (DB2400_182,'select DatosTarjeta.TANROIDE 
	                                    ,DatosTarjeta.TANROTRJ  
								  		,DatosTarjeta.TATIPNOV 
								  		,DatosTarjeta.TAFECPRC 
										,DatosCuentaTarjeta.TANROCTA 
										,DatosCuentaTarjeta.TAFECEXP 
										,DatosCuentaTarjeta.TASUBTIP 
										,DatosCuentaTarjeta.TAFECMOD 
										,DatosCuentaTarjeta.TACODEST  
	                              from BNKPRD01.TAADMAUTF as DatosTarjeta
								  left join (select *
								              from (select TANROCTA
								                          ,TANROIDE
											  		      ,TANROTRJ
														  ,TAFECMOD
								                          ,TAFECEXP
											  		      ,TASUBTIP
														  ,TACODEST
											  		      ,row_number() over(partition by TANROCTA
								                                                         ,TANROIDE
											  		                                     ,TANROTRJ
											  	                        order by TAFECMOD desc) as Duplicados
								                    from BNKPRD01.TAMIGEFPF) as DatosTotalesTarjeta
											  where Duplicados=1) as DatosCuentaTarjeta
								  on cast(DatosTarjeta.TANROIDE as char(40))=cast(DatosCuentaTarjeta.TANROIDE as char(40)) and 
								     cast(DatosTarjeta.TANROTRJ as char(40))=cast(DatosCuentaTarjeta.TANROTRJ as char(40))')'''
datos_tarjeta_debito_limpio =  cargue_openquery(conn, datos_tarjeta_debito_limpio)




datos_tarjeta_debito_limpio=datos_tarjeta_debito_limpio.dropna(subset=['Documento','NumeroCuenta'])
datos_tarjeta_debito_limpio[['Documento','NumeroCuenta']]=datos_tarjeta_debito_limpio[['Documento','NumeroCuenta']].astype('int64').astype(str)
datos_tarjeta_debito_limpio['FechaCondicion'] = pd.to_datetime(datos_tarjeta_debito_limpio['FechaCondicion'].astype(str).str.rstrip('.0'), format='%Y%m%d')
# datos_tarjeta_debito_limpio['Fecha de expiración'] = pd.to_datetime(datos_tarjeta_debito_limpio['Fecha de expiración'].astype(str).str.rstrip('.0'), format='%Y%m%d')
datos_tarjeta_debito_limpio['FechaEstado'] = pd.to_datetime(datos_tarjeta_debito_limpio['FechaEstado'].astype(str).str.rstrip('.0'), format='%Y%m%d')
datos_expedicion_debito = datos_tarjeta_debito_limpio[(datos_tarjeta_debito_limpio['FechaCondicion'] >= pd.to_datetime('today') - pd.DateOffset(days=30)) & (datos_tarjeta_debito_limpio['CondicionTarjeta'] == 'REX')]

# Obtener la fecha del día anterior

# Filtrar los registros del día anterior por documento y contar la cantidad
reexpediciones_dia_anterior = datos_expedicion_debito[datos_expedicion_debito['FechaCondicion'].dt.date == datetime.now().date() - timedelta(days=1)]

datos_expedicion_debito['Reexpediciones_ultimos_30_dias'] = datos_expedicion_debito.groupby('Documento')['Documento'].transform('count')

reexpediciones_dia_anterior['Reexpediciones_dia_anterior'] = reexpediciones_dia_anterior.groupby('Documento')['Documento'].transform('count')

cruce = pd.merge(datos_expedicion_debito,reexpediciones_dia_anterior[['Documento','Reexpediciones_dia_anterior']],on='Documento',how='inner')


alertas_reexpedicion_debito = (cruce[(cruce['Reexpediciones_ultimos_30_dias'] > 3) | (cruce['Reexpediciones_dia_anterior'] > 1)]).drop_duplicates()

alertas_reexpedicion_debito= pd.merge(alertas_reexpedicion_debito,reconocer,on=['Documento'],how='left')


AS0002= pd.merge(alertas_reexpedicion_debito,reconocer,on=['Documento'],how='left')
AS0002 = alertas_reexpedicion_debito[~alertas_reexpedicion_debito['Documento'].isin(vip)]

AS0002['Fecha_ejecucion'] = fecha_actual

AS0002 = AS0002.drop_duplicates()
AS0002 = AS0002[AS0002['Documento']!='0']

AS0002.to_sql('AS0002',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AS0002.to_sql(name='AS0002', con=engine_r, if_exists='append', index=False, schema='dbo')

print('16. Alertamiento AS0002 actualizado con éxito')

# %%% AS0003

''' # Más de dos cuentas de ahorro canceladas o bloqueadas en una semana con nuevas cuentas abiertas a nombre de la misma persona # '''


import pandas as pd


conteos_generales = cuentas_flexi[['Documento','Estado_Descripcion']].pivot_table(index='Documento', columns='Estado_Descripcion', aggfunc='size', fill_value=0)

# El resultado se almacena en 'pivoted_df', que tiene los documentos como filas, las categorías de estado como columnas y la cantidad de registros como valores


aperturas_semana_atras = cuentas_flexi[cuentas_flexi['FechaApertura']>=datetime.now() - timedelta(days=7)]

conteos_generales_semana = aperturas_semana_atras[['Documento','Estado_Descripcion']].pivot_table(index='Documento', columns='Estado_Descripcion', aggfunc='size', fill_value=0)


consolidado_pj = (conteos_generales_semana.merge(conteos_generales, on='Documento', how='left', suffixes=('_semana', '_historico'))).reset_index()


try:
    df_filtrado = consolidado_pj[consolidado_pj[[col for col in consolidado_pj.columns if 'semana' in col]].apply(lambda x: x > 0).any(axis=1) & (consolidado_pj['Cancelada_historico'] > 2)]
except:
    df_filtrado = pd.DataFrame(columns=['Documento'])  # Crear un DataFrame vacío con la columna 'Documento'


# df_filtrado = consolidado_pj[consolidado_pj[[col for col in consolidado_pj.columns if 'semana' in col]].apply(lambda x: x > 0).any(axis=1) & (consolidado_pj['Cancelada_historico'] > 2)]


AS0003= pd.merge(df_filtrado,reconocer,on=['Documento'],how='left')
AS0003 = AS0003[~AS0003['Documento'].isin(vip)]
AS0003['Fecha_ejecucion'] = fecha_actual
AS0003 = AS0003.drop_duplicates()
AS0003 = AS0003[AS0003['Documento']!='0']

# AS0003.to_sql('AS0003',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')


AS0003.to_sql(name='AS0003', con=engine_r, if_exists='append', index=False, schema='dbo')


print('17. Alertamiento AS0003 actualizado con éxito')

#%% ALERTAS MONETARIAS



#%%% AM0001

''' # Transacciones de entrada en el comercio ONOFF soluciones en línea # '''


base_onoff = '''SELECT 
       CAST(CUSSNR AS VARCHAR(255)) AS Documento
      ,CAST(CUX1CS AS VARCHAR(255)) AS IdAS400Cliente
      ,CAST(DHACCT AS VARCHAR(255)) AS NumeroCuenta
      ,DMDOPN as FechaApertura
	  ,DMSTAT AS EstadoCuenta
      ,DHSER  AS NumeroSerieCheque
      ,DHTYP  AS TipoCuenta
	  ,DHTYPE AS CodigoTipoProducto
	  ,CFPRNM AS DescripcionTipoProducto
      ,CAST(DHEFF AS VARCHAR(255)) AS FechaTransaccionEfectiva
	  ,DHDATE AS FechaPublicacion
	  ,DHDSCA AS RegistroAsociadoTransaccion
	  ,DHBTSQ AS CodigoSecuenciaBatch
	  ,DHITC  AS CodigoInternoTransaccion
	  ,DHOTC  AS CodigoTransaccion1
	  ,DHOTC2 AS CodigoTransaccion2
	  ,DHTLTR AS CodigoTransaccionTeller
	  ,DHDRCR AS CodigoDebitoCredito
	  ,CASE WHEN DHDRCR>=6 THEN 'Salida'
	        ELSE 'Entrada'
			  END CaracterTransaccion 
	  ,DHDSC1 AS DescripcionTransaccional1
	  ,DHDSC2 AS DescripcionTransaccional2
	  ,DHDSC3 AS DescripcionTransaccional3
	  ,CFTCD2 AS DescripcionTransaccional4
	  ,DHAMT  AS MontoTransaccion
	  ,DHWHTY AS MontoRetenido
	  ,DHNBR  AS NumeroTransaccionDia
	  ,DHSTRN AS FuenteTransaccion
	  ,DHBRCH AS CodigoSucursal
	  ,CFBRNM AS DescripcionSucursal 
      FROM OPENQUERY(DB2400_182,'select TransaccionesAhorro.DHACCT   
									   ,TransaccionesAhorro.DHSER    
									   ,TransaccionesAhorro.DHTYP    
									   ,TransaccionesAhorro.DHTYPE   
									   ,TransaccionesAhorro.DHEFF    
									   ,TransaccionesAhorro.DHDATE   
									   ,TransaccionesAhorro.DHDSCA   
									   ,TransaccionesAhorro.DHBTSQ   
									   ,TransaccionesAhorro.DHITC    
									   ,TransaccionesAhorro.DHOTC    
									   ,TransaccionesAhorro.DHOTC2   
									   ,TransaccionesAhorro.DHTLTR   
									   ,TransaccionesAhorro.DHDRCR   
									   ,TransaccionesAhorro.DHSTRN   
									   ,TransaccionesAhorro.DHBRCH   
									   ,TransaccionesAhorro.DHAMT    
									   ,TransaccionesAhorro.DHWHTY   
									   ,TransaccionesAhorro.DHNBR
									   ,Ahorro.DMSTAT
                                       ,Ahorro.DMDOPN
									   ,TransaccionesDescritas.DHDSC1
									   ,TransaccionesDescritas.DHDSC2
									   ,TransaccionesDescritas.DHDSC3
									   ,CatalogoTransacciones.CFTCD2
									   ,CatalogoProducto.CFPRNM
									   ,CatalogoSucursales.CFBRNM
									   ,Enlace.CUX1CS
									   ,Cliente.CUSSNR
							     from BNKPRD01.TAP00501 as TransaccionesAhorro
								 left join BNKPRD01.TAP002 as Ahorro
								 on TransaccionesAhorro.DHACCT=Ahorro.DMACCT
							     left join BNKPRD01.TAP020 as TransaccionesDescritas
							     on TransaccionesAhorro.DHACCT=TransaccionesDescritas.DHACCT and
								    TransaccionesAhorro.DHDATE=TransaccionesDescritas.DHDATE and
									TransaccionesAhorro.DHNBR=TransaccionesDescritas.DHNBR
								 left join BNKPRD01.CFP220L3 as CatalogoTransacciones
							     on TransaccionesAhorro.DHITC=CatalogoTransacciones.CFTC
							     left join BNKPRD01.CFP210 as CatalogoProducto
							     on TransaccionesAhorro.DHTYPE=CatalogoProducto.CFTNBR
							     left join BNKPRD01.CFP102 as CatalogoSucursales
							     on TransaccionesAhorro.DHBRCH=CatalogoSucursales.CFBRCH
								 left join (select case when left(ltrim(rtrim(CUX1AC)),1)<>9 then right(ltrim(rtrim(CUX1AC)),length(ltrim(rtrim(CUX1AC)))-1) 
			 	                                        else ltrim(rtrim(CUX1AC))
			 	                                   end as CUX1AC
												  ,CUX1CS
												  ,CUX1AP
												  ,CUXREC
												  ,CUXREL
												  ,CUX1TY
								 from BNKPRD01.CUP009 
								 where CUXREL in (''SOW'',''JOF'')) as Enlace
								 on cast(TransaccionesAhorro.DHACCT as char(30))=cast(Enlace.CUX1AC as char(30))
								 left join BNKPRD01.CUP003 as Cliente
								 on Enlace.CUX1CS=Cliente.CUNBR
								 where TransaccionesDescritas.DHDSC3 in (''ON OFF SOLUCIONE'',''ON OFF SOLUCIONES EN LINEA S.A'',''ON OFF SOLUCIONES EN LINEA SAS'')
								 and  DHEFF>=''2023001'' ')'''

# where ProductoAhorro.DMSTAT = 1

base_onoff = cargue_openquery(conn, base_onoff)

base_onoff['FechaApertura'] = base_onoff['FechaApertura'].astype(int).astype(str).apply(agregar_cero)
base_onoff['FechaApertura'] = base_onoff['FechaApertura'].apply(agregar_20)
base_onoff['FechaApertura'] = base_onoff['FechaApertura'].apply(convertir_a_fecha)


base_onoff = base_onoff[(base_onoff['CaracterTransaccion']=='Entrada') & ((base_onoff['CodigoTipoProducto']==34) | (base_onoff['CodigoTipoProducto']==23) ) ]

AM0001 = pd.merge(base_onoff,reconocer,on=['Documento'],how='left')
AM0001 = AM0001[~AM0001['Documento'].isin(vip)]
AM0001['Fecha_ejecucion'] = fecha_actual
AM0001 = AM0001.drop_duplicates()
AM0001 = AM0001[AM0001['Documento']!='0']

# AM0001.to_sql('AM0001',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AM0001.to_sql(name='AM0001', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('18. Alertamiento AM0001 actualizado con éxito')

# %%% AM0002


''' # Clientes con olvido de contraseña y en el mismo dia realizan transferencias a otras cuentas # '''

datos_demograficos = '''select cussnr as Documento,Cuna1 as Cuna1 from openquery(DB2400_182,'select cussnr,cuna1 from BNKPRD01.cup003 ')'''

# where ProductoAhorro.DMSTAT = 1

datos_demograficos = cargue_openquery(conn, datos_demograficos)


datos_mensajeria_claves_1='''SELECT 
Documento,Fecha,Estrategia
FROM {tabla_datos}
 where Estrategia = 'GestionClaves'
 and IdTipoMensaje = 2
 and Fecha >= GETDATE()-1
'''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_claves_1=load_data(datos_mensajeria_claves_1,config_db_riesgo[1],model_logger=logger)


tx_semana = '''SELECT CAST(CUSSNR AS VARCHAR(255)) AS Documento,CAST(TLTACT AS VARCHAR(255)) as NumeroCuenta,TLPRDT as FechaTx,TLTTC as Monto,TLTAL1 as Descripcion
FROM OPENQUERY(DB2400_182, 
    'SELECT transacciones.TLTACT,transacciones.TLPRDT,transacciones.TLTTC,transacciones.TLTAL1,demograficos.CUSSNR
     FROM BNKSRV01.PST001L92 as transacciones
     left join BNKPRD01.cup003 as demograficos
     ON transacciones.TLTSHN = demograficos.CUNA1
     WHERE transacciones.TLTRCR >= ''6'' AND transacciones.TLTAPP = 26 AND transacciones.TLPRDT >= CURRENT DATE - 1 DAY'
)
                '''
                




# where ProductoAhorro.DMSTAT = 1
tx_semana = cargue_openquery(conn, tx_semana)




tx_semana['Descripcion'] = tx_semana['Descripcion'].str.upper()



tx_semana = tx_semana[(tx_semana['Descripcion'].str.contains('TRANSFIYA')) | (tx_semana['Descripcion']=='TRANSFERENCIA INMEDIATA')]

tx_semana['TipoTransaccion']=np.where(tx_semana['Descripcion'].str.contains('TRANSFIYA'),'Transfiya','Transferencia')


tx_semana = tx_semana.groupby(['Documento', 'TipoTransaccion']).agg({
    'Documento': 'count',  # Contar la cantidad de registros
    'Monto': 'sum',         # Sumar la columna H
    'FechaTx': ['min']  # Encontrar el valor mínimo y máximo de la columna Fecha
})


tx_semana.columns = ['Transacciones', 'Monto', 'FechaTx']

tx_semana=tx_semana.reset_index()




alerta=pd.merge(tx_semana,datos_mensajeria_claves_1,on='Documento',how='inner' )


AM0002= pd.merge(alerta,reconocer,on=['Documento'],how='left')


AM0002 = AM0002[~AM0002['Documento'].isin(vip)]
AM0002['Fecha_ejecucion'] = fecha_actual
AM0002 = AM0002.drop_duplicates()
AM0002 = AM0002[AM0002['Documento']!='0']


# AM0002.to_sql('AM0002',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')


AM0002.to_sql(name='AM0002', con=engine_r, if_exists='append', index=False, schema='dbo')


print('19. Alertamiento AM0002 actualizado con éxito')


# %%% AM0007


fecha_tx = (datetime.now()- timedelta(days=1)).strftime('%Y-%m-%d')

transacciones_ayer = f'''Select CAST(MCNROCTA AS VARCHAR(255)) as NumeroCuenta,MCFHRPTA as Fecha,MCVLRMOV as valor,MCDEBCRE as TipoTransaccion
from openquery (DB2400_182,'select MCNROCTA,MCFHRPTA,MCVLRMOV,MCDEBCRE from BNKPRD01.EXTMOVCAJ
                where  MCFHRPTA >= ''{fecha_tx}'' ')
                '''
# where ProductoAhorro.DMSTAT = 1

transacciones_ayer = cargue_openquery(conn, transacciones_ayer)

transacciones_ayer['TipoTransaccion'] = transacciones_ayer['TipoTransaccion'].astype(int)

transacciones_ayer['TipoTransaccion'] = np.where(transacciones_ayer['TipoTransaccion']>=6,'Salidas','Entradas')


# Agrupar por 'Cuenta' y realizar operaciones de agregación
import pandas as pd

# Supongamos que ya tienes un DataFrame llamado df

# Agrupar por 'Cuenta' y 'OtroCampo' y realizar operaciones de agregación
transacciones_ayer = transacciones_ayer.groupby(['NumeroCuenta', 'TipoTransaccion']).agg({
    'NumeroCuenta': 'count',  # Contar la cantidad de registros
    'valor': 'sum',         # Sumar la columna H
    'Fecha': ['min', 'max']  # Encontrar el valor mínimo y máximo de la columna Fecha
})

# Cambiar el nombre de las columnas resultantes
transacciones_ayer.columns = ['Transacciones', 'Monto', 'Fecha_minima', 'Fecha_maxima']

transacciones_ayer=transacciones_ayer.reset_index()
transacciones_ayer['Alertamiento_altas_salidas'] = np.where((transacciones_ayer['Monto']>5000000) & (transacciones_ayer['TipoTransaccion']=='Salidas'),1,0)
transacciones_ayer['Alertamiento_altas_entradas'] = np.where((transacciones_ayer['Monto']>8000000) & (transacciones_ayer['TipoTransaccion']=='Entradas'),1,0)


alerta= pd.merge(transacciones_ayer, cuentas_flexi,on='NumeroCuenta',how='inner')
alerta = alerta[alerta[[col for col in alerta.columns if 'Alertamiento' in col]].apply(lambda x: x > 0).any(axis=1)]
alerta['Documento'] = alerta['Documento'].fillna(0).astype('int64').astype(str).str.strip() 
alerta['Celular'] = alerta['Celular'].fillna(0).astype('int64').astype(str).str.strip() 

AM0007= pd.merge(alerta,reconocer,on=['Documento'],how='left')
AM0007 = AM0007[~AM0007['Documento'].isin(vip)]
AM0007['Fecha_ejecucion'] = fecha_actual
AM0007 = AM0007.drop_duplicates()
AM0007 = AM0007[AM0007['Documento']!='0']

# AM0007.to_sql('AM0007',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AM0007.to_sql(name='AM0007', con=engine_r, if_exists='append', index=False, schema='dbo')

print('20. Alertamiento AM0007 actualizado con éxito')


#%%% AM0009 ajustada

''' # Clientes que no posean transferencias en los últimos 60 dias y empiecen a realizar movimientos # '''

ultima_tx = '''SELECT MCNROCTA as NumeroCuenta,ultimatx
FROM OPENQUERY(DB2400_182,
   'SELECT MCNROCTA, MAX(MCFHRPTA) AS ultimatx
    FROM BNKPRD01.EXTMOVCAJ
    GROUP BY MCNROCTA'
)
where ultimatx <= getdate()-60'''
                

# where ProductoAhorro.DMSTAT = 1
ultima_tx = cargue_openquery(conn, ultima_tx)



tx_ayer = '''SELECT CAST(TLTACT AS VARCHAR(255)) AS NumeroCuenta,maxtx as txayer,tx
FROM OPENQUERY(DB2400_182, 
    'SELECT demograficos.CUSSNR,transacciones.TLTACT,max(transacciones.TLPRDT) as maxtx,count(*) as tx
     FROM BNKSRV01.PST001L92 as transacciones
     left join BNKPRD01.cup003 as demograficos
     ON transacciones.TLTSHN = demograficos.CUNA1
     WHERE transacciones.TLTRCR >= ''6'' AND transacciones.TLTAPP = 26 AND transacciones.TLPRDT = CURRENT DATE - 1 DAY
     GROUP BY demograficos.CUSSNR,transacciones.TLTACT')'''
                

# where ProductoAhorro.DMSTAT = 1
tx_ayer = cargue_openquery(conn, tx_ayer)

ultima_tx['NumeroCuenta'] = ultima_tx['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()
tx_ayer['NumeroCuenta'] = tx_ayer['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()

alerta= pd.merge(ultima_tx, tx_ayer,on='NumeroCuenta',how='inner')

alerta= pd.merge(alerta, cuentas_flexi,on='NumeroCuenta',how='inner')
AM0009= pd.merge(alerta,reconocer,on=['Documento'],how='left')
AM0009 = AM0009[~AM0009['Documento'].isin(vip)]
AM0009['Fecha_ejecucion'] = fecha_actual
AM0009 = AM0009.drop_duplicates()
AM0009 = AM0009[AM0009['Documento']!='0']
# AM0009.to_sql('AM0009',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')
AM0009.to_sql(name='AM0009', con=engine_r, if_exists='append', index=False, schema='dbo')

print('21. Alertamiento AM0009 actualizado con éxito')

#%%% AM0010 ajustado

tx_ayer = '''select * from openquery(DB2400_182,'select mcnrocta,mcdestra,mcdebcre,mcvlrmov,mcobser from bnkprd01.extmovcaj where mcobser not in(''GRAVAMEN MOV. FINANCIERO'',''Banco Finandina. Pago NORMAL'') AND MCFHRPTA >= CURRENT DATE - 2 DAY  
')'''

tx_ayer = (cargue_openquery(conn, tx_ayer))

base=tx_ayer[(tx_ayer['MCOBSER'].str.contains('Transferencia Inmediata')) | (tx_ayer['MCOBSER'].str.contains('E-TransfiYa'))] 

base['categoria'] = np.where(base['MCOBSER'].str.contains('Transferencia Inmediata'),'movimiento_a_cuentas_FA','Transfiya')

agrupado = base.groupby(['MCNROCTA', 'categoria']).agg({'MCVLRMOV': ['count', 'sum']}).reset_index()

# Renombra las columnas resultantes
agrupado.columns = ['NumeroCuenta', 'categoria', 'cantidad_tx', 'suma_monto_tx']

agrupado['NumeroCuenta'] = agrupado['NumeroCuenta'].fillna(0).astype('int64').astype(str).str.strip()

pivotado = agrupado.pivot_table(index='NumeroCuenta', columns='categoria', values=['cantidad_tx', 'suma_monto_tx'], fill_value=0, aggfunc='sum', margins=True, margins_name='Total', dropna=False)

# Eliminar el nombre del índice de las columnas
pivotado.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivotado.columns]

# Conservar 'MCNROCTA' como índice
pivotado = pivotado.reset_index()


alerta=pivotado[(pivotado['cantidad_tx_Transfiya']>0) & (pivotado['cantidad_tx_movimiento_a_cuentas_FA']>0) & (pivotado['NumeroCuenta']!='Total')  ]

alerta = pd.merge(alerta,cuentas_flexi,on='NumeroCuenta',how='inner')


AM0010= pd.merge(alerta,reconocer,on=['Documento'],how='left')
AM0010 = AM0010[~AM0010['Documento'].isin(vip)]
AM0010['Fecha_ejecucion'] = fecha_actual
AM0010 = AM0010.drop_duplicates()
AM0010 = AM0010[AM0010['Documento']!='0']
#AM0010 = AM0010[AM0010['Documento'].isin(fraudes_llamada['DOCUMENTO'])]
#AM0010 = AM0010[AM0010['Documento'].isin(auditoria['ID_Cliente'])]

# AM0010.to_sql('AM0010',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AM0010.to_sql(name='AM0010', con=engine_r, if_exists='append', index=False, schema='dbo')

print('22. Alertamiento AM0010 actualizado con éxito')




#%%% AM0011_AM0012

''' # Aquellos clientes que sobrapasen su deuda en un 30% adicional de lo que se otorgo como cupo disponible pago con cheque
   Alertamiento de TDC que superan el 30% de su cupo aprobado # '''



query_cupo_limite = '''select *  FROM OPENQUERY(DB2400_182,' select ContratoTDC.CUENTA 
								 	   ,ContratoTDC.NUMBENCTA
									   ,ContratoTDC.CALPART 
                                        ,ContratoTDC.FECALTA 
						   		 	   ,ContratoTDC.FECBAJA
									   ,TarjetaTDC.PAN
									   ,TarjetaTDC.INDSITTAR   
									   ,TarjetaTDC.FECBAJA AS FECBAJAT
									   ,TarjetaTDC.FECALTA AS FECALTAT
									   ,InformacionComunTDC.NUMDOC
									   ,InformacionComunTDC.ESTPER
									   ,TarjetaClienteTDC.INDBLQOPE
                                        ,SuptiposTarjetaTDC.DESPROD 
									   ,FinancierosCuentaTDC.ITVRCUAC
									   ,FinancierosCuentaTDC.ITVRCUDC
									   ,FinancierosCuentaTDC.ITVLRSDO  
                                    	   from INTTARCRE.SATBENEFI as ContratoTDC
							     inner join INTTARCRE.SATTARJET as TarjetaTDC
							     on ContratoTDC.CUENTA   =TarjetaTDC.CUENTA and
							        ContratoTDC.NUMBENCTA=TarjetaTDC.NUMBENCTA
							     inner join INTTARCRE.SATCTATAR as TarjetaClienteTDC
							     on ContratoTDC.CUENTA=TarjetaClienteTDC.CUENTA 
							     inner join INTTARCRE.SATPLASTI as TarjetaPlasticoTDC
							     on TarjetaTDC.CUENTA    =TarjetaPlasticoTDC.CUENTA and
							        TarjetaTDC.NUMPLASTIC=TarjetaPlasticoTDC.NUMPLAS
							     inner join INTTARCRE.SATTARLIM as LimitesTarjetaTDC
							     on ContratoTDC.CUENTA   =LimitesTarjetaTDC.CUENTA and
							        ContratoTDC.NUMBENCTA=LimitesTarjetaTDC.NUMBENCTA
							     inner join INTTARCRE.SATDACOPE as InformacionComunTDC
							     on ContratoTDC.IDENTCLI=InformacionComunTDC.IDENTCLI
								 inner join INTTARCRE.SATTIPTAR as TipoTarjetaTDC
							     on TarjetaTDC.CODMAR =TipoTarjetaTDC.CODMAR and
							        TarjetaTDC.INDTIPT=TipoTarjetaTDC.INDTIPT
							     left join INTTARCRE.SATPRODUC as SuptiposTarjetaTDC
							     on TarjetaClienteTDC.SUBPRODU=SuptiposTarjetaTDC.SUBPRODU
							     left join INTTARCRE.SATCODBLQ as BloqueoTDC
							     on TarjetaTDC.CODBLQ=BloqueoTDC.CODBLQ
								 left join INTERFACES.INTDIATEC as FinancierosCuentaTDC
								 on TarjetaTDC.CUENTA=FinancierosCuentaTDC.ITNROCTO and
								    TarjetaTDC.PAN=FinancierosCuentaTDC.ITNROPAN
								 left join BNKPRD01.CFP102 as CatalogoSucursales
								 on TarjetaTDC.CENTALTA=CatalogoSucursales.CFBRCH')
                '''

# where ProductoAhorro.DMSTAT = 1

cupo_limite = (cargue_openquery(conn, query_cupo_limite)).rename(columns={'CUENTA':'NumeroContrato',
                                                                                   'NUMBENCTA':'NumeroBeneficiario',
                                                                                   'FECALTA':'FechaAltaContrato',
                                                                                   'FECBAJA':'FechaBajaContrato',
                                                                                   'PAN':'NumeroTarjeta',
                                                                                   'FECBAJAT':'FechaBajaTarjeta',
                                                                                   'FECALTAT':'FechaAltaTarjeta',
                                                                                   'NUMDOC':'Documento',
                                                                                   'ITVRCUAC':'CupoAprobadoTarjeta',
                                                                                   'ITVRCUDC':'CupoDisponibleTarjeta',
                                                                                   'ITVLRSDO':'SaldoActualTarjeta',
                                                                                   'CALPART':'TipoBeneficiario',
                                                                                   'DESPROD':'DescripcionSubproductoTarjeta'})



# limpieza de datossss

cupo_limite['Documento']=cupo_limite['Documento'].astype(str).str.strip()
cupo_limite['NumeroTarjeta']=cupo_limite['NumeroTarjeta'].astype(str).str.strip()
cupo_limite['FechaAltaContrato']=cupo_limite['FechaAltaContrato'].astype(str).str.strip()
cupo_limite['FechaBajaContrato']=cupo_limite['FechaBajaContrato'].astype(str).str.strip()
cupo_limite['FechaBajaTarjeta']=cupo_limite['FechaBajaTarjeta'].astype(str).str.strip()
cupo_limite['FechaAltaTarjeta']=cupo_limite['FechaAltaTarjeta'].astype(str).str.strip()
cupo_limite['DescripcionSubproductoTarjeta'] = cupo_limite['DescripcionSubproductoTarjeta'].str.upper()
cupo_limite['ESTPER'] = cupo_limite['ESTPER'].astype(str).str.strip()
cupo_limite['INDBLQOPE'] = cupo_limite['INDBLQOPE'].astype(str).str.strip()
cupo_limite['INDSITTAR'] = cupo_limite['INDSITTAR'].astype(int)

# Supongamos que tienes un DataFrame llamado df que contiene la columna INDSITTAR
# Definir un diccionario que mapee los valores de INDSITTAR a las descripciones correspondientes

descripciones = {
    1: 'SELECCIONADA ESTAMPACION ALTA',
    2: 'PENDIENTE ACUSE POR ALTA',
    3: 'SELECCIONADA ESTAM. RENOVACION',
    4: 'PENDIENTE ACUSE POR RENOVACION',
    5: 'EN PODER DEL CLIENTE',
    6: 'PENDIENTE CAMBIO PIN POR ALTA',
    7: 'PDTE. PRIMERA OP. PIN OK ALTA',
    8: 'RECOGIDA POR BAJA',
    9: 'PENDIENTE RECOGER BAJA',
    10: 'NO EMITIDA',
    11: 'PENDIENTE POR RECOGER OTRA',
    12: 'SELEC. ESTAMPACION ALTA MASIVA',
    13: 'PENDIENTE ACUSE POR ALTA MAS.',
    14: 'INACTIVA REEM/RENO CAMBIO PAN',
    15: 'ERROR EN ESTAMPACION',
    16: 'PENDIENTE CAMBIO PIN RENOVAC.',
    17: 'PDTE. PRIMERA OP. PIN OK RENO',
    18: 'ACTIVA EN PERIODO DE RENOVAC.',
    19: 'SELECCIONADA ESTAM. REEMISION',
    20: 'PENDIENTE ACUSE POR REEMISION',
    21: 'PENDIENTE CAMBIO PIN REEMISION',
    22: 'PDTE. PRIMERA OP. PIN OK REEM',
}

# Aplicar el mapeo utilizando la función map() y crear la columna adicional
cupo_limite['DescripcionSituacionTarjeta'] = cupo_limite['INDSITTAR'].map(descripciones)


## estamación estado de la tarjeta 

cupo_limite['DescripcionEstadoGeneralTarjeta'] = cupo_limite.apply(lambda row: 
    'Activa' if row['FechaBajaContrato'] == '0001-01-01' and row['FechaBajaTarjeta'] == '0001-01-01' and row['INDBLQOPE'] == 'N' and row['ESTPER'] == 'A' and row['DescripcionSituacionTarjeta'] == 'EN PODER DEL CLIENTE'
    else 'Bloqueada' if row['FechaBajaContrato'] == '0001-01-01' and row['FechaBajaTarjeta'] == '0001-01-01' and row['INDBLQOPE'] == 'S'
    else 'Cancelada' if row['FechaBajaContrato'] != '0001-01-01' or row['FechaBajaTarjeta'] != '0001-01-01'
    else 'Inactiva/Indeterminada',
    axis=1
)
    

# Agregar una nueva columna y asignar valores 1 o 0 según la condición
cupo_limite['TipoTarjeta'] = cupo_limite['DescripcionSubproductoTarjeta'].apply(lambda x: 'DIGITAL' if 'DIGITAL' in x else 'FISICA')



query_pagos_cheque = '''SELECT  *
FROM openquery(DB2400_182, '
    SELECT *
    FROM INTTARCRE.CANMAND
    WHERE DTIPMO = 24 AND FECHACREA >= CURRENT_TIMESTAMP - 60 DAYS   
')'''

# where ProductoAhorro.DMSTAT = 1

pagos_cheques = (cargue_openquery(conn, query_pagos_cheque)).rename(columns={ 'DBINFU':'BINFUENTE',
                                                                                        'DBINDE':'BINDESTINO',
                                                                                        'FECHACREA':'FECHACHEQUE',
                                                                                        'DFECPR':'FECHACANJE',
                                                                                        'DTARJE':'NumeroTarjeta',
                                                                                        'DVALTO':'VALORCHEQUE'})[['BINFUENTE','BINDESTINO','FECHACHEQUE','FECHACANJE','NumeroTarjeta','VALORCHEQUE']]



cupo_limite['CantidadTarjetas'] = cupo_limite.groupby('Documento')['Documento'].transform('count')
cupo_limite['CupoTotal'] = cupo_limite.groupby('Documento')['SaldoActualTarjeta'].transform('sum')
cupo_limite['SaldoTotal'] = cupo_limite.groupby('Documento')['CupoAprobadoTarjeta'].transform('sum')
cupo_limite['Superaparametrolocal']=np.where(cupo_limite['SaldoActualTarjeta']> (1.3 * cupo_limite['CupoAprobadoTarjeta']),1,0)
cupo_limite['Superaparametroglobal']=np.where(cupo_limite['SaldoTotal']> (1.3 * cupo_limite['CupoTotal']),1,0)
cupo_limite=cupo_limite[((cupo_limite['Superaparametrolocal']==1) | (['Superaparametroglobal']==1)) & (cupo_limite['DescripcionEstadoGeneralTarjeta']!='Cancelada')]
cupo_limite=cupo_limite.sort_values('Documento',ascending=True)
cupo_limite=cupo_limite.drop(['INDSITTAR', 'ESTPER', 'INDBLQOPE','NumeroBeneficiario'], axis=1)



alerta_cupo_excedido = pd.merge(cupo_limite, pagos_cheques, on='NumeroTarjeta', how='inner')

alerta_cupo_excedido=(alerta_cupo_excedido[alerta_cupo_excedido['DescripcionEstadoGeneralTarjeta']!='Cancelada']).rename(columns={'DocumentoCliente':'Documento'})

alerta_cupo_excedido= pd.merge(alerta_cupo_excedido,reconocer,on=['Documento'],how='left')

alerta_cupo_excedido = alerta_cupo_excedido[~alerta_cupo_excedido['Documento'].isin(vip)]

alerta_cupo_excedido['Fecha_ejecucion'] = fecha_actual

alerta_cupo_excedido = alerta_cupo_excedido.drop_duplicates()
alerta_cupo_excedido = alerta_cupo_excedido[alerta_cupo_excedido['Documento']!='0']


AM0011=alerta_cupo_excedido
# AM0011.to_sql('AM0011',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AM0011.to_sql(name='AM0011', con=engine_r, if_exists='append', index=False, schema='dbo')

cupo_limite = cupo_limite.drop_duplicates()
cupo_limite = cupo_limite[cupo_limite['Documento']!='0']


AM0012=cupo_limite
# AM0012.to_sql('AM0012',con=config_db_fraude[1],if_exists='replace',index=False,schema='dbo')

AM0012.to_sql(name='AM0012', con=engine_r, if_exists='replace', index=False, schema='dbo')


print('23. Alertamiento AM0011_AM0012 actualizados con éxito')



#%%% AM0013 Alertamiento extmovcaj


# Alertamiento SARLAFT 

# def conexion_fabogriesgo():
#     server_riesgo = "192.168.60.152:49505"
#     database_riesgo = "Productos y transaccionalidad"
#     # user='Usr_lkfraude'
#     # password='Sq5q7v@K67nw'
#     user='FINANDINA\josgom'
#     password=str(lineas[1].strip())
#     conn_riesgo = connect_to_database(server_riesgo, database_riesgo, user, password)
  
#     return conn_riesgo



def connect_to_database2(server, database, user, password):
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
    conn_riesgo = connect_to_database2(server_riesgo, database_riesgo, user, password)
  
    return conn_riesgo

conexion = conexion_fabogriesgo()

fecha_inicial = ((datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0)).strftime('%Y-%m-%d %H:%M:%S')

fecha_final = ((datetime.now() - timedelta(days=1)).replace(hour=11, minute=59, second=59)).strftime('%Y-%m-%d %H:%M:%S')


# query_cuenta_documento='''SELECT [DocumentoCliente]
#       ,[NumeroCuenta] as Numero_Cuenta
# FROM [Productos y transaccionalidad].[dbo].[ConsolidadoProductos]'''
# documento_cuenta = pd.read_sql(query_cuenta_documento,conexion_fabogriesgo())




query_cuenta_documento='''select CAST(CUSSNR AS VARCHAR(255)) as Documento,CAST(DMACCT AS VARCHAR(255)) AS Numero_Cuenta from openquery(DB2400_182,'select  Cliente.cussnr,ProductoAhorro.DMACCT from BNKPRD01.TAP002 
 AS ProductoAhorro
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
									   on Enlace.CUX1CS=Cliente.CUNBR ')'''
# documento_cuenta = pd.read_sql(query_cuenta_documento,conexion_fabogriesgo())

documento_cuenta = pd.read_sql(query_cuenta_documento,conexion)


query_entradas_extmovcaj= f'''Select
MCFECCAL AS fecha_publicación,
MCOBSER As Descripcion_Transacción,
MCFHRPTA As Fecha_hora_transacción,
MCVLRMOV As Valor_Transacción,
MCNOMCTA As Nombre_Titular,
CAST(MCNROCTA AS VARCHAR(255)) AS Numero_Cuenta,
MCDEBCRE As Tipo_Transacción,
MCCAJA As Cod_Transacción,
MCDESTRA As lugar_transacción
from openquery (DB2400_182,'select * from BNKPRD01.EXTMOVCAJ 
where MCDEBCRE IN(''0'',''1'',''2'',''3'',''4'',''5'') and MCFHRPTA>= ''{fecha_inicial}'' AND MCFHRPTA < ''{fecha_final}'' 
')'''

entradas_exmovcaj = pd.read_sql(query_entradas_extmovcaj,conexion)
entradas_exmovcaj['Fecha_hora_transacción'] = pd.to_datetime(entradas_exmovcaj['Fecha_hora_transacción'], format='%Y-%m-%d %H:%M:%S.%f')


query_salidas_extmovcaj=f'''Select
MCFECCAL AS fecha_publicación,
MCOBSER As Descripcion_Transacción,
MCFHRPTA As Fecha_hora_transacción,
MCVLRMOV As Valor_Transacción,
MCNOMCTA As Nombre_Titular,
CAST(MCNROCTA AS VARCHAR(255)) AS Numero_Cuenta,
MCDEBCRE As Tipo_Transacción,
MCCAJA As Cod_Transacción,
MCDESTRA As lugar_transacción
from openquery (DB2400_182,'select * from BNKPRD01.EXTMOVCAJ 
where MCDEBCRE NOT IN(''0'',''1'',''2'',''3'',''4'',''5'') and MCFHRPTA>= ''{fecha_inicial}'' AND MCFHRPTA < ''{fecha_final}'' 
')'''
salidas_exmovcaj = pd.read_sql(query_salidas_extmovcaj,conexion)
salidas_exmovcaj['Fecha_hora_transacción'] = pd.to_datetime(salidas_exmovcaj['Fecha_hora_transacción'], format='%Y-%m-%d %H:%M:%S.%f')




resultados_salidas = salidas_exmovcaj.groupby('Numero_Cuenta').agg(
    Cantidad_Transacciones_salidas=pd.NamedAgg(column='Fecha_hora_transacción', aggfunc='count'),
    Fecha_Minima_salidas=pd.NamedAgg(column='Fecha_hora_transacción', aggfunc='min'),
    Fecha_Maxima_salidas=pd.NamedAgg(column='Fecha_hora_transacción', aggfunc='max'),
    Valor_Total_Monto_salidass=pd.NamedAgg(column='Valor_Transacción', aggfunc='sum'),
    Promedio_Monto_salidas=pd.NamedAgg(column='Valor_Transacción', aggfunc='mean'),
    Transaccion_Mayor_10M_salidas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 10000000)),
    Transaccion_Mayor_20M_salidas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 20000000)),
    Transaccion_Mayor_50M_salidas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 50000000)),
    Transaccion_Mayor_100M_salidas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 100000000))
).reset_index()



resultados_entradas = entradas_exmovcaj.groupby('Numero_Cuenta').agg(
    Cantidad_Transacciones_entradas=pd.NamedAgg(column='Fecha_hora_transacción', aggfunc='count'),
    Fecha_Minima_entradas=pd.NamedAgg(column='Fecha_hora_transacción', aggfunc='min'),
    Fecha_Maxima_entradas=pd.NamedAgg(column='Fecha_hora_transacción', aggfunc='max'),
    Valor_Total_Monto_entradas=pd.NamedAgg(column='Valor_Transacción', aggfunc='sum'),
    Promedio_Monto_entradas=pd.NamedAgg(column='Valor_Transacción', aggfunc='mean'),
    Transaccion_Mayor_10M_entradas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 10000000)),
    Transaccion_Mayor_20M_entradas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 20000000)),
    Transaccion_Mayor_50M_entradas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 50000000)),
    Transaccion_Mayor_100M_entradas=pd.NamedAgg(column='Valor_Transacción', aggfunc=lambda x: any(x > 100000000))
).reset_index()


# Concatena las columnas y crea la Serie 'movimientos'.
movimientos = pd.concat([entradas_exmovcaj['Numero_Cuenta'], salidas_exmovcaj['Numero_Cuenta']])
# Obtén los valores únicos de la Serie 'movimientos' y conviértelos en un DataFrame.
valores_unicos_df = pd.DataFrame({'Numero_Cuenta': movimientos.unique()})

Alerta_sarlaft = pd.merge(valores_unicos_df, resultados_entradas,on='Numero_Cuenta',how = 'left')

Alerta_sarlaft = pd.merge(Alerta_sarlaft, resultados_salidas,on='Numero_Cuenta',how = 'left')

Alertamiento_entradas_salidas = Alerta_sarlaft[(Alerta_sarlaft['Transaccion_Mayor_10M_entradas']==True) |(Alerta_sarlaft['Transaccion_Mayor_10M_entradas']==True) | (Alerta_sarlaft['Transaccion_Mayor_20M_entradas']==True) | (Alerta_sarlaft['Transaccion_Mayor_50M_entradas']==True) | (Alerta_sarlaft['Transaccion_Mayor_100M_entradas']==True) |(Alerta_sarlaft['Transaccion_Mayor_10M_salidas']==True) | (Alerta_sarlaft['Transaccion_Mayor_20M_salidas']==True) | (Alerta_sarlaft['Transaccion_Mayor_50M_salidas']==True) | (Alerta_sarlaft['Transaccion_Mayor_100M_salidas']==True) ]
AM0013 = pd.merge(Alertamiento_entradas_salidas, documento_cuenta,on='Numero_Cuenta',how = 'left')


AM0013['FechaAlerta']=fecha_actual
AM0013 = AM0013[AM0013['Documento']!='0']

# AM0013.to_sql('AM0013',con=config_db_fraude[1],if_exists='append',index=False,schema='dbo')

AM0013.to_sql(name='AM0013', con=engine_r, if_exists='append', index=False, schema='dbo')

import pandas as pd


# Supongamos que tienes tus DataFrames con los nombres que proporcionaste
# Por ejemplo, df_AI0001, df_AI0002, ..., df_AM0012

# AM0002 = pd.DataFrame(columns=['Documento'])

# Lista de nombres de DataFrames
nombres_dfs = ["AI0001", "AI0002", "AI0003", "AI0004", "AI0005", "AI0006", "AI0007", "AI0008", "AI0009",
               "AI0010", "AI0011", "AI0012", "AI0013", "AI0014","AI0015", "AS0001", "AS0002", "AS0003",
               "AM0001", "AM0002", "AM0007", "AM0009", "AM0010", "AM0011", "AM0012","AM0013"]

# Crear un DataFrame para almacenar la cantidad de registros
df_cantidad_registros = pd.DataFrame(columns=["Alertamiento", "Registros"])

# Calcular y almacenar la cantidad de registros para cada DataFrame
for nombre_df in nombres_dfs:
    # Supongamos que tienes los DataFrames en variables con los nombres proporcionados
    # Por ejemplo, df_AI0001, df_AI0002, ..., df_AM0012
    cantidad_registros = globals()[nombre_df].shape[0]
    
    # Agregar la información al nuevo DataFrame
    df_cantidad_registros = pd.concat([df_cantidad_registros, pd.DataFrame({"Alertamiento": [nombre_df], "Registros": [cantidad_registros]})], ignore_index=True)

# Mostrar el DataFrame con la cantidad de registros
# print(df_cantidad_registros)




#%% Malla de alertamientos


alertas = {
 'AI0001' : AI0001
,'AI0002' : AI0002
,'AI0003' : AI0003
,'AI0004' : AI0004
,'AI0005' : AI0005
,'AI0006' : AI0006
,'AI0007' : AI0007
,'AI0008' : AI0008
,'AI0009' : AI0009
,'AI0010' : AI0010
,'AI0011' : AI0011
,'AI0012' : AI0012
,'AI0013' : AI0013
,'AI0014' : AI0014
,'AI0015' : AI0015
,'AS0001' : AS0001
,'AS0002' : AS0002
,'AS0003' : AS0003
,'AM0001' : AM0001
,'AM0002' : AM0002
,'AM0007' : AM0007
,'AM0009' : AM0009
,'AM0010' : AM0010
,'AM0011' : AM0011
,'AM0012' : AM0012
,'AM0013' : AM0013}

# ,'AlertaDisparidadCIIU' esta base tiene que ir en el espacio en blanco anteriorrrr

# ,'AlertaActualizacionRiesgo' : alerta_actualizan_reciente
# ,'AlertaSimilitudIPCreacion' : alerta_creacion_cantidad_ip


documentos = set()
for df in alertas.values(): documentos.update(df['Documento'].drop_duplicates())


documentos = list(documentos)

# Crea un DataFrame vacío con los documentos únicos como índices
malla_pivoteada = pd.DataFrame(index=documentos)

# Llena la malla pivoteada con 1 si el documento está en la alerta y 0 si no
for alerta, dataframe in alertas.items():malla_pivoteada[alerta] = malla_pivoteada.index.isin(dataframe['Documento']).astype(int)

# Llena los valores faltantes con 0
malla_pivoteada.fillna(0, inplace=True)

minima_alerta_generada=(malla_pivoteada.reset_index()).rename(columns={'index':'Documento'})

columnas_a_sumar = [col for col in minima_alerta_generada.columns if col != "Documento"]

# Crea la nueva columna "SumaColumnas" que contiene la suma de las columnas
minima_alerta_generada["TotalAlertamientos"] = minima_alerta_generada[columnas_a_sumar].sum(axis=1)

# minima_alerta_generada.to_sql('ReporteConsolidadoAlertasPreguntasNegocio',con=config_db_alerta[1],if_exists='replace',index=False,schema='dbo')

minima_alerta_generada.to_sql(name='ReporteConsolidadoAlertasPreguntasNegocio', con=engine_r, if_exists='replace', index=False, schema='dbo')

print('Malla de alertamientos enviada')


tiempo_ejecucion_codigo = round(float(format(time.time() - startTime0))/60,2)

#-------------------------------------------------------------------------------------------------#
#                                     ENVÍO DE RESULTADOS                                         #
#-------------------------------------------------------------------------------------------------#

# skipped your comments for readability
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

me = "jose.gomezv@bancofinandina.com"
my_password = r"{}".format(str(lineas[2].strip()))

# you=["jose.gomezv@bancofinandina.com" , "jesus.alvear@bancofinandina.com","astrid.bermudez@bancofinandina.com","sandra.suarez@bancofinandina.com"]
you=["jose.gomezv@bancofinandina.com" ,"sandra.suarez@bancofinandina.com","julian.hernandez@bancofinandina.com"]
# you=["jose.gomezv@bancofinandina.com"]
msg = MIMEMultipart('alternative')
msg['Subject'] = "Actualización Alertamientos de FRAUDE"
msg['From'] = me
msg['To'] = ",".join(you)



# Convertir el DataFrame a una tabla HTML
html_table = df_cantidad_registros.to_html(index=False)

# Tu código HTML existente...
html = """\
<html>
  <head>
    <style>
      table {
        border-collapse: collapse;
        width: 100%;
      }

      th, td {
        border: 1px solid #dddddd;
        text-align: center; /* Centra el texto en las celdas */
        padding: 8px;
      }

      th {
        background-color: #f2f2f2;
      }
    </style>
  </head>
  <body>
    <p> 📊 📈 Buen Día, el presente correo contiene la última fecha de los alertamientos de fraude 📊 📈<br>
    ✅ RESULTADOS OBTENIDOS : 🕓 """ + str(fecha_actual) + """<br>
    """" ✅ TIEMPO DE EJECUCIÓN ALERTAMIENTOS : 🕓  " + str(tiempo_ejecucion_codigo) + " " + "minutos"+ """<br>
    </p>
    """ + html_table + """
  </body>
</html>
"""

# Guarda el código HTML en un archivo o haz lo que necesites con él

# Envía el correo electrónico con el DataFrame en el cuerpo
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





## correo de emergencia 


# skipped your comments for readability
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText

# me = "jose.gomezv@bancofinandina.com"
# my_password = r"{}".format(str(lineas[2].strip()))

# # you=["jose.gomezv@bancofinandina.com" , "jesus.alvear@bancofinandina.com","astrid.bermudez@bancofinandina.com","sandra.suarez@bancofinandina.com"]
# you=["jose.gomezv@bancofinandina.com" ,"sandra.suarez@bancofinandina.com","leidy.cadenas@bancofinandina.com","jesus.alvear@bancofinandina.com"]
# # you=["jose.gomezv@bancofinandina.com"]
# msg = MIMEMultipart('alternative')
# msg['Subject'] = "Actualización Alertamientos de FRAUDE"
# msg['From'] = me
# msg['To'] = ",".join(you)


# html = """\
# <html>
#   <head></head>
#   <body>
#     <p> 📊 📈 Buen Día, el presente correo contiene la última fecha de la que se tienen registros en los alertamientos de fraude 📊 📈<br>
#    </p>
#   </body>
# </html>
# """


# part2 = MIMEText(html, 'html')

# msg.attach(part2)

# # Send the message via gmail's regular server, over SSL - passwords are being sent, afterall
# s = smtplib.SMTP_SSL('smtp.gmail.com')
# # uncomment if interested in the actual smtp conversation
# # s.set_debuglevel(1)
# # do the smtp auth; sends ehlo if it hasn't been sent already
# s.login(me, my_password)

# s.sendmail(me,you, msg.as_string())
# s.quit()


