
#### nuevo alertamiento

#### NUEVO ALERTAMIENTO 

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
from sqlalchemy import create_engine
from multiprocessing import Pool


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



query_datos_tarjeta_credito='''SELECT 
 NumeroContrato as 'Número de cuenta'
,DocumentoCliente as 'Documento'
,NumeroTarjeta as 'Número de tarjeta'
,TipoBeneficiario as 'Tipo de beneficiario'
,NumeroTarjetaAnterior as 'Número de tarjeta anterior'
,DescripcionEstadoCuenta as 'Estado de la cuenta'
,DescripcionSituacionTarjeta as 'Situación de tarjeta'
,DescripcionBloqueo as 'Descripción del bloqueo'
,DescripcionMotivoBaja as 'Descripción de baja'
,FechaAltaTarjeta as 'Fecha de alta de la tarjeta'
,FechaBajaTarjeta as 'Fecha de baja de la tarjeta'
,DescripcionEstadoGeneralTarjeta as 'Descripción estado general tarjeta'
,CupoAprobadoTarjeta as 'Cupo aprobado de la tarjeta' 
,SaldoActualTarjeta as 'Saldo actual tarjeta'
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ProductoTDC]')
datos_tarjeta_credito_limpio=load_data(query_datos_tarjeta_credito,config_db_riesgo[1],model_logger=logger)


### escenario 1

datos_alerta = datos_tarjeta_credito_limpio[datos_tarjeta_credito_limpio['Descripción estado general tarjeta']!='Cancelada'].loc[:, ['Documento', 'Número de cuenta', 'Cupo aprobado de la tarjeta','Saldo actual tarjeta']]
datos_alerta.dtypes
datos_alerta['Cupo limite tarjeta']=datos_alerta['Cupo aprobado de la tarjeta']*1.1
datos_alerta['Balance'] = datos_alerta['Cupo limite tarjeta'] - datos_alerta['Saldo actual tarjeta']
alertamientos = datos_alerta[datos_alerta['Balance']<0]
alertamientos['Balance']=abs(alertamientos['Balance'])
alertamientos=alertamientos[['Documento', 'Número de cuenta','Cupo aprobado de la tarjeta','Cupo limite tarjeta','Balance']]


## escenario 2

datos_alerta = datos_tarjeta_credito_limpio[datos_tarjeta_credito_limpio['Descripción estado general tarjeta']!='Cancelada'].loc[:, ['Documento', 'Número de cuenta', 'Cupo aprobado de la tarjeta','Saldo actual tarjeta']]
datos_alerta['Cupo limite tarjeta']=datos_alerta['Cupo aprobado de la tarjeta']*1.1
datos_alerta['Balance'] = datos_alerta['Cupo limite tarjeta'] - datos_alerta['Saldo actual tarjeta']
esc2 = datos_alerta.groupby(['Documento'])['Cupo limite tarjeta','Saldo actual tarjeta'].sum().reset_index()
esc2['Balance']=esc2['Cupo limite tarjeta']-esc2['Saldo actual tarjeta']
alertamientos2 = esc2[esc2['Balance']<0]
alertamientos2['Balance']=abs(esc2['Balance'])




alertamientos2_limpio=alertamientos2[['Documento']]
alertamientos2_limpio['Evento']='nueva alerta'
alertamientos2_limpio=alertamientos2.drop_duplicates(subset=['Documento']).reset_index(drop=True)







alerta_cuenta_multiple_correo.to_sql('AlertaAhorroRegistroCorreoSimilitud',
                                      con=config_db_fraude[1],
                                      if_exists='replace',
                                      index=False,
                                      schema='dbo')










# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 09:54:50 2022

@author: josgom
"""

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
from sqlalchemy import create_engine
from multiprocessing import Pool

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



query_datos_demograficos='''SELECT 
 DocumentoCliente as 'Documento'
,NombreCliente as 'Nombre'
,FechaVinculacion as 'Fecha de vinculación'
,FechaNacimiento as 'Fecha de nacimiento'
,FechaUltimoMantenimiento as 'Fecha última actualización'
,Celular
,Correo
,MontoIngresos as 'Ingresos'
FROM {tabla_datos}
WHERE [FechaVinculacion]>'2022-12-26' '''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoNaturalDemografia]')
datos_demograficos_limpio=load_data(query_datos_demograficos,config_db_riesgo[1],model_logger=logger)


query_datos_productos='''SELECT 
DocumentoCliente as 'Documento'
      ,NumeroCuenta as 'Número de cuenta'
      ,FechaApertura as 'Fecha de apertura'
      ,TipoProducto as 'Linea de producto'
      ,LineaProducto
      ,DescripcionProducto as 'Descripción de producto'
      ,EstadoCuenta as 'Estado de la cuenta'
      ,DescripcionSucursal
      ,CiudadSucursal
      ,SaldoCapital as 'Saldo capital'
      ,FechaUltimoUso as 'Fecha de último uso'
      ,CodigoVendedor
      FROM {tabla_datos}
      WHERE [FechaApertura]>'2022-12-26' '''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoProductos]')
datos_productos_limpio=load_data(query_datos_productos,config_db_riesgo[1],model_logger=logger)


datos_demograficos_limpio['Documento']=datos_demograficos_limpio['Documento'].astype(str).str.strip()
datos_demograficos_limpio['Documento']=pd.to_numeric(datos_demograficos_limpio['Documento'],errors='coerce',downcast='integer')
datos_demograficos_limpio=datos_demograficos_limpio.dropna(subset=['Documento'])
datos_demograficos_limpio['Documento']=datos_demograficos_limpio['Documento'].astype(np.int64).astype(str)
datos_demograficos_limpio['Celular']=datos_demograficos_limpio['Celular'].astype(str).str.strip()
datos_demograficos_limpio['Celular']=pd.to_numeric(datos_demograficos_limpio['Celular'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_limpio['Fecha de vinculación']=pd.to_datetime(datos_demograficos_limpio['Fecha de vinculación'],format='%Y-%m-%d',errors='coerce')
datos_demograficos_limpio['Fecha de nacimiento']=pd.to_datetime(datos_demograficos_limpio['Fecha de nacimiento'],format='%Y-%m-%d',errors='coerce')
datos_demograficos_limpio['Fecha última actualización']=pd.to_datetime(datos_demograficos_limpio['Fecha última actualización'],format='%Y-%m-%d',errors='coerce')
datos_demograficos_limpio['Documento']=datos_demograficos_limpio['Documento'].str.strip()
datos_demograficos_limpio=datos_demograficos_limpio.sort_values(by=['Fecha de vinculación'])
datos_demograficos_limpio=datos_demograficos_limpio.drop_duplicates(subset=['Documento'],keep='last').reset_index(drop=True)



datos_productos_limpio=datos_productos_limpio.dropna(subset=['Documento','Número de cuenta'])
datos_productos_limpio[['Número de cuenta']]=datos_productos_limpio[['Número de cuenta']].astype(np.int64).astype(str)
datos_productos_limpio['Fecha de apertura']=pd.to_datetime(datos_productos_limpio['Fecha de apertura'])
datos_productos_limpio['Descripción de producto']=datos_productos_limpio['Descripción de producto'].str.strip()
datos_productos_limpio['Documento']=datos_productos_limpio['Documento'].astype(str).str.strip()
datos_productos_limpio=datos_productos_limpio.sort_values(by=['Documento','Número de cuenta','Fecha de apertura']).reset_index(drop=True)


datos_producto_cliente=pd.merge(datos_productos_limpio,datos_demograficos_limpio,on='Documento',how='inner')


# Flexi

datos_flexi_cliente=datos_producto_cliente.loc[datos_producto_cliente['Descripción de producto']=='FlexiDigital']

# Flexi activa

datos_flexi_activa_cliente=datos_flexi_cliente.loc[datos_flexi_cliente['Estado de la cuenta']=='Activa']

# TDC

datos_tdc_cliente=datos_producto_cliente.loc[datos_producto_cliente['Linea de producto']=='TDC']

datos_cantidad_cuentas_asociado_celular=pd.DataFrame(datos_flexi_activa_cliente.groupby(by=['Celular']).size(),columns=['Cuentas asociadas al celular']).reset_index()
datos_cuentas_ceular=pd.merge(datos_flexi_activa_cliente,datos_cantidad_cuentas_asociado_celular,on=['Celular'])

alerta_cuenta_multiple_celular=datos_cuentas_ceular[datos_cuentas_ceular['Cuentas asociadas al celular']>1]




#####################


datos_cantidad_cuentas_asociado_correo=pd.DataFrame(datos_flexi_activa_cliente.drop_duplicates().groupby(by=['Correo']).size(),columns=['Cuentas asociadas al correo']).reset_index()
datos_cuentas_correo=pd.merge(datos_flexi_activa_cliente,datos_cantidad_cuentas_asociado_correo,on=['Correo'])

alerta_cuenta_multiple_correo=datos_cuentas_correo[datos_cuentas_correo['Cuentas asociadas al correo']>1]




#### codigo bacup 

### fecha actual 
from datetime import datetime
from datetime import timedelta
fecha_actual=datetime.now()


## temporizador 
import time 
startTime0 = time.time()
startTime = time.time()

#-------------------------------------------------------------------------------------------------#
#                             ACTUALIZACIÓN ALERTAMIENTOS DE FRAUDE                               #
#-------------------------------------------------------------------------------------------------#

# librerías necesarias

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
from sqlalchemy import create_engine
from multiprocessing import Pool

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
 DocumentoCliente as 'Documento'
,NombreCliente as 'Nombre'
,FechaVinculacion as 'Fecha de vinculación'
,FechaNacimiento as 'Fecha de nacimiento'
,FechaUltimoMantenimiento as 'Fecha última actualización'
,Celular
,Correo
,MontoIngresos as 'Ingresos'
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoNaturalDemografia]')
datos_demograficos_limpio=load_data(query_datos_demograficos,config_db_riesgo[1],model_logger=logger)



# Carga de datos del cliente core # 2

query_datos_demograficos_core='''SELECT 
  IdAS400Cliente as 'Id AS400'
 ,DocumentoCliente as 'Documento'
 ,NombreCliente as 'Nombre'
 ,FechaVinculacion as 'Fecha de vinculación'
 ,FechaNacimiento as 'Fecha de nacimiento'
 ,FechaUltimoMantenimiento as 'Fecha última actualización'
 ,CiudadActual as 'Ciudad actual'
 ,CelularCliente as 'Celular'
 ,CorreoCliente as 'Correo'
 ,CodigoActividadEconomica as 'Código CIIU'
 ,DescripcionActividadEconomica as 'Descripción CIIU'
 ,IngresosEnMiles as'Ingresos'
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[DemografiaNaturalCore]')
datos_demograficos_core_limpio=load_data(query_datos_demograficos_core,config_db_riesgo[1],model_logger=logger)


# Carga de datos del cliente AGIL # 3

query_datos_demograficos_agil='''SELECT 
NumeroDocumento as 'Documento'
,TipoDocumento as 'Tipo de documento'
,NombreCompleto as'Nombre'
,FechaNacimiento as 'Fecha de nacimiento'
,FechaSolicitud as 'Fecha de solicitud'
,CiudadResidencia as 'Ciudad actual'
,Celular
,Correo
,CodigoCIIU as 'Código CIIU'
,DescripcionActividadCIIU as 'Descripción CIIU'
,MontoIngresos as 'Ingresos'
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[DemografiaNaturalAgil]')
datos_demograficos_agil_limpio=load_data(query_datos_demograficos_agil,config_db_riesgo[1],model_logger=logger)



# log portal # 4


query_datos_log_portal='''SELECT 
IP 
,Usuario as 'Documento'
,Fecha as 'Fecha de evento'
,SessionID as 'Sesión'
,Descripcion as 'Descripción'
,Metodo as 'Método' 
,Mensaje
FROM {tabla_datos} '''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[LogTransaccional].[dbo].[LogTransaccionalAlertaHistorico]')
datos_log_portal_limpio=load_data(query_datos_log_portal,config_db_riesgo[1],model_logger=logger)


# Carga de datos del cliente LP # 5

query_datos_demograficos_lp='''SELECT 
NumeroDocumento as 'Documento'
,TipoDocumento as 'Tipo de documento'
,Nombres as 'Nombre'
,FechaNacimiento as 'Fecha de nacimiento'
,FechaSolicitud as 'Fecha de solicitud'
,CiudadResidencia as 'Ciudad actual'
,Celular as 'Celular'
,Correo as 'Correo'
,CodigoCIIU as 'Código CIIU'
,DescripcionCIIU as 'Descripción CIIU'
,MontoIngresos as 'Ingresos'
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[DemografiaNaturalLP]')
datos_demograficos_lp_limpio=load_data(query_datos_demograficos_lp,config_db_riesgo[1],model_logger=logger)



# Carga datos cuentas creadas ahorro digital cliente # 6

query_datos_log_creacion='''SELECT 
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
FROM {tabla_datos}
WHERE  [FechaRegistro] >=  DATEADD(dd,DATEDIFF(dd,3,GETDATE()),0) and [FechaRegistro] <  DATEADD(dd,DATEDIFF(dd,0,GETDATE()),0) '''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[LogTransaccional].[dbo].[RegistroAhorroCreado]')
datos_log_creacion_limpio=load_data(query_datos_log_creacion,config_db_riesgo[1],model_logger=logger)


# Carga datos registro creacion digital # 7

query_datos_registro_creacion='''SELECT 
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
FROM {tabla_datos} '''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[LogTransaccional].[dbo].[LogCreacionCliente]')
datos_registro_creacion_limpio=load_data(query_datos_registro_creacion,config_db_riesgo[1],model_logger=logger)


# Carga datos de enrolamiento # 8


query_datos_log_enrolamiento='''SELECT 
TipoDocumento as 'Tipo de documento'
,NumDocumento as 'Documento'
,Estado as 'Estado'
,Nombre as 'Nombre'
,FechaEnrolamiento as 'Fecha de enrolamiento' 
FROM {tabla_datos} '''.format(tabla_datos='[FABOGSQLCLU].[Usuarios].[dbo].[Usuarios]')
datos_log_enrolamiento_limpio=load_data(query_datos_log_enrolamiento,config_db_riesgo[1],model_logger=logger)


# Carga datos de evidente # 9

query_datos_log_evidente='''SELECT 
NumeroDocumento as 'Documento'
,Response as 'Respuesta'
,FechaRegistro as 'Fecha de registro'
,TipoConsultaEvidente as 'Tipo de consulta'
FROM {tabla_datos} 
WHERE  [FechaRegistro] >=  DATEADD(dd,DATEDIFF(dd,3,GETDATE()),0) and [FechaRegistro] <  DATEADD(dd,DATEDIFF(dd,0,GETDATE()),0) '''.format(tabla_datos='[FABOGSQLCLU].[Usuarios].[dbo].[HistoricoEvidente]')
datos_log_evidente_limpio=load_data(query_datos_log_evidente,config_db_riesgo[1],model_logger=logger)


# Carga datos dispositivo seguro # 10


query_dispositivo_vinculado='''SELECT *
FROM {tabla_datos}
WHERE [ChangeDate] >= DATEADD(dd,DATEDIFF(dd,3,GETDATE()),0) AND [ChangeDate] <  DATEADD(dd,DATEDIFF(dd,0,GETDATE()),0) '''.format(tabla_datos='[FABOGREPORTS].[AppFinandina].[Core].[SecureDevice]')
datos_disposito_vinculado=load_data(query_dispositivo_vinculado,config_db_riesgo[1],model_logger=logger)


# Datos de mensajería # 11


query_mensajeria='''SELECT 
Documento
,Nombre
,Fecha as 'Fecha de evento'
,Mensaje
,Estrategia as 'Evento'
,IdEstado as 'Estado'
FROM {tabla_datos}
WHERE (Estrategia='ActualizacionDatos' OR
                           Estrategia='SMS418' OR
                           Estrategia='SMS419') AND
                           IdTipoMensaje=2 '''.format(tabla_datos='[FABOGSQLCLU].[Mensajeria].[dbo].[Envio]')
datos_mensajeria_limpio=load_data(query_mensajeria,config_db_riesgo[1],model_logger=logger)


# Carga de datos de productos de los clientes consolidados # 12

query_datos_productos='''SELECT 
DocumentoCliente as 'Documento'
      ,NumeroCuenta as 'Número de cuenta'
      ,FechaApertura as 'Fecha de apertura'
      ,TipoProducto as 'Linea de producto'
      ,LineaProducto
      ,DescripcionProducto as 'Descripción de producto'
      ,EstadoCuenta as 'Estado de la cuenta'
      ,DescripcionSucursal
      ,CiudadSucursal
      ,SaldoCapital as 'Saldo capital'
      ,FechaUltimoUso as 'Fecha de último uso'
      ,CodigoVendedor
      FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoProductos]')
datos_productos_limpio=load_data(query_datos_productos,config_db_riesgo[1],model_logger=logger)


### ajuste alertamiento intentos evidente 

query_datos_productos2='''SELECT 
DocumentoCliente as 'Documento'
      ,NumeroCuenta as 'Número de cuenta'
      ,FechaApertura as 'Fecha de apertura'
      ,TipoProducto as 'Linea de producto'
      ,LineaProducto
      ,DescripcionProducto as 'Descripción de producto'
      ,EstadoCuenta as 'Estado de la cuenta'
      ,DescripcionSucursal
      ,CiudadSucursal
      ,SaldoCapital as 'Saldo capital'
      ,FechaUltimoUso as 'Fecha de último uso'
      ,CodigoVendedor
      FROM {tabla_datos}
      WHERE  [FechaApertura] >=  DATEADD(dd,DATEDIFF(dd,3,GETDATE()),0) and [FechaApertura] <  DATEADD(dd,DATEDIFF(dd,0,GETDATE()),0) and [DescripcionProducto] = 'FlexiDigital'  '''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoProductos]')
datos_productos_limpio2=load_data(query_datos_productos2,config_db_riesgo[1],model_logger=logger)





# Carga de datos transaccionales consolidados # 13 

query_datos_transacciones='''SELECT 
 [DocumentoCliente] as 'Documento'
      ,[NumeroCuenta] as 'Número de cuenta'
      ,[FechaTransaccion] as 'Fecha de la transacción'
      ,[CodigoTransaccional1] as  'Código de transacción 1'
      ,[CodigoTransaccional2] as 'Código de transacción 2'
      ,[TipoTransaccion] as 'Codigo tipo transacción'
      ,[MontoTransacción] as 'Monto de la transacción'
      ,[DescripcionTransaccion] as 'Descripción de la transacción'
      ,[CaracterTransaccion]
      ,[LugarTransaccion] as 'Lugar de la transacción'
      ,[TipoProducto] as 'Linea de producto'
      ,[DescripcionProducto] as 'Descripción del producto'
      FROM {tabla_datos}
      WHERE FechaTransaccion BETWEEN '2022-01-01' AND GETDATE()'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoTransacciones]')
datos_transacciones_limpio=load_data(query_datos_transacciones,config_db_riesgo[1],model_logger=logger)



# Carga de datos de tarjeta de débito # 14

query_datos_tarjeta_debito='''SELECT 
Documento as 'Documento'
,NumeroCuenta as 'Número de cuenta'
,NumeroTarjeta as 'Número de tarjeta'
,CondicionTarjeta as 'Condición de tarjeta'
,EstadoTarjeta as 'Estado de tarjeta'
,TipoTarjeta as 'Tipo de tarjeta'
,FechaCondicion as 'Fecha de condición'
,FechaEstado as 'Fecha de estado'
,FechaExpiracion as 'Fecha de expiración' 
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[TarjetaDebito]')
datos_tarjeta_debito_limpio=load_data(query_datos_tarjeta_debito,config_db_riesgo[1],model_logger=logger)


# Carga de datos de tarjeta de crédito # 15

query_datos_tarjeta_credito='''SELECT 
 NumeroContrato as 'Número de cuenta'
,DocumentoCliente as 'Documento'
,NumeroTarjeta as 'Número de tarjeta'
,TipoBeneficiario as 'Tipo de beneficiario'
,NumeroTarjetaAnterior as 'Número de tarjeta anterior'
,DescripcionEstadoCuenta as 'Estado de la cuenta'
,DescripcionSituacionTarjeta as 'Situación de tarjeta'
,DescripcionBloqueo as 'Descripción del bloqueo'
,DescripcionMotivoBaja as 'Descripción de baja'
,FechaAltaTarjeta as 'Fecha de alta de la tarjeta'
,FechaBajaTarjeta as 'Fecha de baja de la tarjeta'
,DescripcionEstadoGeneralTarjeta as 'Descripción estado general tarjeta'
,CupoAprobadoTarjeta as 'Cupo aprobado de la tarjeta' 
,LimiteCupoContrato as 'Cupo limite Banco'
,SaldoActualTarjeta as 'Saldo actual tarjeta'
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ProductoTDC]')
datos_tarjeta_credito_limpio=load_data(query_datos_tarjeta_credito,config_db_riesgo[1],model_logger=logger)


# Base para exportar datos

# Configuración de la base


print('Inicio limpieza de datos')


# LIMPIEZA DE DATOS 

datos_demograficos_limpio['Documento']=datos_demograficos_limpio['Documento'].astype(str).str.strip()
datos_demograficos_limpio['Documento']=pd.to_numeric(datos_demograficos_limpio['Documento'],errors='coerce',downcast='integer')
datos_demograficos_limpio=datos_demograficos_limpio.dropna(subset=['Documento'])
datos_demograficos_limpio['Documento']=datos_demograficos_limpio['Documento'].astype(np.int64).astype(str)
datos_demograficos_limpio['Celular']=datos_demograficos_limpio['Celular'].astype(str).str.strip()
datos_demograficos_limpio['Celular']=pd.to_numeric(datos_demograficos_limpio['Celular'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_limpio['Fecha de vinculación']=pd.to_datetime(datos_demograficos_limpio['Fecha de vinculación'],format='%Y-%m-%d',errors='coerce')
datos_demograficos_limpio['Fecha de nacimiento']=pd.to_datetime(datos_demograficos_limpio['Fecha de nacimiento'],format='%Y-%m-%d',errors='coerce')
datos_demograficos_limpio['Fecha última actualización']=pd.to_datetime(datos_demograficos_limpio['Fecha última actualización'],format='%Y-%m-%d',errors='coerce')
datos_demograficos_limpio['Documento']=datos_demograficos_limpio['Documento'].str.strip()
datos_demograficos_limpio=datos_demograficos_limpio.sort_values(by=['Fecha de vinculación'])
datos_demograficos_limpio=datos_demograficos_limpio.drop_duplicates(subset=['Documento'],keep='last').reset_index(drop=True)



datos_demograficos_core_limpio['Documento']=datos_demograficos_core_limpio['Documento'].astype(str).str.strip()
datos_demograficos_core_limpio['Documento']=pd.to_numeric(datos_demograficos_core_limpio['Documento'],errors='coerce',downcast='integer')
datos_demograficos_core_limpio=datos_demograficos_core_limpio.dropna(subset=['Documento'])
datos_demograficos_core_limpio['Documento']=datos_demograficos_core_limpio['Documento'].astype(np.int64).astype(str)
datos_demograficos_core_limpio['Celular']=datos_demograficos_core_limpio['Celular'].astype(str).str.strip()
datos_demograficos_core_limpio['Celular']=pd.to_numeric(datos_demograficos_core_limpio['Celular'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_core_limpio['Código CIIU']=datos_demograficos_core_limpio['Código CIIU'].astype(str).str.strip()
datos_demograficos_core_limpio['Código CIIU']=pd.to_numeric(datos_demograficos_core_limpio['Código CIIU'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_core_limpio['Fecha de vinculación']=pd.to_datetime(pd.to_datetime(datos_demograficos_core_limpio['Fecha de vinculación'].astype(np.int64).astype(str).str.zfill(6),format='%d%m%y',errors='coerce').dt.strftime('%Y-%m-%d'))
datos_demograficos_core_limpio['Fecha de nacimiento']=pd.to_datetime(pd.to_datetime(datos_demograficos_core_limpio['Fecha de nacimiento'].astype(np.int64),format='%Y%m%d',errors='coerce').dt.strftime('%Y-%m-%d'))
datos_demograficos_core_limpio['Fecha última actualización']=pd.to_datetime(pd.to_datetime(datos_demograficos_core_limpio['Fecha última actualización'].astype(np.int64),format='%Y-%m-%d',errors='coerce').dt.strftime('%Y-%m-%d'))
datos_demograficos_core_limpio['Documento']=datos_demograficos_core_limpio['Documento'].str.strip()
datos_demograficos_core_limpio['Descripción CIIU']=datos_demograficos_core_limpio['Descripción CIIU'].str.strip()
datos_demograficos_core_limpio['Ingresos']=datos_demograficos_core_limpio['Ingresos']*1000
datos_demograficos_core_limpio=datos_demograficos_core_limpio.sort_values(by=['Fecha de vinculación'])
datos_demograficos_core_limpio=datos_demograficos_core_limpio.drop_duplicates(subset=['Documento'],keep='last').reset_index(drop=True)



datos_demograficos_agil_limpio['Documento']=datos_demograficos_agil_limpio['Documento'].astype(str).str.strip()
datos_demograficos_agil_limpio['Documento']=pd.to_numeric(datos_demograficos_agil_limpio['Documento'],errors='coerce')
datos_demograficos_agil_limpio=datos_demograficos_agil_limpio.dropna(subset=['Documento'])
datos_demograficos_agil_limpio['Documento']=datos_demograficos_agil_limpio['Documento'].astype(np.int64).astype(str)
datos_demograficos_agil_limpio['Celular']=datos_demograficos_agil_limpio['Celular'].astype(str).str.strip()
datos_demograficos_agil_limpio['Celular']=pd.to_numeric(datos_demograficos_agil_limpio['Celular'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_agil_limpio['Código CIIU']=datos_demograficos_agil_limpio['Código CIIU'].astype(str).str.strip()
datos_demograficos_agil_limpio['Código CIIU']=pd.to_numeric(datos_demograficos_agil_limpio['Código CIIU'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_agil_limpio['Fecha de nacimiento']=pd.to_datetime(pd.to_datetime(datos_demograficos_agil_limpio['Fecha de nacimiento'],format='%Y-%m-%d',errors='coerce').dt.strftime('%Y-%m-%d'))
datos_demograficos_agil_limpio['Fecha de solicitud']=pd.to_datetime(pd.to_datetime(datos_demograficos_agil_limpio['Fecha de solicitud'],errors='coerce').dt.strftime('%Y-%m-%d'))
datos_demograficos_agil_limpio['Ingresos']=datos_demograficos_agil_limpio['Ingresos'].astype(float)
datos_demograficos_agil_limpio=datos_demograficos_agil_limpio.sort_values(by=['Documento','Fecha de solicitud']).reset_index(drop=True)




datos_demograficos_lp_limpio['Documento']=datos_demograficos_lp_limpio['Documento'].astype(str).str.strip()
datos_demograficos_lp_limpio['Documento']=pd.to_numeric(datos_demograficos_lp_limpio['Documento'],errors='coerce',downcast='integer')
datos_demograficos_lp_limpio=datos_demograficos_lp_limpio.dropna(subset=['Documento'])
datos_demograficos_lp_limpio['Documento']=datos_demograficos_lp_limpio['Documento'].astype(np.int64).astype(str)
datos_demograficos_lp_limpio['Celular']=datos_demograficos_lp_limpio['Celular'].astype(str).str.strip()
datos_demograficos_lp_limpio['Celular']=pd.to_numeric(datos_demograficos_lp_limpio['Celular'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_lp_limpio['Código CIIU']=datos_demograficos_lp_limpio['Código CIIU'].astype(str).str.strip()
datos_demograficos_lp_limpio['Código CIIU']=pd.to_numeric(datos_demograficos_lp_limpio['Código CIIU'],errors='coerce',downcast='integer').astype(str)
datos_demograficos_lp_limpio['Fecha de nacimiento']=pd.to_datetime(pd.to_datetime(datos_demograficos_lp_limpio['Fecha de nacimiento'],format='%d%m%Y',errors='coerce').dt.strftime('%Y-%m-%d'))
datos_demograficos_lp_limpio['Fecha de solicitud']=pd.to_datetime(pd.to_datetime(datos_demograficos_lp_limpio['Fecha de solicitud'],format='%d%m%Y',errors='coerce').dt.strftime('%Y-%m-%d'))
datos_demograficos_lp_limpio['Ingresos']=pd.to_numeric(datos_demograficos_lp_limpio['Ingresos'],errors='coerce')
datos_demograficos_lp_limpio=datos_demograficos_lp_limpio.sort_values(by=['Documento','Fecha de solicitud']).reset_index(drop=True)




datos_log_portal_limpio['Documento']=datos_log_portal_limpio['Documento'].astype(str).str.strip()
datos_log_portal_limpio['Documento']=pd.to_numeric(datos_log_portal_limpio['Documento'],errors='coerce',downcast='integer')
datos_log_portal_limpio=datos_log_portal_limpio.dropna(subset=['Documento'])
datos_log_portal_limpio['Documento']=datos_log_portal_limpio['Documento'].astype(np.int64).astype(str)
datos_log_portal_limpio=datos_log_portal_limpio.dropna()
datos_log_portal_limpio['Fecha de evento']=pd.to_datetime(datos_log_portal_limpio['Fecha de evento'])
datos_log_portal_limpio['Descripción']=datos_log_portal_limpio['Descripción'].str.strip()
datos_log_portal_limpio=datos_log_portal_limpio.sort_values(by=['Documento','Fecha de evento']).reset_index(drop=True)




datos_log_creacion_limpio['Documento']=datos_log_creacion_limpio['Documento'].astype(str).str.strip()
datos_log_creacion_limpio['Documento']=pd.to_numeric(datos_log_creacion_limpio['Documento'],errors='coerce',downcast='integer')
datos_log_creacion_limpio=datos_log_creacion_limpio.dropna(subset=['Documento'])
datos_log_creacion_limpio['Documento']=datos_log_creacion_limpio['Documento'].astype(np.int64).astype(str)
datos_log_creacion_limpio['Fecha de registro']=pd.to_datetime(datos_log_creacion_limpio['Fecha de registro'])
datos_log_creacion_limpio=datos_log_creacion_limpio.sort_values(by=['Fecha de registro']).reset_index(drop=True)




datos_registro_creacion_limpio['Documento']=datos_registro_creacion_limpio['Documento'].astype(str).str.strip()
datos_registro_creacion_limpio['Documento']=pd.to_numeric(datos_registro_creacion_limpio['Documento'],errors='coerce',downcast='integer')
datos_registro_creacion_limpio=datos_registro_creacion_limpio.dropna(subset=['Documento'])
datos_registro_creacion_limpio['Documento']=datos_registro_creacion_limpio['Documento'].astype(np.int64).astype(str)
datos_registro_creacion_limpio['Fecha inicial']=pd.to_datetime(datos_registro_creacion_limpio['Fecha inicial'])
datos_registro_creacion_limpio['Fecha final']=pd.to_datetime(datos_registro_creacion_limpio['Fecha final'])
datos_registro_creacion_limpio=datos_registro_creacion_limpio.sort_values(by=['Documento','Fecha inicial']).reset_index(drop=True)




datos_log_enrolamiento_limpio['Documento']=datos_log_enrolamiento_limpio['Documento'].astype(str).str.strip()
datos_log_enrolamiento_limpio['Documento']=pd.to_numeric(datos_log_enrolamiento_limpio['Documento'],errors='coerce',downcast='integer')
datos_log_enrolamiento_limpio=datos_log_enrolamiento_limpio.dropna(subset=['Documento'])
datos_log_enrolamiento_limpio['Documento']=datos_log_enrolamiento_limpio['Documento'].astype(np.int64).astype(str)
datos_log_enrolamiento_limpio['Fecha de enrolamiento']=pd.to_datetime(datos_log_enrolamiento_limpio['Fecha de enrolamiento'])
datos_log_enrolamiento_limpio=datos_log_enrolamiento_limpio.sort_values(by=['Fecha de enrolamiento']).reset_index(drop=True)


 

datos_log_evidente_limpio['Documento']=datos_log_evidente_limpio['Documento'].astype(str).str.strip()
datos_log_evidente_limpio['Documento']=pd.to_numeric(datos_log_evidente_limpio['Documento'],errors='coerce',downcast='integer')
datos_log_evidente_limpio=datos_log_evidente_limpio.dropna(subset=['Documento'])
datos_log_evidente_limpio['Documento']=datos_log_evidente_limpio['Documento'].astype(np.int64).astype(str)
datos_log_evidente_limpio['Fecha de registro']=pd.to_datetime(datos_log_evidente_limpio['Fecha de registro'])
datos_log_evidente_limpio['Respuesta']=datos_log_evidente_limpio['Respuesta'].apply(lambda x: dict(subString.split(':') for subString in x[1:-1].split(',')))
datos_log_evidente_limpio=datos_log_evidente_limpio.sort_values(by=['Fecha de registro']).reset_index(drop=True)





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



datos_mensajeria_limpio['Documento']=datos_mensajeria_limpio['Documento'].astype(str).str.strip()
datos_mensajeria_limpio['Documento']=pd.to_numeric(datos_mensajeria_limpio['Documento'],errors='coerce',downcast='integer')
datos_mensajeria_limpio=datos_mensajeria_limpio.dropna(subset=['Documento'])
datos_mensajeria_limpio['Documento']=datos_mensajeria_limpio['Documento'].astype(np.int64).astype(str)
datos_mensajeria_limpio['Fecha de evento']=pd.to_datetime(datos_mensajeria_limpio['Fecha de evento'])
datos_mensajeria_limpio['Evento']=datos_mensajeria_limpio['Evento'].map({'ActualizacionDatos':'Actualización exitosa','SMS418':'Correo actualizado','SMS419':'Celular actualizado'})
datos_mensajeria_limpio['Estado']=datos_mensajeria_limpio['Estado'].map({1:'En proceso',2:'Enviado',3:'Error'})
datos_mensajeria_limpio=datos_mensajeria_limpio.sort_values(by=['Documento','Fecha de evento']).reset_index(drop=True)
datos_mensajeria_limpio['Fecha de evento']=pd.to_datetime(datos_mensajeria_limpio['Fecha de evento'].dt.strftime('%Y-%m-%d %H:%M:%s'))




datos_productos_limpio=datos_productos_limpio.dropna(subset=['Documento','Número de cuenta'])
datos_productos_limpio[['Número de cuenta']]=datos_productos_limpio[['Número de cuenta']].astype(np.int64).astype(str)
datos_productos_limpio['Fecha de apertura']=pd.to_datetime(datos_productos_limpio['Fecha de apertura'])
datos_productos_limpio['Descripción de producto']=datos_productos_limpio['Descripción de producto'].str.strip()
datos_productos_limpio['Documento']=datos_productos_limpio['Documento'].astype(str).str.strip()
datos_productos_limpio=datos_productos_limpio.sort_values(by=['Documento','Número de cuenta','Fecha de apertura']).reset_index(drop=True)

### Productos 2 

datos_productos_limpio2=datos_productos_limpio2.dropna(subset=['Documento','Número de cuenta'])
datos_productos_limpio2[['Número de cuenta']]=datos_productos_limpio2[['Número de cuenta']].astype(np.int64).astype(str)
datos_productos_limpio2['Fecha de apertura']=pd.to_datetime(datos_productos_limpio2['Fecha de apertura'])
datos_productos_limpio2['Descripción de producto']=datos_productos_limpio2['Descripción de producto'].str.strip()
datos_productos_limpio2['Documento']=datos_productos_limpio2['Documento'].astype(str).str.strip()
datos_productos_limpio2=datos_productos_limpio2.sort_values(by=['Documento','Número de cuenta','Fecha de apertura']).reset_index(drop=True)



datos_transacciones_limpio=datos_transacciones_limpio.dropna(subset=['Documento','Número de cuenta'])
datos_transacciones_limpio[['Documento','Número de cuenta']]=datos_transacciones_limpio[['Documento','Número de cuenta']].astype(np.int64).astype(str)
datos_transacciones_limpio['Fecha de la transacción']=pd.to_datetime(datos_transacciones_limpio['Fecha de la transacción'])
datos_transacciones_limpio['Descripción del producto']=datos_transacciones_limpio['Descripción del producto'].str.strip()
datos_transacciones_limpio=datos_transacciones_limpio.sort_values(by=['Documento','Número de cuenta','Fecha de la transacción']).reset_index(drop=True)



datos_tarjeta_debito_limpio=datos_tarjeta_debito_limpio.dropna(subset=['Documento','Número de cuenta'])
datos_tarjeta_debito_limpio[['Documento','Número de cuenta']]=datos_tarjeta_debito_limpio[['Documento','Número de cuenta']].astype(np.int64).astype(str)
datos_tarjeta_debito_limpio['Fecha de condición']=pd.to_datetime(pd.to_datetime(datos_tarjeta_debito_limpio['Fecha de condición'].astype(str),format='%Y%m%d').dt.strftime('%Y-%m-%d'))
datos_tarjeta_debito_limpio=datos_tarjeta_debito_limpio.sort_values(by=['Documento','Número de cuenta','Fecha de condición']).reset_index(drop=True)



datos_tarjeta_credito_limpio=datos_tarjeta_credito_limpio.dropna(subset=['Documento','Número de cuenta'])
datos_tarjeta_credito_limpio[['Número de cuenta']]=datos_tarjeta_credito_limpio[['Número de cuenta']].astype(np.int64).astype(str)
datos_tarjeta_credito_limpio['Documento']=datos_tarjeta_credito_limpio['Documento'].astype(str).str.strip()
datos_tarjeta_credito_limpio['Fecha de alta de la tarjeta']=pd.to_datetime(datos_tarjeta_credito_limpio['Fecha de alta de la tarjeta'],format='%Y-%m-%d',errors='coerce')
datos_tarjeta_credito_limpio['Fecha de baja de la tarjeta']=pd.to_datetime(datos_tarjeta_credito_limpio['Fecha de baja de la tarjeta'],format='%Y-%m-%d',errors='coerce')
datos_tarjeta_credito_limpio['Número de tarjeta']=datos_tarjeta_credito_limpio['Número de tarjeta'].astype(str).str.strip()
datos_tarjeta_credito_limpio['Número de tarjeta anterior']=datos_tarjeta_credito_limpio['Número de tarjeta anterior'].str.strip()
datos_tarjeta_credito_limpio=datos_tarjeta_credito_limpio.sort_values(by=['Documento','Número de cuenta','Fecha de alta de la tarjeta']).reset_index(drop=True)


print('Fin limpieza de datos')

# Datos generales

# Producto y demográficos

datos_producto_cliente=pd.merge(datos_productos_limpio,datos_demograficos_limpio,on='Documento',how='inner')

datos_producto_cliente2=pd.merge(datos_productos_limpio2,datos_demograficos_limpio,on='Documento',how='inner')

# Flexi

datos_flexi_cliente=datos_producto_cliente.loc[datos_producto_cliente['Descripción de producto']=='FlexiDigital']
datos_flexi_cliente2=datos_producto_cliente2

# Flexi activa

datos_flexi_activa_cliente=datos_flexi_cliente.loc[datos_flexi_cliente['Estado de la cuenta']=='Activa']
datos_flexi_activa_cliente2=datos_flexi_cliente2
#.loc[datos_flexi_cliente2['Estado de la cuenta']=='Activa']

# TDC

datos_tdc_cliente=datos_producto_cliente.loc[datos_producto_cliente['Linea de producto']=='TDC']

# TDC Activa

datos_tdc_activa_cliente=datos_producto_cliente.loc[datos_producto_cliente['Estado de la cuenta']=='Activa']
datos_tdc_activa_tarjeta=datos_tarjeta_credito_limpio.loc[datos_tarjeta_credito_limpio['Descripción estado general tarjeta']=='Activa']


# Transacciones y demográficos

datos_transacciones_cliente=pd.merge(datos_transacciones_limpio,datos_demograficos_limpio,on='Documento',how='inner')


# FlexiDigital

datos_transacciones_flexi_cliente=datos_transacciones_cliente.loc[datos_transacciones_cliente['Descripción del producto']=='FlexiDigital']


# Operaciones digitales 

# Login

datos_portal_login=datos_log_portal_limpio[datos_log_portal_limpio['Método']=='Login']


# Olvido de contraseñas

datos_portal_olvido_contraseña=datos_log_portal_limpio[(datos_log_portal_limpio['Descripción']=='ForgotPassword')]


# Actualización datos contacto

datos_mensajeria_limpio['Delta transaccional']=datos_mensajeria_limpio.groupby(by=['Documento'])['Fecha de evento'].diff()
datos_mensajeria_limpio_depurado=datos_mensajeria_limpio.loc[(datos_mensajeria_limpio['Delta transaccional']>pd.Timedelta(minutes=1))|(datos_mensajeria_limpio['Delta transaccional'].isnull())]


datos_actualizan_celular=datos_mensajeria_limpio_depurado.loc[datos_mensajeria_limpio_depurado['Evento']=='Celular actualizado']
datos_actualizan_correo=datos_mensajeria_limpio_depurado.loc[datos_mensajeria_limpio_depurado['Evento']=='Correo actualizado']

datos_contacto_actualizan=pd.concat([datos_actualizan_celular,datos_actualizan_correo]).sort_values(by=['Documento','Fecha de evento']).reset_index(drop=True)


# Producto y acciones en el portal

datos_producto_actualizacion=pd.merge(datos_productos_limpio,datos_contacto_actualizan,how='inner',on='Documento')


# Algoritmos de fraude

# Cuentas Flexi con múltiples números de celular

datos_cantidad_cuentas_asociado_celular=pd.DataFrame(datos_flexi_activa_cliente2.groupby(by=['Celular']).size(),columns=['Cuentas asociadas al celular']).reset_index()
datos_cuentas_ceular=pd.merge(datos_flexi_activa_cliente2,datos_cantidad_cuentas_asociado_celular,on=['Celular'])

alerta_cuenta_multiple_celular=datos_cuentas_ceular[datos_cuentas_ceular['Cuentas asociadas al celular']>1]

alerta_cuenta_multiple_celular.to_sql('AlertaAhorroRegistroCelularSimilitud',
                                       con=config_db_fraude[1],
                                       if_exists='replace',
                                       index=False,
                                       schema='dbo')

print('1. AlertaAhorroRegistroCelularSimilitud Actualizada')

# Cuentas Flexi con múltiples correos electrónicos

datos_cantidad_cuentas_asociado_correo=pd.DataFrame(datos_flexi_activa_cliente2.drop_duplicates().groupby(by=['Correo']).size(),columns=['Cuentas asociadas al correo']).reset_index()
datos_cuentas_correo=pd.merge(datos_flexi_activa_cliente,datos_cantidad_cuentas_asociado_correo,on=['Correo'])

alerta_cuenta_multiple_correo=datos_cuentas_correo[datos_cuentas_correo['Cuentas asociadas al correo']>1]


alerta_cuenta_multiple_correo.to_sql('AlertaAhorroRegistroCorreoSimilitud',
                                      con=config_db_fraude[1],
                                      if_exists='replace',
                                      index=False,
                                      schema='dbo')

print('2. AlertaAhorroRegistroCorreoSimilitud Actualizada')

# Clientes con tarjeta de titular y amparado


datos_titularidad_tdc=datos_tdc_activa_tarjeta[['Documento','Número de cuenta','Número de tarjeta','Tipo de beneficiario','Estado de la cuenta','Situación de tarjeta','Descripción del bloqueo','Descripción de baja','Fecha de alta de la tarjeta','Fecha de baja de la tarjeta']]


clientes_doble_titularidad=pd.DataFrame(datos_titularidad_tdc.groupby(by=['Documento']).apply(lambda x: ((x['Tipo de beneficiario']=='TI').any())and((x['Tipo de beneficiario']=='BE').any())),columns=['Alerta doble titularidad']).reset_index()

alerta_doble_titularidad=pd.merge(datos_titularidad_tdc,clientes_doble_titularidad.loc[clientes_doble_titularidad['Alerta doble titularidad']==True],on=['Documento'],how='inner')


alerta_doble_titularidad.to_sql('AlertaTDCCondicionSimilitud',
                                 con=config_db_fraude[1],
                                 if_exists='replace',
                                 index=False,
                                 schema='dbo')

print('3. AlertaTDCCondicionSimilitud Actualizada')

# Clientes con múltiples reexpediciones

# TDC

datos_expedicion_tdc=datos_tarjeta_credito_limpio[['Documento','Número de cuenta','Número de tarjeta','Número de tarjeta anterior','Situación de tarjeta','Descripción de baja','Fecha de alta de la tarjeta','Fecha de baja de la tarjeta']]
datos_reexpediciones_tdc=datos_expedicion_tdc.loc[(datos_expedicion_tdc['Número de tarjeta anterior']!='')|((datos_expedicion_tdc['Número de tarjeta']==datos_expedicion_tdc['Número de tarjeta anterior'].shift(-1))&(datos_expedicion_tdc['Documento']==datos_expedicion_tdc['Documento'].shift(-1)))].reset_index(drop=True)


datos_reexpedicion_tdc=pd.merge(datos_reexpediciones_tdc,pd.DataFrame(datos_reexpediciones_tdc.groupby(by=['Documento']).size(),columns=['Cantidad de expediciones totales']).reset_index(),on=['Documento'])


datos_reexpedicion_tdc['Cantidad de altas en 30 días']=datos_reexpedicion_tdc.groupby(by=['Documento','Número de cuenta']).rolling('30d',on='Fecha de alta de la tarjeta')['Número de tarjeta'].count().reset_index()['Número de tarjeta']
datos_reexpedicion_tdc['Cantidad de altas en 1 día']=datos_reexpedicion_tdc.groupby(by=['Documento','Número de cuenta']).rolling('2d',on='Fecha de alta de la tarjeta')['Número de tarjeta'].count().reset_index()['Número de tarjeta']


datos_reexpedicion_tdc['Alerta de altas mes']=np.where(datos_reexpedicion_tdc['Cantidad de altas en 30 días']>2,1,0)
datos_reexpedicion_tdc['Alerta de altas día']=np.where(datos_reexpedicion_tdc['Cantidad de altas en 1 día']>1,1,0)


datos_reexpedicion_tdc['Delta transaccional']=datos_reexpedicion_tdc.groupby(by=['Documento','Número de cuenta'])['Fecha de alta de la tarjeta'].diff()




datos_reexpedicion_tdc['Reexpedición día']=np.where((datos_reexpedicion_tdc['Delta transaccional']<=pd.Timedelta(days=1)),1,0)
datos_reexpedicion_tdc['Reexpedición mes']=np.where((datos_reexpedicion_tdc['Delta transaccional']<=pd.Timedelta(days=31)),1,0)


indicador_alerta_dia_reexpedicion_tdc=pd.DataFrame(datos_reexpedicion_tdc.loc[datos_reexpedicion_tdc['Reexpedición día']==1].groupby(by=['Documento']).size(),columns=['Alertas reexpediciones día']).reset_index()
indicador_alerta_mes_reexpedicion_tdc=pd.DataFrame(datos_reexpedicion_tdc.loc[datos_reexpedicion_tdc['Reexpedición mes']==1].groupby(by=['Documento']).size(),columns=['Alertas reexpediciones mes']).reset_index()


alertas_reexpedicion_tdc=pd.merge(datos_reexpedicion_tdc,indicador_alerta_dia_reexpedicion_tdc,on=['Documento'],how='left')
alertas_reexpedicion_tdc=pd.merge(alertas_reexpedicion_tdc,indicador_alerta_mes_reexpedicion_tdc,on=['Documento'],how='left')


alertas_reexpedicion_tdc = alertas_reexpedicion_tdc[alertas_reexpedicion_tdc['Fecha de alta de la tarjeta']>=datetime.now()-timedelta(days=31)]
alertas_reexpedicion_tdc = alertas_reexpedicion_tdc[alertas_reexpedicion_tdc['Delta transaccional']<=pd.Timedelta(days=30)]
alertas_reexpedicion_tdc['Delta transaccional']=alertas_reexpedicion_tdc['Delta transaccional'].astype(str)


alertas_reexpedicion_tdc.to_sql('AlertaTDCReexpedicion',
                                 con=config_db_fraude[1],
                                 if_exists='replace',
                                 index=False,
                                 schema='dbo')

print('4. AlertaTDCReexpedicion Actualizada')

# Débito

datos_expedicion_debito=datos_tarjeta_debito_limpio[['Documento','Número de cuenta','Número de tarjeta','Condición de tarjeta','Estado de tarjeta','Fecha de condición','Fecha de estado']]
datos_reexpediciones_debito=datos_expedicion_debito.loc[datos_expedicion_debito['Condición de tarjeta']=='REX']

datos_reexpedicion_debito=pd.merge(datos_reexpediciones_debito,pd.DataFrame(datos_reexpediciones_debito.groupby(by=['Documento']).size(),columns=['Cantidad de reexpediciones totales']).reset_index(),on=['Documento'])

datos_reexpedicion_debito['Cantidad de altas en 30 días']=np.array(datos_reexpedicion_debito.groupby(by=['Documento','Número de cuenta']).rolling('30d',on='Fecha de condición')['Número de tarjeta'].count().reset_index()['Número de tarjeta'])
datos_reexpedicion_debito['Cantidad de altas en 1 día']=np.array(datos_reexpedicion_debito.groupby(by=['Documento','Número de cuenta']).rolling('2d',on='Fecha de condición')['Número de tarjeta'].count().reset_index()['Número de tarjeta'])

datos_reexpedicion_debito['Alerta de altas mes']=np.where(datos_reexpedicion_debito['Cantidad de altas en 30 días']>2,1,0)
datos_reexpedicion_debito['Alerta de altas día']=np.where(datos_reexpedicion_debito['Cantidad de altas en 1 día']>1,1,0)


datos_reexpedicion_debito['Delta transaccional']=datos_reexpedicion_debito.groupby(by=['Documento','Número de cuenta'])['Fecha de condición'].diff()


datos_reexpedicion_debito['Reexpedición día']=np.where((datos_reexpedicion_debito['Delta transaccional']<=pd.Timedelta(days=1)),1,0)
datos_reexpedicion_debito['Reexpedición mes']=np.where((datos_reexpedicion_debito['Delta transaccional']<=pd.Timedelta(days=31)),1,0)


indicador_alerta_dia_reexpedicion_debito=pd.DataFrame(datos_reexpedicion_debito.loc[datos_reexpedicion_debito['Reexpedición día']==1].groupby(by=['Documento']).size(),columns=['Alertas reexpediciones día']).reset_index()
indicador_alerta_mes_reexpedicion_debito=pd.DataFrame(datos_reexpedicion_debito.loc[datos_reexpedicion_debito['Reexpedición mes']==1].groupby(by=['Documento']).size(),columns=['Alertas reexpediciones mes']).reset_index()


alertas_reexpedicion_debito=pd.merge(datos_reexpedicion_debito,indicador_alerta_dia_reexpedicion_debito,on=['Documento'],how='left')
alertas_reexpedicion_debito=pd.merge(alertas_reexpedicion_debito,indicador_alerta_mes_reexpedicion_debito,on=['Documento'],how='left')


alertas_reexpedicion_debito['Delta transaccional']=alertas_reexpedicion_debito['Delta transaccional'].astype(str)


alertas_reexpedicion_debito.to_sql('AlertaDebitoReexpedicion',
                                    con=config_db_fraude[1],
                                    if_exists='replace',
                                    index=False,
                                    schema='dbo')

print('5. AlertaDebitoReexpedicion Actualizada')

# Clientes con cambios de celular o correo y apertura de producto

datos_creacion_actualizacion=datos_producto_actualizacion[['Documento','Número de cuenta','Fecha de apertura','Linea de producto','Descripción de producto','Estado de la cuenta','Saldo capital','Fecha de evento','Evento','Mensaje']]


alerta_creacion_actualizacion_celular=datos_creacion_actualizacion.loc[(datos_creacion_actualizacion['Evento']=='Celular actualizado')&((datos_creacion_actualizacion['Fecha de evento'].dt.strftime('%Y%m%d')==datos_creacion_actualizacion['Fecha de apertura'].dt.strftime('%Y%m%d'))|(pd.to_datetime(datos_creacion_actualizacion['Fecha de evento'].dt.strftime('%Y%m%d'))+pd.DateOffset(days=1)==pd.to_datetime(datos_creacion_actualizacion['Fecha de apertura'].dt.strftime('%Y%m%d')))|(pd.to_datetime(datos_creacion_actualizacion['Fecha de evento'].dt.strftime('%Y%m%d'))-pd.DateOffset(days=1)==pd.to_datetime(datos_creacion_actualizacion['Fecha de apertura'].dt.strftime('%Y%m%d'))))]
alerta_creacion_actualizacion_correo=datos_creacion_actualizacion.loc[(datos_creacion_actualizacion['Evento']=='Correo actualizado')&((datos_creacion_actualizacion['Fecha de evento'].dt.strftime('%Y%m%d')==datos_creacion_actualizacion['Fecha de apertura'].dt.strftime('%Y%m%d'))|(pd.to_datetime(datos_creacion_actualizacion['Fecha de evento'].dt.strftime('%Y%m%d'))+pd.DateOffset(days=1)==pd.to_datetime(datos_creacion_actualizacion['Fecha de apertura'].dt.strftime('%Y%m%d')))|(pd.to_datetime(datos_creacion_actualizacion['Fecha de evento'].dt.strftime('%Y%m%d'))-pd.DateOffset(days=1)==pd.to_datetime(datos_creacion_actualizacion['Fecha de apertura'].dt.strftime('%Y%m%d'))))]


alerta_creacion_actualizacion=pd.concat([alerta_creacion_actualizacion_celular,alerta_creacion_actualizacion_correo]).reset_index(drop=True)


alerta_creacion_actualizacion.to_sql('AlertaCreacionActualizacion',
                                      con=config_db_fraude[1],
                                      if_exists='replace',
                                      index=False,
                                      schema='dbo')

print('6. AlertaCreacionActualizacion Actualizada')

# Clientes que registran múltiples IPs para transacciones en corto tiempo

datos_transaccion_digital=datos_log_portal_limpio[(datos_log_portal_limpio['Descripción']=='EnviarDinero')&(datos_log_portal_limpio['Mensaje'].str.contains('Transferencia Exitosa'))].reset_index(drop=True)
datos_transaccion_digital['Evento']='Transancción completada'


datos_transaccion_digital['Delta transaccional']=datos_transaccion_digital.groupby(['Documento'])['Fecha de evento'].diff()


datos_transaccion_digital['IP Distinta']=np.where((datos_transaccion_digital.shift(1)['IP']!=datos_transaccion_digital['IP'])&(datos_transaccion_digital.shift(1)['Documento']==datos_transaccion_digital['Documento']),True,False)
datos_ip_distintas=datos_transaccion_digital.loc[(datos_transaccion_digital['IP Distinta']==True)|(datos_transaccion_digital['IP Distinta']==True).shift(-1)].drop_duplicates()


alerta_transacciones_ip_distinta=datos_ip_distintas.loc[((datos_ip_distintas['Delta transaccional']<=pd.Timedelta(days=1))&(datos_ip_distintas['IP Distinta']==True))|(((datos_ip_distintas['Delta transaccional']<=pd.Timedelta(days=1))&(datos_ip_distintas['IP Distinta']==True)).shift(-1))]
alerta_transacciones_ip_distinta['Delta transaccional']=alerta_transacciones_ip_distinta['Delta transaccional'].astype(str)

alerta_transacciones_ip_distinta.to_sql('AlertaDisparidadIP',
                                         con=config_db_fraude[1],
                                         if_exists='replace',
                                         index=False,
                                         schema='dbo')

print('7. AlertaDisparidadIP Actualizada')

# Clientes con múltiples registros de cuentas


datos_cantidad_productos=pd.merge(datos_productos_limpio2,pd.DataFrame(datos_productos_limpio2.groupby(by=['Documento'])['Documento'].size()).rename(columns={'Documento':'Cantidad de productos'}).reset_index(),on='Documento',how='inner')

datos_cantidad_productos['Delta transaccional']=datos_cantidad_productos.sort_values(by=['Documento','Fecha de apertura']).groupby(by=['Documento'])['Fecha de apertura'].diff()


alerta_registros_multiples=datos_cantidad_productos.loc[(datos_cantidad_productos['Delta transaccional']<=pd.Timedelta(days=1))|((datos_cantidad_productos['Delta transaccional']<=pd.Timedelta(days=1)).shift(-1))]
alerta_registros_multiples['Delta transaccional']=alerta_registros_multiples['Delta transaccional'].astype(str)


alerta_registros_multiples.to_sql('AlertaRegistrosAtipicos',
                                   con=config_db_fraude[1],
                                   if_exists='replace',
                                   index=False,
                                   schema='dbo')

print('8. AlertaRegistrosAtipicos Actualizada')

# Clientes con olvido de contraseña y realizan transferencias

datos_olvido_contraseña_efectuado=datos_portal_olvido_contraseña.loc[datos_portal_olvido_contraseña['Mensaje'].str.contains('Se envia doble otp')]
datos_olvido_contraseña_efectuado['Evento']='Olvido de contraseña'

datos_olvido_contraseña_efectuado['Fecha de evento']=pd.to_datetime(datos_olvido_contraseña_efectuado['Fecha de evento'].dt.strftime('%Y-%m-%d'))
datos_olvido_contraseña_efectuado_unico=datos_olvido_contraseña_efectuado.drop_duplicates(subset=['Fecha de evento','Documento'])


alerta_transacciones_contraseña_olvido=pd.merge(datos_olvido_contraseña_efectuado_unico,datos_transacciones_limpio,left_on=['Documento','Fecha de evento'],right_on=['Documento','Fecha de la transacción'],how='inner')

alerta_transacciones_contraseña_olvido.to_sql('AlertaTransaccionesOlvido',
                                               con=config_db_fraude[1],
                                               if_exists='replace',
                                               index=False,
                                               schema='dbo')

print('9. AlertaTransaccionesOlvido Actualizada')

# Clientes con actualización de datos y realizan transferencias

datos_actualizacion_dia=datos_contacto_actualizan.drop(columns=['Delta transaccional'])
datos_actualizacion_dia['Fecha de evento']=pd.to_datetime(datos_actualizacion_dia['Fecha de evento'].dt.strftime('%Y-%m-%d'))

datos_actualizacion_dia_unico=datos_actualizacion_dia.drop_duplicates(subset=['Fecha de evento','Documento'])

alerta_transacciones_actualizacion=pd.merge(datos_actualizacion_dia_unico,datos_transacciones_limpio,left_on=['Documento','Fecha de evento'],right_on=['Documento','Fecha de la transacción'],how='inner')

alerta_transacciones_actualizacion.to_sql('AlertaTransaccionesActualizacion',
                                           con=config_db_fraude[1],
                                           if_exists='replace',
                                           index=False,
                                           schema='dbo')

print('10. AlertaTransaccionesActualizacion Actualizada')

# Clientes con alto uso de tarjeta de crédito o débito por encima de sus ingresos

datos_transacciones_ingreso=datos_transacciones_cliente[['Documento','Número de cuenta','Linea de producto','Fecha de la transacción','Monto de la transacción','Descripción de la transacción','Codigo tipo transacción','Lugar de la transacción','Ingresos']]
datos_transacciones_ingreso=datos_transacciones_ingreso.loc[(datos_transacciones_ingreso['Ingresos']>0)&((datos_transacciones_ingreso['Linea de producto']=='TDC')|(datos_transacciones_ingreso['Linea de producto']=='Ahorro'))]

alerta_transaccionalidad_ingreso=datos_transacciones_ingreso.loc[(datos_transacciones_ingreso['Monto de la transacción']>2*1000*datos_transacciones_ingreso['Ingresos'])&(datos_transacciones_ingreso['Monto de la transacción']>0)]


alerta_transaccionalidad_ingreso.to_sql('AlertaTransaccionesIngresosTDC',
                                         con=config_db_fraude[1],
                                         if_exists='replace',
                                         index=False,
                                         schema='dbo')

print('11. AlertaTransaccionesIngresosTDC Actualizada')

# Clientes que realizan transferencias tras alta inactividad 
datos_transacciones_inactividad=datos_transacciones_limpio[['Documento','Número de cuenta','Fecha de la transacción','Codigo tipo transacción','Monto de la transacción','Descripción de la transacción','Lugar de la transacción','Linea de producto','Descripción del producto']]
datos_transacciones_inactividad=datos_transacciones_inactividad.loc[(datos_transacciones_inactividad['Linea de producto']=='Ahorro')|(datos_transacciones_inactividad['Linea de producto']=='TDC')]

datos_transacciones_inactividad['Delta transaccional']=datos_transacciones_inactividad.sort_values(by=['Documento','Fecha de la transacción']).groupby(by=['Documento'])['Fecha de la transacción'].diff()
alerta_transacciones_inactividad=datos_transacciones_inactividad.loc[(datos_transacciones_inactividad['Delta transaccional']>=pd.Timedelta(days=60))|((datos_transacciones_inactividad['Delta transaccional']>=pd.Timedelta(days=60)).shift(-1))]
alerta_transacciones_inactividad['Delta transaccional']=alerta_transacciones_inactividad['Delta transaccional'].astype(str)


alerta_transacciones_inactividad.to_sql('AlertaTransaccionesInactividad',
                                         con=config_db_fraude[1],
                                         if_exists='replace',
                                         index=False,
                                         schema='dbo')

print('12. AlertaTransaccionesInactividad Actualizada')

# Clientes con bajos movimientos que los incrementan de manera súbita 
datos_transacciones_productos_movimientos=datos_transacciones_limpio.loc[(datos_transacciones_limpio['Linea de producto']=='Ahorro')|(datos_transacciones_limpio['Linea de producto']=='TDC')]

datos_transacciones_productos_movimientos['Media transaccional de movimientos (largo)']=np.array(datos_transacciones_productos_movimientos.groupby(by=['Documento','Número de cuenta']).rolling('150d',on='Fecha de la transacción')['Monto de la transacción'].mean().reset_index()['Monto de la transacción'])
datos_transacciones_productos_movimientos['Media transaccional de movimientos (corto)']=np.array(datos_transacciones_productos_movimientos.groupby(by=['Documento','Número de cuenta']).rolling('1d',on='Fecha de la transacción')['Monto de la transacción'].mean().reset_index()['Monto de la transacción'])

datos_transacciones_productos_movimientos['Alerta cambio comportamiento transaccional']=np.where(datos_transacciones_productos_movimientos['Media transaccional de movimientos (corto)']>10*datos_transacciones_productos_movimientos['Media transaccional de movimientos (largo)'],1,0)

alerta_cambio_comportamiento_transaccional=datos_transacciones_productos_movimientos.loc[datos_transacciones_productos_movimientos['Alerta cambio comportamiento transaccional']==1]

alerta_cambio_comportamiento_transaccional.to_sql('AlertaVariacionTransaccional',
                                                   con=config_db_fraude[1],
                                                   if_exists='replace',
                                                   index=False,
                                                   schema='dbo')

print('13. AlertaVariacionTransaccional Actualizada')

# Cambio de datos personales en los últimos 90 días 



datos_ultimo_cambio_actualizacion=datos_demograficos_limpio.drop_duplicates(subset=['Documento'],keep='last')
datos_ultimo_cambio_actualizacion['Tiempo última actualización']=pd.to_datetime('now')-datos_ultimo_cambio_actualizacion['Fecha última actualización']
datos_ultimo_cambio_actualizacion['Alerta última actualización']=np.where((datos_ultimo_cambio_actualizacion['Tiempo última actualización']<pd.Timedelta(days=30))&(datos_ultimo_cambio_actualizacion['Fecha última actualización']>datos_ultimo_cambio_actualizacion['Fecha de vinculación']),1,0)
alerta_ultimo_cambio_actualizacion=datos_ultimo_cambio_actualizacion.loc[datos_ultimo_cambio_actualizacion['Alerta última actualización']==1]


alerta_ultimo_cambio_actualizacion['Tiempo última actualización']=alerta_ultimo_cambio_actualizacion['Tiempo última actualización'].astype(str)


alerta_ultimo_cambio_actualizacion.to_sql('AlertaClienteActualiza',
                                           con=config_db_fraude[1],
                                           if_exists='replace',
                                           index=False,
                                           schema='dbo')

print('14. AlertaClienteActualiza Actualizada')

# Alerta misma IP

datos_enrolados_ip=datos_log_portal_limpio[['Documento','IP','Fecha de evento']].drop_duplicates(subset=['Documento','IP'],keep='last').rename(columns={'Fecha de evento':'Fecha último log'})

datos_cantidad_clientes_asociado_ip=pd.DataFrame(datos_enrolados_ip.groupby(by=['IP']).size(),columns=['Cantidad de Ips asociadas']).reset_index()
datos_cuentas_ips=pd.merge(datos_enrolados_ip,datos_cantidad_clientes_asociado_ip,on=['IP'])

alerta_ips_multiples_clientes=datos_cuentas_ips.loc[datos_cuentas_ips['Cantidad de Ips asociadas']>1].reset_index(drop=True)

alerta_ips_multiples_clientes.to_sql('AlertaIPMultiplesUsuarios',
                                      con=config_db_fraude[1],
                                      if_exists='replace',
                                      index=False,
                                      schema='dbo')

print('15. AlertaIPMultiplesUsuarios Actualizada')

# Clientes en Flexi con salidas de dinero superiores a 5m de pesos

datos_salida_cliente_flexi=datos_transacciones_flexi_cliente.loc[(datos_transacciones_flexi_cliente['Codigo tipo transacción']>='6')]


datos_salida_cliente_flexi['Salidas totales en el último día']=np.array(datos_salida_cliente_flexi.groupby(by=['Documento','Número de cuenta']).rolling('24h',on='Fecha de la transacción')['Monto de la transacción'].sum().reset_index()['Monto de la transacción'])
datos_salida_cliente_flexi['Alerta de altas salidas']=np.where(datos_salida_cliente_flexi['Salidas totales en el último día']>5000000,1,0)


alerta_salidas_cliente_flexi=datos_salida_cliente_flexi.loc[datos_salida_cliente_flexi['Alerta de altas salidas']==1]


alerta_salidas_cliente_flexi.to_sql('AlertaFlexiAltasSalidas',
                                     con=config_db_fraude[1],
                                     if_exists='replace',
                                     index=False,
                                     schema='dbo')

print('16. AlertaFlexiAltasSalidas Actualizada')

# Cuentas Flexi con entradas superiores a 8m de pesos 

datos_entrada_cliente_flexi=datos_transacciones_flexi_cliente.loc[(datos_transacciones_flexi_cliente['Codigo tipo transacción']<'6')]


datos_entrada_cliente_flexi['Entradas totales en el último día']=np.array(datos_entrada_cliente_flexi.groupby(by=['Documento','Número de cuenta']).rolling('24h',on='Fecha de la transacción')['Monto de la transacción'].sum().reset_index()['Monto de la transacción'])
datos_entrada_cliente_flexi['Alerta de altas entradas']=np.where(datos_entrada_cliente_flexi['Entradas totales en el último día']>=8000000,1,0)


alerta_entradas_cliente_flexi=datos_entrada_cliente_flexi.loc[datos_entrada_cliente_flexi['Alerta de altas entradas']==1]


alerta_entradas_cliente_flexi.to_sql('AlertaFlexiAltasEntradas',
                                      con=config_db_fraude[1],
                                      if_exists='replace',
                                      index=False,
                                      schema='dbo')

print('17. AlertaFlexiAltasEntradas Actualizada')


# Clientes con más de dos recuperaciones de contraseñas en un mes


datos_olvido_contraseña_intentado=datos_portal_olvido_contraseña.loc[datos_portal_olvido_contraseña['Mensaje'].str.contains('Se envia doble otp')]


datos_olvido_contraseña_intentado['Cantidad de cambios en 30 días']=np.array(datos_olvido_contraseña_intentado.groupby(by=['Documento']).rolling('30d',on='Fecha de evento')['Sesión'].count().reset_index()['Sesión'])
datos_olvido_contraseña_intentado['Alerta de cambios mes']=np.where(datos_olvido_contraseña_intentado['Cantidad de cambios en 30 días']>1,1,0)

datos_olvido_contraseña_intentado['Delta transaccional']=datos_olvido_contraseña_intentado.sort_values(by=['Documento','Fecha de evento']).groupby(by=['Documento'])['Fecha de evento'].diff()

alerta_olvido_contraseña_multiple=datos_olvido_contraseña_intentado.loc[(datos_olvido_contraseña_intentado['Delta transaccional']<=pd.Timedelta(days=30))|((datos_olvido_contraseña_intentado['Delta transaccional']<=pd.Timedelta(days=30)).shift(-1))]
alerta_olvido_contraseña_multiple['Alerta cambio contraseña']=np.where(alerta_olvido_contraseña_multiple['Delta transaccional']<=pd.Timedelta(days=30),1,0)
alerta_olvido_contraseña_multiple=pd.merge(alerta_olvido_contraseña_multiple,pd.DataFrame(alerta_olvido_contraseña_multiple.groupby(by='Documento').size(),columns=['Cantidad alertas cambio contraseña']).reset_index(),on=['Documento'],how='inner')
alerta_olvido_contraseña_multiple['Delta transaccional']=alerta_olvido_contraseña_multiple['Delta transaccional'].astype(str)

alerta_olvido_contraseña_multiple.to_sql('AlertaRecuperacionContraseñaMultiple',
                                          con=config_db_fraude[1],
                                          if_exists='replace',
                                          index=False,
                                          schema='dbo')

print('18. AlertaRecuperacionContraseñaMultiple Actualizada')

# Clientes con correo de riesgo

datos_cliente_correo=datos_producto_cliente[['Documento','Nombre','Celular','Correo','Número de cuenta','Fecha de apertura','Linea de producto','Descripción de producto','Estado de la cuenta']]


alerta_correo_riesgo=datos_cliente_correo.loc[(datos_cliente_correo['Correo'].str.lower().str.contains('@protonmail'))|(datos_cliente_correo['Correo'].str.lower().str.contains('@mail'))].sort_values(by=['Fecha de apertura'],ascending=False)

alerta_correo_riesgo.to_sql('AlertaCorreoRiesgo',
                             con=config_db_fraude[1],
                             if_exists='replace',
                             index=False,
                             schema='dbo')

print('19. AlertaCorreoRiesgo Actualizada')

# Actualización reciente datos de contacto

alerta_actualizan_reciente=datos_contacto_actualizan.copy(deep=True)
alerta_actualizan_reciente['Tiempo desde actualización']=pd.to_datetime('now')-alerta_actualizan_reciente['Fecha de evento']


alerta_actualizan_reciente['Alerta actualización reciente']=np.where(alerta_actualizan_reciente['Tiempo desde actualización']<=pd.Timedelta(days=30),1,0)
alerta_actualizan_reciente['Alerta alto riesgo']=np.where(((alerta_actualizan_reciente['Delta transaccional']<=pd.Timedelta(hours=4))&(alerta_actualizan_reciente['Evento']=='Correo actualizado')&((alerta_actualizan_reciente['Evento'].shift(-1))=='Celular actualizado'))|((alerta_actualizan_reciente['Delta transaccional']<=pd.Timedelta(hours=4))&(alerta_actualizan_reciente['Evento']=='Celular actualizado')&((alerta_actualizan_reciente['Evento']).shift(-1)=='Correo actualizado')),1,0)
alerta_actualizan_reciente[['Delta transaccional','Tiempo desde actualización']]=alerta_actualizan_reciente[['Delta transaccional','Tiempo desde actualización']].astype(str)


alerta_actualizan_reciente.to_sql('AlertaActualizacionRiesgo',
                                   con=config_db_fraude[1],
                                   if_exists='replace',
                                   index=False,
                                   schema='dbo')

print('20. AlertaActualizacionRiesgo Actualizada')

# Misma IP de creación del producto para



datos_registro_creacion=datos_log_creacion_limpio[['Documento','Fecha de registro','Sesión','IP','Mensaje','Descripción de tipo']]


datos_registro_creacion_cantidad=pd.merge(datos_registro_creacion,pd.DataFrame(datos_registro_creacion.groupby(by=['IP','Descripción de tipo']).size(),columns=['Cantidad de cuentas asociadas']).reset_index(),how='inner',on=['IP'])


alerta_creacion_cantidad_ip=datos_registro_creacion_cantidad.loc[datos_registro_creacion_cantidad['Cantidad de cuentas asociadas']>3]


alerta_creacion_cantidad_ip.to_sql('AlertaSimilitudIPCreacion',
                                    con=config_db_fraude[1],
                                    if_exists='replace',
                                    index=False,
                                    schema='dbo')

print('21. AlertaSimilitudIPCreacion Actualizada')

# Intentos de vinculación fallida


datos_intentos_vinculacion=datos_registro_creacion_limpio.drop_duplicates(subset=['Código de solicitud'])



datos_intentos_vinculacion['Delta transaccional']=datos_intentos_vinculacion.groupby(by=['Documento'])[['Fecha inicial']].diff()
datos_intentos_vinculacion['Intento vinculación reciente']=np.where(datos_intentos_vinculacion['Delta transaccional']<=pd.Timedelta(hours=72),True,False)


alerta_intentos_vinculacion=datos_intentos_vinculacion.loc[(datos_intentos_vinculacion['Intento vinculación reciente']==True)|((datos_intentos_vinculacion['Intento vinculación reciente']==True).shift(-1))].reset_index(drop=True)
alerta_intentos_vinculacion['Delta transaccional']=alerta_intentos_vinculacion['Delta transaccional'].astype(str)

alerta_intentos_vinculacion.to_sql('AlertaIntentosVinculacion',
                                    con=config_db_fraude[1],
                                    if_exists='replace',
                                    index=False,
                                    schema='dbo')

print('22. AlertaIntentosVinculacion Actualizada')

# Uso de dipositivo no seguro


datos_dispositivo_vinculado_disponible=datos_disposito_vinculado_limpio.dropna(subset=['Fingerprint']).reset_index(drop=True)
datos_dispositivo_vinculado_disponible['Fingerprint']=datos_dispositivo_vinculado_disponible['Fingerprint'].apply(lambda x: json.loads(x))
datos_dispositivo_descompuesto=pd.concat([datos_dispositivo_vinculado_disponible.drop(['Fingerprint'],axis=1),pd.DataFrame(datos_dispositivo_vinculado_disponible['Fingerprint'].tolist())],axis=1)
datos_dispositivo_vinculado_especifico=datos_dispositivo_descompuesto[['Documento','IP','SO','DeviceType','Latitud','Longitud','Fecha de registro','Fecha de cambio','Identificador dispositivo','androidId','platform','brand','id','manufacturer','model','identifierForVendor','systemName','utsname.machine:','systemVersion']]
datos_dispositivo_vinculado_especifico['Id único dispositivo']=datos_dispositivo_vinculado_especifico['androidId'].fillna(datos_dispositivo_vinculado_especifico['identifierForVendor'])
datos_dispositivo_vinculado_especifico['Marca']=datos_dispositivo_vinculado_especifico['brand'].fillna(datos_dispositivo_vinculado_especifico['model'])
datos_dispositivo_vinculado_especifico['Fabricante']=datos_dispositivo_vinculado_especifico['manufacturer'].fillna('Apple')
datos_dispositivo_vinculado_especifico['Modelo']=datos_dispositivo_vinculado_especifico['model'].fillna(datos_dispositivo_vinculado_especifico['utsname.machine:'])
datos_dispositivo_vinculado_especifico['Plataforma']=datos_dispositivo_vinculado_especifico['platform'].fillna('iOS')


datos_dispositivo_vinculado_detalle=datos_dispositivo_vinculado_especifico[['Documento','IP','SO','DeviceType','Latitud','Longitud','Fecha de registro','Fecha de cambio','Id único dispositivo','Marca','Fabricante','Modelo','Plataforma']]


datos_cantidad_clientes_dispositivos=pd.merge(datos_dispositivo_vinculado_detalle,pd.DataFrame(datos_dispositivo_vinculado_detalle.groupby(by=['Id único dispositivo']).size(),columns=['Cantidad de usuarios asociados']).reset_index(),on=['Id único dispositivo'],how='inner')


alerta_cantidad_dispositivos_vinculados=datos_cantidad_clientes_dispositivos.loc[datos_cantidad_clientes_dispositivos['Cantidad de usuarios asociados']>1]


alerta_cantidad_dispositivos_vinculados.to_sql('AlertaMultipleDispositivo',
                                                con=config_db_fraude[1],
                                                if_exists='replace',
                                                index=False,
                                                schema='dbo')

print('23. AlertaMultipleDispositivo Actualizada')

# Gran cantidad de intentos de evidente


datos_evidente_descompuesto=pd.concat([datos_log_evidente_limpio.drop(['Respuesta'],axis=1),pd.DataFrame(datos_log_evidente_limpio['Respuesta'].tolist())],axis=1)

datos_intentos_evidente=datos_evidente_descompuesto.dropna(subset=[' StatusDesc'])[['Documento','Fecha de registro']]

datos_validaciones_evidente=datos_evidente_descompuesto.dropna(subset=[' Aprobación'])[['Documento','Fecha de registro',' Aprobación',' Aprobado100PorCientoOK',' PreguntasCompletas']]
datos_validaciones_evidente_positivas=datos_validaciones_evidente.loc[datos_validaciones_evidente[' Aprobación']==' true']
datos_validaciones_evidente_negativas=datos_validaciones_evidente.loc[datos_validaciones_evidente[' Aprobación']==' false']


datos_evidente_consolidado=pd.merge(datos_intentos_evidente,pd.DataFrame(datos_intentos_evidente.groupby(by=['Documento']).size(),columns=['Intentos evidente']).reset_index(),how='left',on=['Documento'])
datos_evidente_consolidado=pd.merge(datos_evidente_consolidado,pd.DataFrame(datos_validaciones_evidente.groupby(by=['Documento']).size(),columns=['Validaciones evidente']).reset_index(),how='left',on=['Documento'])
datos_evidente_consolidado=pd.merge(datos_evidente_consolidado,pd.DataFrame(datos_validaciones_evidente_positivas.groupby(by=['Documento']).size(),columns=['Validaciones evidente positivos']).reset_index(),how='left',on=['Documento'])
datos_evidente_consolidado=pd.merge(datos_evidente_consolidado,pd.DataFrame(datos_validaciones_evidente_negativas.groupby(by=['Documento']).size(),columns=['Validaciones evidente negativos']).reset_index(),how='left',on=['Documento'])
datos_evidente_consolidado=datos_evidente_consolidado.fillna(0).sort_values(by=['Fecha de registro']).drop_duplicates(subset=['Documento']).rename(columns={'Fecha de registro':'Fecha de último intento'})
datos_evidente_consolidado=pd.merge(datos_evidente_consolidado,datos_producto_cliente2.sort_values(by=['Fecha de apertura']).drop_duplicates(subset=['Documento'],keep='last')[['Documento','Número de cuenta','Fecha de apertura','Estado de la cuenta']],how='inner',on=['Documento']).rename(columns={'Fecha de apertura':'Fecha de última apertura','Estado de la cuenta':'Estado de última cuenta','Número de cuenta':'Número de última cuenta'})
datos_evidente_consolidado=pd.merge(datos_evidente_consolidado,datos_producto_cliente2.sort_values(by=['Fecha de apertura']).drop_duplicates(subset=['Documento'],keep='first')[['Documento','Número de cuenta','Fecha de apertura','Estado de la cuenta']],how='inner',on=['Documento']).rename(columns={'Fecha de apertura':'Fecha de primera apertura','Estado de la cuenta':'Estado de primera cuenta','Número de cuenta':'Número de primera cuenta'})

alerta_intentos_evidente=datos_evidente_consolidado.loc[datos_evidente_consolidado['Intentos evidente']>3]

alerta_intentos_evidente.to_sql('AlertaIntentosEvidente',
                                 con=config_db_fraude[1],
                                 if_exists='replace',
                                 index=False,
                                 schema='dbo')

print('24. AlertaIntentosEvidente Actualizada')

# Clientes con diferencias entre actividad económica y originadores


datos_core_actividad=datos_demograficos_core_limpio[['Documento','Celular','Correo','Código CIIU','Descripción CIIU','Fecha de vinculación']].rename(columns={'Fecha de vinculación':'Fecha'})
datos_core_actividad['Originador']='Core'

datos_agil_actividad=datos_demograficos_agil_limpio[['Documento','Celular','Correo','Código CIIU','Descripción CIIU','Fecha de solicitud']].rename(columns={'Fecha de solicitud':'Fecha'})
datos_agil_actividad['Originador']='AGIL'

datos_lp_actividad=datos_demograficos_lp_limpio[['Documento','Celular','Correo','Código CIIU','Descripción CIIU','Fecha de solicitud']].rename(columns={'Fecha de solicitud':'Fecha'})
datos_lp_actividad['Originador']='LP'



datos_clientes_actividad_economica=pd.concat([datos_core_actividad,datos_agil_actividad,datos_lp_actividad])
datos_clientes_actividad_economica=datos_clientes_actividad_economica.dropna(subset=['Código CIIU','Descripción CIIU'],how='any').sort_values(by=['Documento','Originador','Fecha'])
datos_clientes_actividad_economica['Código CIIU']=datos_clientes_actividad_economica['Código CIIU'].astype(float).astype(np.int64)
datos_clientes_actividad_economica_sin_repetir=datos_clientes_actividad_economica.drop_duplicates(subset=['Documento','Originador','Código CIIU','Descripción CIIU'],keep='last').sort_values(by=['Documento','Fecha'])


datos_clientes_filtro_ciiu=datos_clientes_actividad_economica_sin_repetir.drop_duplicates(subset=['Documento','Código CIIU'],keep='last')



datos_clientes_filtro_ciiu_cantidad=pd.merge(datos_clientes_filtro_ciiu,pd.DataFrame(datos_clientes_filtro_ciiu.groupby(by=['Documento']).size(),columns=['Cantidad de registros distintos']).reset_index(),how='inner',on=['Documento'])
alerta_clientes_disparidad_ciiu=datos_clientes_filtro_ciiu_cantidad.loc[datos_clientes_filtro_ciiu_cantidad['Cantidad de registros distintos']>1].sort_values(by=['Documento','Fecha']).reset_index(drop=True)



alerta_clientes_disparidad_ciiu.to_sql('AlertaDisparidadCIIU',
                                        con=config_db_fraude[1],
                                        if_exists='replace',
                                        index=False,
                                        schema='dbo')

print('25. AlertaDisparidadCIIU Actualizada')

# Transferencias en Transfiya y entre cuentas Finandina

datos_transacciones_ahorro_seleccion=datos_transacciones_limpio.loc[datos_transacciones_limpio['Linea de producto']=='Ahorro'][['Documento','Número de cuenta','Descripción del producto','Fecha de la transacción','Monto de la transacción','Descripción de la transacción','Lugar de la transacción','CaracterTransaccion']]
datos_transacciones_ahorro_seleccion['Lugar de la transacción']=datos_transacciones_ahorro_seleccion['Lugar de la transacción'].fillna('No reporta')


datos_transacciones_ahorro_vigilancia=datos_transacciones_ahorro_seleccion.loc[(datos_transacciones_ahorro_seleccion['Lugar de la transacción'].str.contains('E-TransfiYa'))|(datos_transacciones_ahorro_seleccion['Lugar de la transacción'].str.contains('Transferencia Inmediata'))].sort_values(by=['Documento','Número de cuenta','Fecha de la transacción'])
transacciones_sopechosas_transfiya=pd.DataFrame(datos_transacciones_ahorro_vigilancia.groupby(by=['Documento','Fecha de la transacción']).apply(lambda x: (x['Lugar de la transacción'].str.contains('E-TransfiYa').any())and(x['Lugar de la transacción'].str.contains('Transferencia Inmediata').any())),columns=['Día transacción sospechosa']).reset_index()


datos_transacciones_sospechosas_transfiya=pd.merge(datos_transacciones_ahorro_vigilancia,transacciones_sopechosas_transfiya,on=['Documento','Fecha de la transacción'],how='inner').sort_values(by=['Documento','Fecha de la transacción'])


alerta_transacciones_sospechosas=datos_transacciones_sospechosas_transfiya.loc[datos_transacciones_sospechosas_transfiya['Día transacción sospechosa']==True]

alerta_transacciones_sospechosas.to_sql('AlertaSospechaTransfiya',
                                         con=config_db_fraude[1],
                                         if_exists='replace',
                                         index=False,
                                         schema='dbo')

print('26. AlertaSospechaTransfiya Actualizada')

# Enrolamiento 3 meses después del último producto adquirido

datos_ultimo_producto=datos_productos_limpio.sort_values(by=['Documento','Fecha de apertura']).drop_duplicates(subset=['Documento'],keep='last').reset_index(drop=True)
datos_ultimo_producto_enrolamiento=pd.merge(datos_ultimo_producto,datos_log_enrolamiento_limpio,on=['Documento'],how='inner')

datos_ultimo_producto_enrolamiento['Delta de tiempo']=datos_ultimo_producto_enrolamiento['Fecha de enrolamiento']-datos_ultimo_producto_enrolamiento['Fecha de apertura']
datos_ultimo_producto_enrolamiento['Alerta de enrolamiento']=np.where(datos_ultimo_producto_enrolamiento['Delta de tiempo']>=pd.Timedelta(days=90),1,0)

alerta_creacion_enrolamiento=datos_ultimo_producto_enrolamiento.loc[datos_ultimo_producto_enrolamiento['Alerta de enrolamiento']==1]
alerta_creacion_enrolamiento['Delta de tiempo']=alerta_creacion_enrolamiento['Delta de tiempo'].astype(str)

alerta_creacion_enrolamiento.to_sql('AlertaCreacionEnrolamiento',
                                     con=config_db_fraude[1],
                                     if_exists='replace',
                                     index=False,
                                     schema='dbo')

print('27. AlertaCreacionEnrolamiento Actualizada')


# Cliente recibe y envía altas cantidades de dinero

datos_transacciones_ahorro=datos_transacciones_limpio.loc[datos_transacciones_limpio['Linea de producto']=='Ahorro']
datos_transacciones_ahorro_entradas=datos_transacciones_ahorro.loc[datos_transacciones_ahorro['CaracterTransaccion']=='Entrada']
datos_transacciones_ahorro_salidas=datos_transacciones_ahorro.loc[datos_transacciones_ahorro['CaracterTransaccion']=='Salida']

datos_transacciones_dia_entradas=datos_transacciones_ahorro_entradas.groupby(by=['Documento','Fecha de la transacción'])[['Documento','Fecha de la transacción','Monto de la transacción']].sum().reset_index().rename(columns={'Monto de la transacción':'Monto entradas'})
datos_transacciones_dia_salidas=datos_transacciones_ahorro_salidas.groupby(by=['Documento','Fecha de la transacción'])[['Documento','Fecha de la transacción','Monto de la transacción']].sum().reset_index().rename(columns={'Monto de la transacción':'Monto salidas'})


datos_transacciones_dia_consolidado=pd.merge(datos_transacciones_dia_entradas,datos_transacciones_dia_salidas,how='inner',on=['Documento','Fecha de la transacción'])
alerta_transacciones_entrada_salida_alta=datos_transacciones_dia_consolidado.loc[(datos_transacciones_dia_consolidado['Monto entradas']>=2000000)&(datos_transacciones_dia_consolidado['Monto salidas']>=2000000)]


alerta_transacciones_entrada_salida_alta.to_sql('AlertaAltaTransaccionalidad',
                                                 con=config_db_fraude[1],
                                                 if_exists='replace',
                                                 index=False,
                                                 schema='dbo')

print('28. AlertaAltaTransaccionalidad Actualizada')


## nuevo alertamiento CupoLimiteExcedido 

datos_alerta = datos_tarjeta_credito_limpio[datos_tarjeta_credito_limpio['Descripción estado general tarjeta']!='Cancelada'].loc[:, ['Documento', 'Número de cuenta', 'Cupo aprobado de la tarjeta','Cupo limite Banco','Saldo actual tarjeta']]
datos_alerta['Cupo limite tarjeta calculado']=datos_alerta['Cupo aprobado de la tarjeta'] +  datos_alerta['Cupo aprobado de la tarjeta']* 0.1
alertacupo = datos_alerta.groupby(['Documento'])['Cupo limite tarjeta calculado','Saldo actual tarjeta','Cupo limite Banco'].sum().reset_index()

alertacupo['Balance']=alertacupo['Cupo limite tarjeta calculado']-alertacupo['Saldo actual tarjeta']

alerta_cupo_excedido = alertacupo[alertacupo['Balance']<0]
alerta_cupo_excedido['Balance']=abs(alertacupo['Balance'])

alerta_cupo_excedido=alerta_cupo_excedido[['Documento','Cupo limite Banco','Cupo limite tarjeta calculado','Saldo actual tarjeta','Balance']]

alerta_cupo_excedido.to_sql('AlertaCupoLimiteExcedido',
                             con=config_db_fraude[1],
                             if_exists='replace',
                             index=False,
                             schema='dbo')

print('29. AlertaCupoLimiteExcedido Actualizada')



# tiempo de ejecucion 

tiempo_ejecucion_codigo = round(float(format(time.time() - startTime))/60,2)

# ajuste valores futuros alerta : AlertaClienteActualiza 

from datetime import datetime

alerta_ultimo_cambio_actualizacion2=alerta_ultimo_cambio_actualizacion[alerta_ultimo_cambio_actualizacion['Fecha última actualización']<=datetime.now()]

# alertamiento



data = {
  "Alertamiento":
  
['AlertaAhorroRegistroCelularSimilitud'
,'AlertaAhorroRegistroCorreoSimilitud'
,'AlertaTDCCondicionSimilitud'
,'AlertaTDCReexpedicion'
,'AlertaDebitoReexpedicion'
,'AlertaCreacionActualizacion'
,'AlertaDisparidadIP'
,'AlertaRegistrosAtipicos'
,'AlertaTransaccionesOlvido'
,'AlertaTransaccionesActualizacion'
,'AlertaTransaccionesIngresosTDC'
,'AlertaTransaccionesInactividad'
,'AlertaVariacionTransaccional'
,'AlertaClienteActualiza'
,'AlertaIPMultiplesUsuarios'
,'AlertaFlexiAltasSalidas'
,'AlertaFlexiAltasEntradas'
,'AlertaCorreoRiesgo'
,'AlertaRecuperacionContraseñaMultiple'
,'AlertaActualizacionRiesgo'
,'AlertaSimilitudIPCreacion'
,'AlertaIntentosVinculacion'
,'AlertaMultipleDispositivo'
,'AlertaIntentosEvidente'
,'AlertaDisparidadCIIU'
,'AlertaSospechaTransfiya'
,'AlertaCreacionEnrolamiento'
,'AlertaAltaTransaccionalidad'],

"última actualización":[
    alerta_cuenta_multiple_celular['Fecha de último uso'].max(),
    alerta_cuenta_multiple_correo['Fecha de último uso'].max(),
    alerta_doble_titularidad['Fecha de alta de la tarjeta'].max(),
    alertas_reexpedicion_tdc['Fecha de alta de la tarjeta'].max(),
    alertas_reexpedicion_debito['Fecha de condición'].max(),
    alerta_creacion_actualizacion['Fecha de apertura'].max(),
    alerta_transacciones_ip_distinta['Fecha de evento'].max(),
    alerta_registros_multiples['Fecha de apertura'].max(),
    alerta_transacciones_contraseña_olvido['Fecha de evento'].max(),
    alerta_transacciones_actualizacion['Fecha de evento'].max(),
    alerta_transaccionalidad_ingreso['Fecha de la transacción'].max(),
    alerta_transacciones_inactividad['Fecha de la transacción'].max(),
    alerta_cambio_comportamiento_transaccional['Fecha de la transacción'].max(),
    alerta_ultimo_cambio_actualizacion2['Fecha última actualización'].max(),
    alerta_ips_multiples_clientes['Fecha último log'].max(),
    alerta_salidas_cliente_flexi['Fecha de la transacción'].max(),
    alerta_entradas_cliente_flexi['Fecha de la transacción'].max(),
    alerta_correo_riesgo['Fecha de apertura'].max(),
    alerta_olvido_contraseña_multiple['Fecha de evento'].max(),
    alerta_actualizan_reciente['Fecha de evento'].max(),
    alerta_creacion_cantidad_ip['Fecha de registro'].max(),
    alerta_intentos_vinculacion['Fecha inicial'].max(),
    alerta_cantidad_dispositivos_vinculados['Fecha de registro'].max(),
    alerta_intentos_evidente['Fecha de última apertura'].max(),
    alerta_clientes_disparidad_ciiu['Fecha'].max(),
    alerta_transacciones_sospechosas['Fecha de la transacción'].max(),
    alerta_creacion_enrolamiento['Fecha de enrolamiento'].max(),
    alerta_transacciones_entrada_salida_alta['Fecha de la transacción'].max()]}


    
# load data into a DataFrame object:
df = pd.DataFrame(data)


#-------------------------------------------------------------------------------------------------#
#                             ACTUALIZACIÓN  MALLA ALERTAMIENTOS DE FRAUDE                        #
#-------------------------------------------------------------------------------------------------#

# librerías necesarias 

## temporizador 
import time 
startTime2 = time.time()



# Extracción de datos
# Inicialización de logger

logger=logger_config('Malla de alertas')

# Configuración base de datos de extracción



# Carga de datos 
# Carga de alertamientos


datos_actualizacion_riesgo = alerta_actualizan_reciente
datos_registro_celular_similitud = alerta_cuenta_multiple_celular
datos_registro_correo_similitud = alerta_cuenta_multiple_correo
datos_alta_transaccionalidad = alerta_transacciones_entrada_salida_alta
datos_cliente_actualiza = alerta_ultimo_cambio_actualizacion
datos_correo_riesgo = alerta_correo_riesgo
datos_creacion_actualizacion_malla = alerta_creacion_actualizacion
datos_creacion_enrolamiento = alerta_creacion_enrolamiento
datos_debito_reexpedicion = alertas_reexpedicion_debito
datos_disparidad_ciiu = alerta_clientes_disparidad_ciiu
datos_disparidad_ip = alerta_transacciones_ip_distinta
datos_flexi_entradas = alerta_entradas_cliente_flexi
datos_flexi_salidas = alerta_salidas_cliente_flexi
datos_intentos_evidente_malla = alerta_intentos_evidente
datos_intentos_vinculacion_malla = alerta_intentos_vinculacion
datos_ip_multiples_usuarios = alerta_ips_multiples_clientes
datos_multiple_dispositivo = alerta_cantidad_dispositivos_vinculados
datos_recuperacion_contraseña_multiple = alerta_olvido_contraseña_multiple
datos_registros_atipicos = alerta_registros_multiples
datos_similitud_ip_creacion = alerta_creacion_cantidad_ip
datos_sospecha_transfiya = alerta_transacciones_sospechosas
datos_tdc_condicion_similitud = alerta_doble_titularidad
datos_tdc_reexpedicion = alertas_reexpedicion_tdc
datos_transacciones_actualizacion = alerta_transacciones_actualizacion
datos_transacciones_inactividad_malla = alerta_transacciones_inactividad
datos_transacciones_ingresos_tdc = alerta_transaccionalidad_ingreso
datos_transacciones_olvido = alerta_transacciones_contraseña_olvido
datos_variacion_transaccional = alerta_cambio_comportamiento_transaccional
datos_cupo_superados = alerta_cupo_excedido


# ## Limpieza de datos

print('Inicio limpieza de datos MALLA')


datos_registro_celular_similitud_limpio=datos_registro_celular_similitud[['Documento']]
datos_registro_celular_similitud_limpio['Evento']='Alerta celular similitud'
datos_registro_celular_similitud_limpio=datos_registro_celular_similitud_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_registro_correo_similitud_limpio=datos_registro_correo_similitud[['Documento']]
datos_registro_correo_similitud_limpio['Evento']='Alerta correo similitud'
datos_registro_correo_similitud_limpio=datos_registro_correo_similitud_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_tdc_condicion_similitud_limpio=datos_tdc_condicion_similitud[['Documento']]
datos_tdc_condicion_similitud_limpio['Evento']='Alerta TDC similitud'
datos_tdc_condicion_similitud_limpio=datos_tdc_condicion_similitud_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_debito_reexpedicion_limpio=datos_debito_reexpedicion.loc[(datos_debito_reexpedicion['Alertas reexpediciones día'].notna())|(datos_debito_reexpedicion['Alertas reexpediciones mes'].notna())][['Documento']]
datos_debito_reexpedicion_limpio['Evento']='Alerta débito reexpedición'
datos_debito_reexpedicion_limpio=datos_debito_reexpedicion_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_tdc_reexpedicion_limpio=datos_tdc_reexpedicion.loc[(datos_tdc_reexpedicion['Alertas reexpediciones día'].notna())|(datos_tdc_reexpedicion['Alertas reexpediciones mes'].notna())][['Documento']]
datos_tdc_reexpedicion_limpio['Evento']='Alerta TDC reexpedición'
datos_tdc_reexpedicion_limpio=datos_tdc_reexpedicion_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_creacion_actualizacion_malla_limpio=datos_creacion_actualizacion_malla[['Documento','Evento']]
datos_creacion_actualizacion_malla_limpio['Evento']=np.where(datos_creacion_actualizacion_malla_limpio['Evento']=='Celular actualizado','Alerta creación actualización celular','Alerta creación actualización correo')
datos_creacion_actualizacion_malla_limpio=datos_creacion_actualizacion_malla_limpio.drop_duplicates(subset=['Documento','Evento']).reset_index(drop=True)

datos_disparidad_ip_limpio=datos_disparidad_ip.loc[datos_disparidad_ip['IP Distinta']==True][['Documento']]
datos_disparidad_ip_limpio['Evento']='Alerta disparidad IP'
datos_disparidad_ip_limpio=datos_disparidad_ip_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_registros_atipicos_limpio=datos_registros_atipicos.loc[datos_registros_atipicos['Cantidad de productos']!=1][['Documento']]
datos_registros_atipicos_limpio['Evento']='Alerta registros atípicos'
datos_registros_atipicos_limpio=datos_registros_atipicos_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_transacciones_olvido_limpio=datos_transacciones_olvido[['Documento']]
datos_transacciones_olvido_limpio['Evento']='Alerta transacciones olvido'
datos_transacciones_olvido_limpio=datos_transacciones_olvido_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_transacciones_actualizacion_limpio=datos_transacciones_actualizacion[['Documento']]
datos_transacciones_actualizacion_limpio['Evento']='Alerta transacciones actualización'
datos_transacciones_actualizacion_limpio=datos_transacciones_actualizacion_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_transacciones_ingresos_tdc_limpio=datos_transacciones_ingresos_tdc[['Documento']]
datos_transacciones_ingresos_tdc_limpio['Evento']='Alerta transacciones ingresos TDC'
datos_transacciones_ingresos_tdc_limpio=datos_transacciones_ingresos_tdc_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_transacciones_inactividad_malla_limpio=datos_transacciones_inactividad_malla[['Documento']]
datos_transacciones_inactividad_malla_limpio['Evento']='Alerta transacciones inactividad'
datos_transacciones_inactividad_malla_limpio=datos_transacciones_inactividad_malla_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_variacion_transaccional_limpio=datos_variacion_transaccional[['Documento']]
datos_variacion_transaccional_limpio['Evento']='Alerta variación transaccional'
datos_variacion_transaccional_limpio=datos_variacion_transaccional_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_cliente_actualiza_limpio=datos_cliente_actualiza[['Documento']]
datos_cliente_actualiza_limpio['Evento']='Alerta cliente actualiza'
datos_cliente_actualiza_limpio=datos_cliente_actualiza_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_ip_multiples_usuarios_limpio=datos_ip_multiples_usuarios[['Documento']]
datos_ip_multiples_usuarios_limpio['Evento']='Alerta IP múltiples usuarios'
datos_ip_multiples_usuarios_limpio=datos_ip_multiples_usuarios_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_flexi_salidas_limpio=datos_flexi_salidas[['Documento']]
datos_flexi_salidas_limpio['Evento']='Alerta flexi salidas'
datos_flexi_salidas_limpio=datos_flexi_salidas_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_flexi_entradas_limpio=datos_flexi_entradas[['Documento']]
datos_flexi_entradas_limpio['Evento']='Alerta flexi entradas'
datos_flexi_entradas_limpio=datos_flexi_entradas_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_correo_riesgo_limpio=datos_correo_riesgo[['Documento']]
datos_correo_riesgo_limpio['Evento']='Alerta correo riesgo'
datos_correo_riesgo_limpio=datos_correo_riesgo_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_recuperacion_contraseña_multiple_limpio=datos_recuperacion_contraseña_multiple[['Documento']]
datos_recuperacion_contraseña_multiple_limpio['Evento']='Alerta recuperacion contraseña múltiple'
datos_recuperacion_contraseña_multiple_limpio=datos_recuperacion_contraseña_multiple_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_actualizacion_riesgo_limpio=datos_actualizacion_riesgo[['Documento']]
datos_actualizacion_riesgo_limpio['Evento']='Alerta actualización riesgo'
datos_actualizacion_riesgo_limpio=datos_actualizacion_riesgo_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_similitud_ip_creacion_limpio=datos_similitud_ip_creacion[['Documento']]
datos_similitud_ip_creacion_limpio['Evento']='Alerta similitud ip creación'
datos_similitud_ip_creacion_limpio=datos_similitud_ip_creacion_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_intentos_vinculacion_malla_limpio=datos_intentos_vinculacion_malla[['Documento']]
datos_intentos_vinculacion_malla_limpio['Evento']='Alerta intentos vinculación'
datos_intentos_vinculacion_malla_limpio=datos_intentos_vinculacion_malla_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_multiple_dispositivo_limpio=datos_multiple_dispositivo[['Documento']]
datos_multiple_dispositivo_limpio['Evento']='Alerta múltiple dispositivo'
datos_multiple_dispositivo_limpio=datos_multiple_dispositivo_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_intentos_evidente_malla_limpio=datos_intentos_evidente_malla[['Documento']]
datos_intentos_evidente_malla_limpio['Evento']='Alerta intentos evidente'
datos_intentos_evidente_malla_limpio=datos_intentos_evidente_malla_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_disparidad_ciiu_limpio=datos_disparidad_ciiu[['Documento']]
datos_disparidad_ciiu_limpio['Evento']='Alerta disparidad CIIU'
datos_disparidad_ciiu_limpio=datos_disparidad_ciiu_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_sospecha_transfiya_limpio=datos_sospecha_transfiya[['Documento']]
datos_sospecha_transfiya_limpio['Evento']='Alerta sospecha transfiya'
datos_sospecha_transfiya_limpio=datos_sospecha_transfiya_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_creacion_enrolamiento_limpio=datos_creacion_enrolamiento[['Documento']]
datos_creacion_enrolamiento_limpio['Evento']='Alerta creación enrolamiento'
datos_creacion_enrolamiento_limpio=datos_creacion_enrolamiento_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_alta_transaccionalidad_limpio=datos_alta_transaccionalidad[['Documento']]
datos_alta_transaccionalidad_limpio['Evento']='Alerta alta transaccionalidad'
datos_alta_transaccionalidad_limpio=datos_alta_transaccionalidad_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)

datos_cupo_superados_limpio=datos_cupo_superados[['Documento']]
datos_cupo_superados_limpio['Evento']='Alerta cupo excedido'
datos_cupo_superados_limpio=datos_cupo_superados_limpio.drop_duplicates(subset=['Documento']).reset_index(drop=True)


print('Fin limpieza de datos MALLA')


datos_alertas_unificado=pd.concat([datos_registro_celular_similitud_limpio,
                                   datos_registro_correo_similitud_limpio,
                                   datos_tdc_condicion_similitud_limpio,
                                   datos_debito_reexpedicion_limpio,
                                   datos_tdc_reexpedicion_limpio,
                                   datos_creacion_actualizacion_malla_limpio,
                                   datos_disparidad_ip_limpio,
                                   datos_registros_atipicos_limpio,
                                   datos_transacciones_olvido_limpio,
                                   datos_transacciones_actualizacion_limpio,
                                   datos_transacciones_ingresos_tdc_limpio,
                                   datos_transacciones_inactividad_malla_limpio,
                                   datos_variacion_transaccional_limpio,
                                   datos_cliente_actualiza_limpio,
                                   datos_ip_multiples_usuarios_limpio,
                                   datos_flexi_salidas_limpio,
                                   datos_flexi_entradas_limpio,
                                   datos_correo_riesgo_limpio,
                                   datos_recuperacion_contraseña_multiple_limpio,
                                   datos_actualizacion_riesgo_limpio,
                                   datos_similitud_ip_creacion_limpio,
                                   datos_intentos_vinculacion_malla_limpio,
                                   datos_multiple_dispositivo_limpio,
                                   datos_intentos_evidente_malla_limpio,
                                   datos_disparidad_ciiu_limpio,
                                   datos_sospecha_transfiya_limpio,
                                   datos_creacion_enrolamiento_limpio,
                                   datos_alta_transaccionalidad_limpio,
                                   datos_cupo_superados_limpio])

datos_alertas_unificado['Cantidad de alertas']=1
minima_alerta_generada=pd.pivot_table(data=datos_alertas_unificado,values=['Cantidad de alertas'],index=['Documento'],columns=['Evento'],aggfunc=np.sum,fill_value=0).droplevel(0,axis=1).reset_index().rename_axis(None,axis=1)

# minima_alerta_generada['VerificacionFechaAlerta'] = dt.date.today()


minima_alerta_generada.to_sql('MallaAlertamientoCualitativa',
                               con=config_db_alerta[1],
                               if_exists='replace',
                               index=False,
                               schema='dbo')




tiempo_ejecucion_malla = round(float(format(time.time() - startTime2))/60,2)

tiempo_ejecucion_completo = round(float(format(time.time() - startTime0))/60,2)

#-------------------------------------------------------------------------------------------------#
#                                     ENVÍO DE RESULTADOS                                         #
#-------------------------------------------------------------------------------------------------#

# skipped your comments for readability
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

me = "jose.gomezv@bancofinandina.com"
my_password = r"mpwohrvzgbstkbvk"

# you=["jose.gomezv@bancofinandina.com" , "jesus.alvear@bancofinandina.com","astrid.bermudez@bancofinandina.com"]
you=["jose.gomezv@bancofinandina.com"]
msg = MIMEMultipart('alternative')
msg['Subject'] = "Actualización Alertamientos de FRAUDE"
msg['From'] = me
msg['To'] = ",".join(you)

html = """\
<html>
  <head></head>
  <body>
    <p> 📊 📈 Buen Día, el presente correo contiene la última fecha de la que se tienen registros en los alertamientos de fraude 📊 📈<br>
"""" ✅ RESULTADOS OBTENIDOS : 🕓 " +str(fecha_actual)+ """<br>
"""" ◾ AlertaAhorroRegistroCelularSimilitud : ▶ " +str(df.loc[0, 'última actualización'])+ """<br>
"""" ◾ AlertaAhorroRegistroCorreoSimilitud  : ▶ " +str(df.loc[1, 'última actualización'])+ """<br>
"""" ◾ AlertaTDCCondicionSimilitud : ▶ " +str(df.loc[2, 'última actualización'])+ """<br>
"""" ◾ AlertaTDCReexpedicion : ▶ " +str(df.loc[3, 'última actualización'])+ """<br>
"""" ◾ AlertaDebitoReexpedicion : ▶ " +str(df.loc[4, 'última actualización'])+ """<br>
"""" ◾ AlertaCreacionActualizacion : ▶ " +str(df.loc[5, 'última actualización'])+ """<br>
"""" ◾ AlertaDisparidadIP : ▶ " +str(df.loc[6, 'última actualización'])+ """<br>
"""" ◾ AlertaRegistrosAtipicos : ▶ " +str(df.loc[7, 'última actualización'])+ """<br>
"""" ◾ AlertaTransaccionesOlvido : ▶ " +str(df.loc[8, 'última actualización'])+ """<br>
"""" ◾ AlertaTransaccionesActualizacion : ▶ " +str(df.loc[9, 'última actualización'])+ """<br>
"""" ◾ AlertaTransaccionesIngresosTDC : ▶ " +str(df.loc[10, 'última actualización'])+ """<br>
"""" ◾ AlertaTransaccionesInactividad : ▶ " +str(df.loc[11, 'última actualización'])+ """<br>
"""" ◾ AlertaVariacionTransaccional : ▶ " +str(df.loc[12, 'última actualización'])+ """<br>
"""" ◾ AlertaClienteActualiza : ▶ " +str(df.loc[13, 'última actualización'])+ """<br>
"""" ◾ AlertaIPMultiplesUsuarios : ▶ " +str(df.loc[14, 'última actualización'])+ """<br>
"""" ◾ AlertaFlexiAltaSalidas : ▶ " +str(df.loc[15, 'última actualización'])+ """<br>
"""" ◾ AlertaFlexiAltasEntradas : ▶ " +str(df.loc[16, 'última actualización'])+ """<br>
"""" ◾ AlertaCorreoRiesgo : ▶ " +str(df.loc[17, 'última actualización'])+ """<br>
"""" ◾ AlertaRecuperacionContraseñaMultiple : ▶ " +str(df.loc[18, 'última actualización'])+ """<br>
"""" ◾ AlertaActualizacionRiesgo : ▶ " +str(df.loc[19, 'última actualización'])+ """<br>
"""" ◾ AlertaSimilitudIPCreacion : ▶ " +str(df.loc[20, 'última actualización'])+ """<br>
"""" ◾ AlertaIntentosVinculacion : ▶ " +str(df.loc[21, 'última actualización'])+ """<br>
"""" ◾ AlertaMultipleDispositivo : ▶ " +str(df.loc[22, 'última actualización'])+ """<br>
"""" ◾ AlertaIntentosEvidente : ▶ " +str(df.loc[23, 'última actualización'])+ """<br>
"""" ◾ AlertaDisparidadCIIU : ▶ " +str(df.loc[24, 'última actualización'])+ """<br>
"""" ◾ AlertaSospechaTransfiya : ▶ " +str(df.loc[25, 'última actualización'])+ """<br>
"""" ◾ AlertaCreacionEnrolamiento : ▶ " +str(df.loc[26, 'última actualización'])+ """<br>
"""" ◾ AlertaAltaTransaccionalidad : ▶   " +str(df.loc[27, 'última actualización'])+ """<br>
"""" ◾ AlertaCupoLimiteExcedido : ▶  " + str(len(alerta_cupo_excedido.axes[0])) + " " + "registros" + """<br>
"""" ✅ TIEMPO DE EJECUCIÓN ALERTAMIENTOS : 🕓  " + str(tiempo_ejecucion_codigo) + " " + "minutos"+ """<br>
"""" ✅ TIEMPO DE EJECUCIÓN MALLA 🚩🏁🚩 : 🕓  " + str(tiempo_ejecucion_malla) + " " + "minutos"+ """<br>
"""" Registros obtenidos MALLA : 🪪  " + str(len(minima_alerta_generada.axes[0])) + " " + "filas"+ " " + str(len(minima_alerta_generada.axes[1])) + " " + "columnas" """<br>
"""" ✅ TIEMPO DE EJECUCIÓN CÓDIGO COMPLETO 🆙 : 🕓  " + str(tiempo_ejecucion_completo) + " " + "minutos"+ """<br>
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

###### alerta TDC condicion similitud 

datos_titularidad_tdc=datos_tdc_activa_tarjeta[['Documento','Número de cuenta','Número de tarjeta','Tipo de beneficiario','Estado de la cuenta','Situación de tarjeta','Descripción del bloqueo','Descripción de baja','Fecha de alta de la tarjeta','Fecha de baja de la tarjeta']]

clientes_doble_titularidad=pd.DataFrame(datos_titularidad_tdc.groupby(by=['Documento']).apply(lambda x: ((x['Tipo de beneficiario']=='TI').any())and((x['Tipo de beneficiario']=='BE').any())),columns=['Alerta doble titularidad']).reset_index()

alerta_doble_titularidad=pd.merge(datos_titularidad_tdc,clientes_doble_titularidad.loc[clientes_doble_titularidad['Alerta doble titularidad']==True],on=['Documento'],how='inner')

partea=alerta_doble_titularidad[alerta_doble_titularidad['Tipo de beneficiario']=="TI"].loc[:, ['Documento', 'Fecha de alta de la tarjeta']] 

parteb=alerta_doble_titularidad[alerta_doble_titularidad['Tipo de beneficiario']=="BE"].loc[:, ['Documento', 'Fecha de alta de la tarjeta']]

cruce=pd.merge(partea,parteb,on=['Documento'],how='left').rename(columns={'Fecha de alta de la tarjeta_x':'Fecha alta tarjeta TI','Fecha de alta de la tarjeta_y':'Fecha alta tarjeta BE'})

cruce['llave']= np.where(cruce['Fecha alta tarjeta TI'] < cruce['Fecha alta tarjeta BE'],1,0)


res=cruce[cruce['llave']==1]



### alerta cliente actualiza 

with  consulta as (SELECT DocumentoCliente,count(*)as tx, max(FechaTransaccion) as ultimatx
  FROM [Productos y transaccionalidad].[dbo].[ConsolidadoTransacciones]
  where CodigoTransaccional2 = '325'
  group by DocumentoCliente)

  select * from consulta
  where ultimatx <= GETDATE()-30
  order by ultimatx desc
  
  
  
   SELECT DocumentoCliente,count(*)as tx, max(FechaTransaccion) as ultimatx
  FROM [Productos y transaccionalidad].[dbo].[ConsolidadoTransacciones]
  where [CodigoTransaccional2] = '325' 
  group by DocumentoCliente 
  order by ultimatx desc




query_datos_transacciones2='''SELECT 
 [DocumentoCliente] as 'Documento',
 count(*)as tx,
 max(FechaTransaccion) as ultimatx,
 avg(MontoTransacción) as promediotx
FROM {tabla_datos}
WHERE [CodigoTransaccional2] = '325' 
GROUP BY [DocumentoCliente]     '''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoTransacciones]')
datos_transacciones_limpio2=load_data(query_datos_transacciones2,config_db_riesgo[1],model_logger=logger)







