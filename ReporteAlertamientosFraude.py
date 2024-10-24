# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:10:12 2023

@author: josgom
"""


## REPORTE ALERTAMIENTOS DE FRAUDE 



# CARGUE DE ALERTAMIENTOS 



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
    password='GrandeLeo2023#'
    conn_riesgo = connect_to_database(server_riesgo, database_riesgo, user, password)
  
    return conn_riesgo



## CARGUE ALERTAMIENTOS  

AlertaAhorroRegistroCelularSimilitud='''Select*
from [AlertasFraude].[dbo].[AlertaAhorroRegistroCelularSimilitud]'''
AlertaAhorroRegistroCelularSimilitud = pd.read_sql(AlertaAhorroRegistroCelularSimilitud,conexion_fabogriesgo())
AlertaAhorroRegistroCelularSimilitud['Documento']=AlertaAhorroRegistroCelularSimilitud['Documento'].astype(str).str.strip()
AlertaAhorroRegistroCelularSimilitud = AlertaAhorroRegistroCelularSimilitud.rename(columns={'Documento':'Cliente Asociado'})


AlertaAhorroRegistroCorreoSimilitud='''Select*
from [AlertasFraude].[dbo].[AlertaAhorroRegistroCorreoSimilitud]'''
AlertaAhorroRegistroCorreoSimilitud = pd.read_sql(AlertaAhorroRegistroCorreoSimilitud,conexion_fabogriesgo())
AlertaAhorroRegistroCorreoSimilitud['Documento']=AlertaAhorroRegistroCorreoSimilitud['Documento'].astype(str).str.strip()
AlertaAhorroRegistroCorreoSimilitud = AlertaAhorroRegistroCorreoSimilitud.rename(columns={'Documento':'Cliente Asociado'})


AlertaTDCCondicionSimilitud='''Select*
from [AlertasFraude].[dbo].[AlertaTDCCondicionSimilitud_historico]'''
AlertaTDCCondicionSimilitud = pd.read_sql(AlertaTDCCondicionSimilitud,conexion_fabogriesgo())
AlertaTDCCondicionSimilitud['DocumentoTI']=AlertaTDCCondicionSimilitud['DocumentoTI'].astype(str).str.strip()
AlertaTDCCondicionSimilitud = AlertaTDCCondicionSimilitud.rename(columns={'DocumentoTI':'Cliente Asociado'})


AlertaTDCReexpedicion='''Select*
from [AlertasFraude].[dbo].[AlertaTDCReexpedicion]'''
AlertaTDCReexpedicion = pd.read_sql(AlertaTDCReexpedicion,conexion_fabogriesgo())
AlertaTDCReexpedicion['Documento']=AlertaTDCReexpedicion['Documento'].astype(str).str.strip()
AlertaTDCReexpedicion = AlertaTDCReexpedicion.rename(columns={'Documento':'Cliente Asociado'})



AlertaDebitoReexpedicion='''Select*
from [AlertasFraude].[dbo].[AlertaDebitoReexpedicion_historico]'''
AlertaDebitoReexpedicion = pd.read_sql(AlertaDebitoReexpedicion,conexion_fabogriesgo())
AlertaDebitoReexpedicion['Documento']=AlertaDebitoReexpedicion['Documento'].astype(str).str.strip()
AlertaDebitoReexpedicion = AlertaDebitoReexpedicion.rename(columns={'Documento':'Cliente Asociado'})




AlertaCreacionActualizacion='''Select*
from [AlertasFraude].[dbo].[AlertaCreacionActualizacion]'''
AlertaCreacionActualizacion = pd.read_sql(AlertaCreacionActualizacion,conexion_fabogriesgo())
AlertaCreacionActualizacion['Documento']=AlertaCreacionActualizacion['Documento'].astype(str).str.strip()
AlertaCreacionActualizacion = AlertaCreacionActualizacion.rename(columns={'Documento':'Cliente Asociado'})



AlertaDisparidadIP='''Select*
from [AlertasFraude].[dbo].[AlertaDisparidadIP]'''
AlertaDisparidadIP = pd.read_sql(AlertaDisparidadIP,conexion_fabogriesgo())
AlertaDisparidadIP['Documento']=AlertaDisparidadIP['Documento'].astype(str).str.strip()
AlertaDisparidadIP = AlertaDisparidadIP.rename(columns={'Documento':'Cliente Asociado'})




AlertaRegistrosAtipicos='''Select*
from [AlertasFraude].[dbo].[AlertaRegistrosAtipicos]'''
AlertaRegistrosAtipicos = pd.read_sql(AlertaRegistrosAtipicos,conexion_fabogriesgo())
AlertaRegistrosAtipicos['Documento']=AlertaRegistrosAtipicos['Documento'].astype(str).str.strip()
AlertaRegistrosAtipicos = AlertaRegistrosAtipicos.rename(columns={'Documento':'Cliente Asociado'})




AlertaTransaccionesOlvido='''Select*
from [AlertasFraude].[dbo].[AlertaTransaccionesOlvido_historico]'''
AlertaTransaccionesOlvido = pd.read_sql(AlertaTransaccionesOlvido,conexion_fabogriesgo())
AlertaTransaccionesOlvido['Documento']=AlertaTransaccionesOlvido['Documento'].astype(str).str.strip()
AlertaTransaccionesOlvido = AlertaTransaccionesOlvido.rename(columns={'Documento':'Cliente Asociado'})




AlertaTransaccionesActualizacion='''Select*
from [AlertasFraude].[dbo].[AlertaTransaccionesActualizacion_historico]'''
AlertaTransaccionesActualizacion = pd.read_sql(AlertaTransaccionesActualizacion,conexion_fabogriesgo())
AlertaTransaccionesActualizacion['Documento']=AlertaTransaccionesActualizacion['Documento'].astype(str).str.strip()
AlertaTransaccionesActualizacion = AlertaTransaccionesActualizacion.rename(columns={'Documento':'Cliente Asociado'})





AlertaTransaccionesIngresosTDC='''Select*
from [AlertasFraude].[dbo].[AlertaTransaccionesIngresosTDC_historico]'''
AlertaTransaccionesIngresosTDC = pd.read_sql(AlertaTransaccionesIngresosTDC,conexion_fabogriesgo())
AlertaTransaccionesIngresosTDC['Documento']=AlertaTransaccionesIngresosTDC['Documento'].astype(str).str.strip()
AlertaTransaccionesIngresosTDC = AlertaTransaccionesIngresosTDC.rename(columns={'Documento':'Cliente Asociado'})







AlertaTransaccionesInactividad='''Select*
from [AlertasFraude].[dbo].[AlertaTransaccionesInactividad]'''
AlertaTransaccionesInactividad = pd.read_sql(AlertaTransaccionesInactividad,conexion_fabogriesgo())
AlertaTransaccionesInactividad['Documento']=AlertaTransaccionesInactividad['Documento'].astype(str).str.strip()
AlertaTransaccionesInactividad = AlertaTransaccionesInactividad.rename(columns={'Documento':'Cliente Asociado'})




AlertaVariacionTransaccional='''Select*
from [AlertasFraude].[dbo].[AlertaVariacionTransaccional]'''
AlertaVariacionTransaccional = pd.read_sql(AlertaVariacionTransaccional,conexion_fabogriesgo())
AlertaVariacionTransaccional['Documento']=AlertaVariacionTransaccional['Documento'].astype(str).str.strip()
AlertaVariacionTransaccional = AlertaVariacionTransaccional.rename(columns={'Documento':'Cliente Asociado'})




AlertaClienteActualiza='''Select*
from [AlertasFraude].[dbo].[AlertaClienteActualiza_historico]'''
AlertaClienteActualiza = pd.read_sql(AlertaClienteActualiza,conexion_fabogriesgo())
AlertaClienteActualiza['Documento']=AlertaClienteActualiza['Documento'].astype(str).str.strip()
AlertaClienteActualiza = AlertaClienteActualiza.rename(columns={'Documento':'Cliente Asociado'})



AlertaIPMultiplesUsuarios='''Select*
from [AlertasFraude].[dbo].[AlertaIPMultiplesUsuarios]'''
AlertaIPMultiplesUsuarios = pd.read_sql(AlertaIPMultiplesUsuarios,conexion_fabogriesgo())
AlertaIPMultiplesUsuarios['Documento']=AlertaIPMultiplesUsuarios['Documento'].astype(str).str.strip()
AlertaIPMultiplesUsuarios = AlertaIPMultiplesUsuarios.rename(columns={'Documento':'Cliente Asociado'})




AlertaFlexiAltaSalidas='''Select*
from [AlertasFraude].[dbo].[AlertaFlexiAltasSalidas_historico]'''
AlertaFlexiAltaSalidas = pd.read_sql(AlertaFlexiAltaSalidas,conexion_fabogriesgo())
AlertaFlexiAltaSalidas['Documento']=AlertaFlexiAltaSalidas['Documento'].astype(str).str.strip()
AlertaFlexiAltaSalidas = AlertaFlexiAltaSalidas.rename(columns={'Documento':'Cliente Asociado'})



AlertaFlexiAltasEntradas='''Select*
from [AlertasFraude].[dbo].[AlertaFlexiAltasEntradas_historico]'''
AlertaFlexiAltasEntradas = pd.read_sql(AlertaFlexiAltasEntradas,conexion_fabogriesgo())
AlertaFlexiAltasEntradas['Documento']=AlertaFlexiAltasEntradas['Documento'].astype(str).str.strip()
AlertaFlexiAltasEntradas = AlertaFlexiAltasEntradas.rename(columns={'Documento':'Cliente Asociado'})









AlertaCorreoRiesgo='''Select*
from [AlertasFraude].[dbo].[AlertaCorreoRiesgo]'''
AlertaCorreoRiesgo = pd.read_sql(AlertaCorreoRiesgo,conexion_fabogriesgo())
AlertaCorreoRiesgo['Documento']=AlertaCorreoRiesgo['Documento'].astype(str).str.strip()
AlertaCorreoRiesgo = AlertaCorreoRiesgo.rename(columns={'Documento':'Cliente Asociado'})








AlertaRecuperacionContraseñaMultiple='''Select*
from [AlertasFraude].[dbo].[AlertaRecuperacionContraseñaMultiple]'''
AlertaRecuperacionContraseñaMultiple = pd.read_sql(AlertaRecuperacionContraseñaMultiple,conexion_fabogriesgo())
AlertaRecuperacionContraseñaMultiple['Documento']=AlertaRecuperacionContraseñaMultiple['Documento'].astype(str).str.strip()
AlertaRecuperacionContraseñaMultiple = AlertaRecuperacionContraseñaMultiple.rename(columns={'Documento':'Cliente Asociado'})







AlertaActualizacionRiesgo='''Select*
from [AlertasFraude].[dbo].[AlertaActualizacionRiesgo]'''
AlertaActualizacionRiesgo = pd.read_sql(AlertaActualizacionRiesgo,conexion_fabogriesgo())
AlertaActualizacionRiesgo['Documento']=AlertaActualizacionRiesgo['Documento'].astype(str).str.strip()
AlertaActualizacionRiesgo = AlertaActualizacionRiesgo.rename(columns={'Documento':'Cliente Asociado'})






AlertaSimilitudIPCreacion='''Select*
from [AlertasFraude].[dbo].[AlertaSimilitudIPCreacion_historico]'''
AlertaSimilitudIPCreacion = pd.read_sql(AlertaSimilitudIPCreacion,conexion_fabogriesgo())
AlertaSimilitudIPCreacion['Documento']=AlertaSimilitudIPCreacion['Documento'].astype(str).str.strip()
AlertaSimilitudIPCreacion = AlertaSimilitudIPCreacion.rename(columns={'Documento':'Cliente Asociado'})




AlertaIntentosVinculacion='''Select*
from [AlertasFraude].[dbo].[AlertaIntentosVinculacion]'''
AlertaIntentosVinculacion = pd.read_sql(AlertaIntentosVinculacion,conexion_fabogriesgo())
AlertaIntentosVinculacion['Documento']=AlertaIntentosVinculacion['Documento'].astype(str).str.strip()
AlertaIntentosVinculacion = AlertaIntentosVinculacion.rename(columns={'Documento':'Cliente Asociado'})



AlertaMultipleDispositivo='''Select*
from [AlertasFraude].[dbo].[AlertaMultipleDispositivo_historico]'''
AlertaMultipleDispositivo = pd.read_sql(AlertaMultipleDispositivo,conexion_fabogriesgo())
AlertaMultipleDispositivo['Documento']=AlertaMultipleDispositivo['Documento'].astype(str).str.strip()
AlertaMultipleDispositivo = AlertaMultipleDispositivo.rename(columns={'Documento':'Cliente Asociado'})




AlertaIntentosEvidente='''Select*
from [AlertasFraude].[dbo].[AlertaIntentosEvidente_historico]'''
AlertaIntentosEvidente = pd.read_sql(AlertaIntentosEvidente,conexion_fabogriesgo())
AlertaIntentosEvidente['Documento']=AlertaIntentosEvidente['Documento'].astype(str).str.strip()
AlertaIntentosEvidente = AlertaIntentosEvidente.rename(columns={'Documento':'Cliente Asociado'})




AlertaDisparidadCIIU='''Select*
from [AlertasFraude].[dbo].[AlertaDisparidadCIIU]'''
AlertaDisparidadCIIU = pd.read_sql(AlertaDisparidadCIIU,conexion_fabogriesgo())
AlertaDisparidadCIIU['Documento']=AlertaDisparidadCIIU['Documento'].astype(str).str.strip()
AlertaDisparidadCIIU = AlertaDisparidadCIIU.rename(columns={'Documento':'Cliente Asociado'})









AlertaSospechaTransfiya='''Select*
from [AlertasFraude].[dbo].[AlertaSospechaTransfiya]'''
AlertaSospechaTransfiya = pd.read_sql(AlertaSospechaTransfiya,conexion_fabogriesgo())
AlertaSospechaTransfiya['Documento']=AlertaSospechaTransfiya['Documento'].astype(str).str.strip()
AlertaSospechaTransfiya = AlertaSospechaTransfiya.rename(columns={'Documento':'Cliente Asociado'})








AlertaCreacionEnrolamiento='''Select*
from [AlertasFraude].[dbo].[AlertaCreacionEnrolamiento]'''
AlertaCreacionEnrolamiento = pd.read_sql(AlertaCreacionEnrolamiento,conexion_fabogriesgo())
AlertaCreacionEnrolamiento['Documento']=AlertaCreacionEnrolamiento['Documento'].astype(str).str.strip()
AlertaCreacionEnrolamiento = AlertaCreacionEnrolamiento.rename(columns={'Documento':'Cliente Asociado'})





AlertaAltaTransaccionalidad='''Select*
from [AlertasFraude].[dbo].[AlertaAltaTransaccionalidad]'''
AlertaAltaTransaccionalidad = pd.read_sql(AlertaAltaTransaccionalidad,conexion_fabogriesgo())
AlertaAltaTransaccionalidad['Documento']=AlertaAltaTransaccionalidad['Documento'].astype(str).str.strip()
AlertaAltaTransaccionalidad = AlertaAltaTransaccionalidad.rename(columns={'Documento':'Cliente Asociado'})




AlertaActualizacionRegistroDispositivo='''Select*
from [AlertasFraude].[dbo].[AlertaActualizacionRegistroDispositivo_historico]'''
AlertaActualizacionRegistroDispositivo = pd.read_sql(AlertaActualizacionRegistroDispositivo,conexion_fabogriesgo())
AlertaActualizacionRegistroDispositivo['Documento']=AlertaActualizacionRegistroDispositivo['Documento'].astype(str).str.strip()
AlertaActualizacionRegistroDispositivo = AlertaActualizacionRegistroDispositivo.rename(columns={'Documento':'Cliente Asociado'})




AlertaRegistroEquipoRecuperacionUsuario='''Select*
from [AlertasFraude].[dbo].[AlertaRegistroEquipoRecuperacionUsuario]'''
AlertaRegistroEquipoRecuperacionUsuario = pd.read_sql(AlertaRegistroEquipoRecuperacionUsuario,conexion_fabogriesgo())
AlertaRegistroEquipoRecuperacionUsuario['Documento']=AlertaRegistroEquipoRecuperacionUsuario['Documento'].astype(str).str.strip()
AlertaRegistroEquipoRecuperacionUsuario = AlertaRegistroEquipoRecuperacionUsuario.rename(columns={'Documento':'Cliente Asociado'})




AlertaSaldoFlexidigitalAhorro='''Select*
from [AlertasFraude].[dbo].[AlertaSaldoFlexidigitalAhorroHistorico]'''
AlertaSaldoFlexidigitalAhorro = pd.read_sql(AlertaSaldoFlexidigitalAhorro,conexion_fabogriesgo())
AlertaSaldoFlexidigitalAhorro['Documento']=AlertaSaldoFlexidigitalAhorro['Documento'].astype(str).str.strip()
AlertaSaldoFlexidigitalAhorro = AlertaSaldoFlexidigitalAhorro.rename(columns={'Documento':'Cliente Asociado'})






EntradasFlexiDigitalplus8smmlv='''Select*
from [AlertasFraude].[dbo].[EntradasFlexiDigitalplus8smmlvHistorico]'''
EntradasFlexiDigitalplus8smmlv = pd.read_sql(EntradasFlexiDigitalplus8smmlv,conexion_fabogriesgo())
EntradasFlexiDigitalplus8smmlv['Documento']=EntradasFlexiDigitalplus8smmlv['Documento'].astype(str).str.strip()
EntradasFlexiDigitalplus8smmlv = EntradasFlexiDigitalplus8smmlv.rename(columns={'Documento':'Cliente Asociado'})



AlertaCupoLimiteExcedido='''Select*
from [AlertasFraude].[dbo].[AlertaCupoLimiteExcedido_historico]'''
AlertaCupoLimiteExcedido = pd.read_sql(AlertaCupoLimiteExcedido,conexion_fabogriesgo())
AlertaCupoLimiteExcedido['Documento']=AlertaCupoLimiteExcedido['Documento'].astype(str).str.strip()
AlertaCupoLimiteExcedido = AlertaCupoLimiteExcedido.rename(columns={'Documento':'Cliente Asociado'})



AlertamientoCupoSuperado='''Select*
from [AlertasFraude].[dbo].[AlertamientoCupoSuperado]'''
AlertamientoCupoSuperado = pd.read_sql(AlertamientoCupoSuperado,conexion_fabogriesgo())
AlertamientoCupoSuperado['DocumentoCliente']=AlertamientoCupoSuperado['DocumentoCliente'].astype(str).str.strip()
AlertamientoCupoSuperado = AlertamientoCupoSuperado.rename(columns={'DocumentoCliente':'Cliente Asociado'})



# BASE DE FRAUDES 

fraudes='''Select*
from Fabogcubox.[Finandina_Cartera].[dbo].[00 Fraudes_Bco_FA]'''
fraudes = pd.read_sql(fraudes,conexion_fabogriesgo())



# fraudes ultimos 3 meses


# fraudes.columns

# Supongamos que tenemos un DataFrame llamado df con una columna 'fecha' que contiene las fechas de los registros

# Convertir la columna 'fecha' al tipo de dato datetime si no está en ese formato
fraudes['Fecha'] = pd.to_datetime(fraudes['Fecha'])

# Obtener la fecha actual
fecha_actual = datetime.now()

# Calcular la fecha límite de tres meses atrás
fecha_limite = fecha_actual - timedelta(days=90)

# Filtrar los registros que están dentro del rango de los últimos tres meses
df_filtrado = fraudes[fraudes['Fecha'] >= fecha_limite]



fraudes=df_filtrado




## REVISIÓN CON BASE FRAUDES 

fraudes['Cliente Asociado']=fraudes['Cliente Asociado'].astype(str).str.strip()

c1 = pd.merge(AlertaAhorroRegistroCelularSimilitud,fraudes,on='Cliente Asociado',how='inner')

c2 = pd.merge(AlertaAhorroRegistroCorreoSimilitud,fraudes,on='Cliente Asociado',how='inner')

c3 = pd.merge(AlertaTDCCondicionSimilitud,fraudes,on='Cliente Asociado',how='inner')

c4 = pd.merge(AlertaTDCReexpedicion,fraudes,on='Cliente Asociado',how='inner')

c5 = pd.merge(AlertaDebitoReexpedicion,fraudes,on='Cliente Asociado',how='inner')

c6 = pd.merge(AlertaCreacionActualizacion,fraudes,on='Cliente Asociado',how='inner')

c7 = pd.merge(AlertaDisparidadIP,fraudes,on='Cliente Asociado',how='inner')

c8 = pd.merge(AlertaRegistrosAtipicos,fraudes,on='Cliente Asociado',how='inner')

c9 = pd.merge(AlertaTransaccionesOlvido,fraudes,on='Cliente Asociado',how='inner')

c10 = pd.merge(AlertaTransaccionesActualizacion,fraudes,on='Cliente Asociado',how='inner')

c11 = pd.merge(AlertaTransaccionesIngresosTDC,fraudes,on='Cliente Asociado',how='inner')

c12 = pd.merge(AlertaTransaccionesInactividad,fraudes,on='Cliente Asociado',how='inner')

c13 = pd.merge(AlertaVariacionTransaccional,fraudes,on='Cliente Asociado',how='inner')

c14 = pd.merge(AlertaClienteActualiza,fraudes,on='Cliente Asociado',how='inner')


c15 = pd.merge(AlertaIPMultiplesUsuarios,fraudes,on='Cliente Asociado',how='inner')

c16 = pd.merge(AlertaFlexiAltaSalidas,fraudes,on='Cliente Asociado',how='inner')

c17 = pd.merge(AlertaFlexiAltasEntradas,fraudes,on='Cliente Asociado',how='inner')

c18 = pd.merge(AlertaCorreoRiesgo,fraudes,on='Cliente Asociado',how='inner')

c19 = pd.merge(AlertaRecuperacionContraseñaMultiple,fraudes,on='Cliente Asociado',how='inner')

c20 = pd.merge(AlertaActualizacionRiesgo,fraudes,on='Cliente Asociado',how='inner')

c21 = pd.merge(AlertaSimilitudIPCreacion,fraudes,on='Cliente Asociado',how='inner')

c22 = pd.merge(AlertaIntentosVinculacion,fraudes,on='Cliente Asociado',how='inner')

c23 = pd.merge(AlertaMultipleDispositivo,fraudes,on='Cliente Asociado',how='inner')

c24 = pd.merge(AlertaIntentosEvidente,fraudes,on='Cliente Asociado',how='inner')

c25 = pd.merge(AlertaDisparidadCIIU,fraudes,on='Cliente Asociado',how='inner')

c26 = pd.merge(AlertaSospechaTransfiya,fraudes,on='Cliente Asociado',how='inner')

c27 = pd.merge(AlertaCreacionEnrolamiento,fraudes,on='Cliente Asociado',how='inner')

c28 = pd.merge(AlertaAltaTransaccionalidad,fraudes,on='Cliente Asociado',how='inner')

c29 = pd.merge(AlertaActualizacionRegistroDispositivo,fraudes,on='Cliente Asociado',how='inner')

c30 = pd.merge(AlertaRegistroEquipoRecuperacionUsuario,fraudes,on='Cliente Asociado',how='inner')

c31 = pd.merge(AlertaSaldoFlexidigitalAhorro,fraudes,on='Cliente Asociado',how='inner')

c32 = pd.merge(EntradasFlexiDigitalplus8smmlv,fraudes,on='Cliente Asociado',how='inner')

c33 = pd.merge(AlertaCupoLimiteExcedido,fraudes,on='Cliente Asociado',how='inner')


reporte = pd.DataFrame({'Alertamiento': ['AlertaAhorroRegistroCelularSimilitud',
                            'AlertaAhorroRegistroCorreoSimilitud',
                            'AlertaTDCCondicionSimilitud',
                            'AlertaTDCReexpedicion',
                            'AlertaDebitoReexpedicion',
                            'AlertaCreacionActualizacion',
                            'AlertaDisparidadIP',
                            'AlertaRegistrosAtipicos',
                            'AlertaTransaccionesOlvido',
                            'AlertaTransaccionesActualizacion',
                            'AlertaTransaccionesIngresosTDC',
                            'AlertaTransaccionesInactividad',
                            'AlertaVariacionTransaccional',
                            'AlertaClienteActualiza',
                            'AlertaIPMultiplesUsuarios',
                            'AlertaFlexiAltaSalidas',
                            'AlertaFlexiAltasEntradas',
                            'AlertaCorreoRiesgo',
                            'AlertaRecuperacionContraseñaMultiple',
                            'AlertaActualizacionRiesgo',
                            'AlertaSimilitudIPCreacion',
                            'AlertaIntentosVinculacion',
                            'AlertaMultipleDispositivo',
                            'AlertaIntentosEvidente',
                            'AlertaDisparidadCIIU',
                            'AlertaSospechaTransfiya',
                            'AlertaCreacionEnrolamiento',
                            'AlertaAltaTransaccionalidad',
                            'AlertaActualizacionRegistroDispositivo',
                            'AlertaRegistroEquipoRecuperacionUsuario',
                            'AlertaSaldoFlexidigitalAhorro',
                            'EntradasFlexiDigitalplus8smmlv',
                            'AlertaCupoLimiteExcedido'],
           'Cantidad Alertamientos': [len(AlertaAhorroRegistroCelularSimilitud),
                                       len(AlertaAhorroRegistroCorreoSimilitud),
                                       len(AlertaTDCCondicionSimilitud),
                                       len(AlertaTDCReexpedicion),
                                       len(AlertaDebitoReexpedicion),
                                       len(AlertaCreacionActualizacion),
                                       len(AlertaDisparidadIP),
                                       len(AlertaRegistrosAtipicos),
                                       len(AlertaTransaccionesOlvido),
                                       len(AlertaTransaccionesActualizacion),
                                       len(AlertaTransaccionesIngresosTDC),
                                       len(AlertaTransaccionesInactividad),
                                       len(AlertaVariacionTransaccional),
                                       len(AlertaClienteActualiza),
                                       len(AlertaIPMultiplesUsuarios),
                                       len(AlertaFlexiAltaSalidas),
                                       len(AlertaFlexiAltasEntradas),
                                       len(AlertaCorreoRiesgo),
                                       len(AlertaRecuperacionContraseñaMultiple),
                                       len(AlertaActualizacionRiesgo),
                                       len(AlertaSimilitudIPCreacion),
                                       len(AlertaIntentosVinculacion),
                                       len(AlertaMultipleDispositivo),
                                       len(AlertaIntentosEvidente),
                                       len(AlertaDisparidadCIIU),
                                       len(AlertaSospechaTransfiya),
                                       len(AlertaCreacionEnrolamiento),
                                       len(AlertaAltaTransaccionalidad),
                                       len(AlertaActualizacionRegistroDispositivo),
                                       len(AlertaRegistroEquipoRecuperacionUsuario),
                                       len(AlertaSaldoFlexidigitalAhorro),
                                       len(EntradasFlexiDigitalplus8smmlv),
                                       len(AlertaCupoLimiteExcedido)],
           'Validacion Fraudes': [len(c1),
                                   len(c2),
                                   len(c3),
                                   len(c4),
                                   len(c5),
                                   len(c6),
                                   len(c7),
                                   len(c8),
                                   len(c9),
                                   len(c10),
                                   len(c11),
                                   len(c12),
                                   len(c13),
                                   len(c14),
                                   len(c15),
                                   len(c16),
                                   len(c17),
                                   len(c18),
                                   len(c19),
                                   len(c20),
                                   len(c21),
                                   len(c22),
                                   len(c23),
                                   len(c24),
                                   len(c25),
                                   len(c26),
                                   len(c27),
                                   len(c28),
                                   len(c29),
                                   len(c30),
                                   len(c31),
                                   len(c32),
                                   len(c33)]})

reporte['Efectividad'] = reporte['Validacion Fraudes']/reporte['Cantidad Alertamientos']

reporte['Observacion'] = np.where(reporte['Cantidad Alertamientos']>=100,'Sobrealertamiento','NO Sobrealertamiento')


### entraga de reporte 



import pandas as pd

# Crear un DataFrame de ejemplo
data = {'Nombre': ['Juan', 'María', 'Pedro'],
        'Edad': [25, 30, 35],
        'Ciudad': ['Madrid', 'Barcelona', 'Sevilla']}

df = pd.DataFrame(data)

# Exportar el DataFrame a un archivo Excel
reporte.to_excel('//192.168.60.149/desarrollo y ciencia de datos/024_Jose_Gomez/reporte.xlsx', index=False)







################################### reporte por clientes unicos 



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
    password='GrandeLeo2023#'
    conn_riesgo = connect_to_database(server_riesgo, database_riesgo, user, password)
  
    return conn_riesgo



## CARGUE ALERTAMIENTOS  

AlertaAhorroRegistroCelularSimilitud='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaAhorroRegistroCelularSimilitud]'''
AlertaAhorroRegistroCelularSimilitud = pd.read_sql(AlertaAhorroRegistroCelularSimilitud,conexion_fabogriesgo())
AlertaAhorroRegistroCelularSimilitud['Documento']=AlertaAhorroRegistroCelularSimilitud['Documento'].astype(str).str.strip()
AlertaAhorroRegistroCelularSimilitud = AlertaAhorroRegistroCelularSimilitud.rename(columns={'Documento':'Cliente Asociado'})


AlertaAhorroRegistroCorreoSimilitud='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaAhorroRegistroCorreoSimilitud]'''
AlertaAhorroRegistroCorreoSimilitud = pd.read_sql(AlertaAhorroRegistroCorreoSimilitud,conexion_fabogriesgo())
AlertaAhorroRegistroCorreoSimilitud['Documento']=AlertaAhorroRegistroCorreoSimilitud['Documento'].astype(str).str.strip()
AlertaAhorroRegistroCorreoSimilitud = AlertaAhorroRegistroCorreoSimilitud.rename(columns={'Documento':'Cliente Asociado'})


AlertaTDCCondicionSimilitud='''Select distinct DocumentoTI
from [AlertasFraude].[dbo].[AlertaTDCCondicionSimilitud_historico]'''
AlertaTDCCondicionSimilitud = pd.read_sql(AlertaTDCCondicionSimilitud,conexion_fabogriesgo())
AlertaTDCCondicionSimilitud['DocumentoTI']=AlertaTDCCondicionSimilitud['DocumentoTI'].astype(str).str.strip()
AlertaTDCCondicionSimilitud = AlertaTDCCondicionSimilitud.rename(columns={'DocumentoTI':'Cliente Asociado'})


AlertaTDCReexpedicion='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaTDCReexpedicion]'''
AlertaTDCReexpedicion = pd.read_sql(AlertaTDCReexpedicion,conexion_fabogriesgo())
AlertaTDCReexpedicion['Documento']=AlertaTDCReexpedicion['Documento'].astype(str).str.strip()
AlertaTDCReexpedicion = AlertaTDCReexpedicion.rename(columns={'Documento':'Cliente Asociado'})



AlertaDebitoReexpedicion='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaDebitoReexpedicion_historico]'''
AlertaDebitoReexpedicion = pd.read_sql(AlertaDebitoReexpedicion,conexion_fabogriesgo())
AlertaDebitoReexpedicion['Documento']=AlertaDebitoReexpedicion['Documento'].astype(str).str.strip()
AlertaDebitoReexpedicion = AlertaDebitoReexpedicion.rename(columns={'Documento':'Cliente Asociado'})




AlertaCreacionActualizacion='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaCreacionActualizacion]'''
AlertaCreacionActualizacion = pd.read_sql(AlertaCreacionActualizacion,conexion_fabogriesgo())
AlertaCreacionActualizacion['Documento']=AlertaCreacionActualizacion['Documento'].astype(str).str.strip()
AlertaCreacionActualizacion = AlertaCreacionActualizacion.rename(columns={'Documento':'Cliente Asociado'})



AlertaDisparidadIP='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaDisparidadIP]'''
AlertaDisparidadIP = pd.read_sql(AlertaDisparidadIP,conexion_fabogriesgo())
AlertaDisparidadIP['Documento']=AlertaDisparidadIP['Documento'].astype(str).str.strip()
AlertaDisparidadIP = AlertaDisparidadIP.rename(columns={'Documento':'Cliente Asociado'})




AlertaRegistrosAtipicos='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaRegistrosAtipicos]'''
AlertaRegistrosAtipicos = pd.read_sql(AlertaRegistrosAtipicos,conexion_fabogriesgo())
AlertaRegistrosAtipicos['Documento']=AlertaRegistrosAtipicos['Documento'].astype(str).str.strip()
AlertaRegistrosAtipicos = AlertaRegistrosAtipicos.rename(columns={'Documento':'Cliente Asociado'})




AlertaTransaccionesOlvido='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaTransaccionesOlvido_historico]'''
AlertaTransaccionesOlvido = pd.read_sql(AlertaTransaccionesOlvido,conexion_fabogriesgo())
AlertaTransaccionesOlvido['Documento']=AlertaTransaccionesOlvido['Documento'].astype(str).str.strip()
AlertaTransaccionesOlvido = AlertaTransaccionesOlvido.rename(columns={'Documento':'Cliente Asociado'})




AlertaTransaccionesActualizacion='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaTransaccionesActualizacion_historico]'''
AlertaTransaccionesActualizacion = pd.read_sql(AlertaTransaccionesActualizacion,conexion_fabogriesgo())
AlertaTransaccionesActualizacion['Documento']=AlertaTransaccionesActualizacion['Documento'].astype(str).str.strip()
AlertaTransaccionesActualizacion = AlertaTransaccionesActualizacion.rename(columns={'Documento':'Cliente Asociado'})





AlertaTransaccionesIngresosTDC='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaTransaccionesIngresosTDC_historico]'''
AlertaTransaccionesIngresosTDC = pd.read_sql(AlertaTransaccionesIngresosTDC,conexion_fabogriesgo())
AlertaTransaccionesIngresosTDC['Documento']=AlertaTransaccionesIngresosTDC['Documento'].astype(str).str.strip()
AlertaTransaccionesIngresosTDC = AlertaTransaccionesIngresosTDC.rename(columns={'Documento':'Cliente Asociado'})







AlertaTransaccionesInactividad='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaTransaccionesInactividad]'''
AlertaTransaccionesInactividad = pd.read_sql(AlertaTransaccionesInactividad,conexion_fabogriesgo())
AlertaTransaccionesInactividad['Documento']=AlertaTransaccionesInactividad['Documento'].astype(str).str.strip()
AlertaTransaccionesInactividad = AlertaTransaccionesInactividad.rename(columns={'Documento':'Cliente Asociado'})




AlertaVariacionTransaccional='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaVariacionTransaccional]'''
AlertaVariacionTransaccional = pd.read_sql(AlertaVariacionTransaccional,conexion_fabogriesgo())
AlertaVariacionTransaccional['Documento']=AlertaVariacionTransaccional['Documento'].astype(str).str.strip()
AlertaVariacionTransaccional = AlertaVariacionTransaccional.rename(columns={'Documento':'Cliente Asociado'})




AlertaClienteActualiza='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaClienteActualiza_historico]'''
AlertaClienteActualiza = pd.read_sql(AlertaClienteActualiza,conexion_fabogriesgo())
AlertaClienteActualiza['Documento']=AlertaClienteActualiza['Documento'].astype(str).str.strip()
AlertaClienteActualiza = AlertaClienteActualiza.rename(columns={'Documento':'Cliente Asociado'})



AlertaIPMultiplesUsuarios='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaIPMultiplesUsuarios]'''
AlertaIPMultiplesUsuarios = pd.read_sql(AlertaIPMultiplesUsuarios,conexion_fabogriesgo())
AlertaIPMultiplesUsuarios['Documento']=AlertaIPMultiplesUsuarios['Documento'].astype(str).str.strip()
AlertaIPMultiplesUsuarios = AlertaIPMultiplesUsuarios.rename(columns={'Documento':'Cliente Asociado'})




AlertaFlexiAltaSalidas='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaFlexiAltasSalidas_historico]'''
AlertaFlexiAltaSalidas = pd.read_sql(AlertaFlexiAltaSalidas,conexion_fabogriesgo())
AlertaFlexiAltaSalidas['Documento']=AlertaFlexiAltaSalidas['Documento'].astype(str).str.strip()
AlertaFlexiAltaSalidas = AlertaFlexiAltaSalidas.rename(columns={'Documento':'Cliente Asociado'})



AlertaFlexiAltasEntradas='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaFlexiAltasEntradas_historico]'''
AlertaFlexiAltasEntradas = pd.read_sql(AlertaFlexiAltasEntradas,conexion_fabogriesgo())
AlertaFlexiAltasEntradas['Documento']=AlertaFlexiAltasEntradas['Documento'].astype(str).str.strip()
AlertaFlexiAltasEntradas = AlertaFlexiAltasEntradas.rename(columns={'Documento':'Cliente Asociado'})









AlertaCorreoRiesgo='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaCorreoRiesgo]'''
AlertaCorreoRiesgo = pd.read_sql(AlertaCorreoRiesgo,conexion_fabogriesgo())
AlertaCorreoRiesgo['Documento']=AlertaCorreoRiesgo['Documento'].astype(str).str.strip()
AlertaCorreoRiesgo = AlertaCorreoRiesgo.rename(columns={'Documento':'Cliente Asociado'})








AlertaRecuperacionContraseñaMultiple='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaRecuperacionContraseñaMultiple]'''
AlertaRecuperacionContraseñaMultiple = pd.read_sql(AlertaRecuperacionContraseñaMultiple,conexion_fabogriesgo())
AlertaRecuperacionContraseñaMultiple['Documento']=AlertaRecuperacionContraseñaMultiple['Documento'].astype(str).str.strip()
AlertaRecuperacionContraseñaMultiple = AlertaRecuperacionContraseñaMultiple.rename(columns={'Documento':'Cliente Asociado'})







AlertaActualizacionRiesgo='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaActualizacionRiesgo]'''
AlertaActualizacionRiesgo = pd.read_sql(AlertaActualizacionRiesgo,conexion_fabogriesgo())
AlertaActualizacionRiesgo['Documento']=AlertaActualizacionRiesgo['Documento'].astype(str).str.strip()
AlertaActualizacionRiesgo = AlertaActualizacionRiesgo.rename(columns={'Documento':'Cliente Asociado'})






AlertaSimilitudIPCreacion='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaSimilitudIPCreacion_historico]'''
AlertaSimilitudIPCreacion = pd.read_sql(AlertaSimilitudIPCreacion,conexion_fabogriesgo())
AlertaSimilitudIPCreacion['Documento']=AlertaSimilitudIPCreacion['Documento'].astype(str).str.strip()
AlertaSimilitudIPCreacion = AlertaSimilitudIPCreacion.rename(columns={'Documento':'Cliente Asociado'})




AlertaIntentosVinculacion='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaIntentosVinculacion]'''
AlertaIntentosVinculacion = pd.read_sql(AlertaIntentosVinculacion,conexion_fabogriesgo())
AlertaIntentosVinculacion['Documento']=AlertaIntentosVinculacion['Documento'].astype(str).str.strip()
AlertaIntentosVinculacion = AlertaIntentosVinculacion.rename(columns={'Documento':'Cliente Asociado'})



AlertaMultipleDispositivo='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaMultipleDispositivo_historico]'''
AlertaMultipleDispositivo = pd.read_sql(AlertaMultipleDispositivo,conexion_fabogriesgo())
AlertaMultipleDispositivo['Documento']=AlertaMultipleDispositivo['Documento'].astype(str).str.strip()
AlertaMultipleDispositivo = AlertaMultipleDispositivo.rename(columns={'Documento':'Cliente Asociado'})




AlertaIntentosEvidente='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaIntentosEvidente_historico]'''
AlertaIntentosEvidente = pd.read_sql(AlertaIntentosEvidente,conexion_fabogriesgo())
AlertaIntentosEvidente['Documento']=AlertaIntentosEvidente['Documento'].astype(str).str.strip()
AlertaIntentosEvidente = AlertaIntentosEvidente.rename(columns={'Documento':'Cliente Asociado'})




AlertaDisparidadCIIU='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaDisparidadCIIU]'''
AlertaDisparidadCIIU = pd.read_sql(AlertaDisparidadCIIU,conexion_fabogriesgo())
AlertaDisparidadCIIU['Documento']=AlertaDisparidadCIIU['Documento'].astype(str).str.strip()
AlertaDisparidadCIIU = AlertaDisparidadCIIU.rename(columns={'Documento':'Cliente Asociado'})









AlertaSospechaTransfiya='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaSospechaTransfiya]'''
AlertaSospechaTransfiya = pd.read_sql(AlertaSospechaTransfiya,conexion_fabogriesgo())
AlertaSospechaTransfiya['Documento']=AlertaSospechaTransfiya['Documento'].astype(str).str.strip()
AlertaSospechaTransfiya = AlertaSospechaTransfiya.rename(columns={'Documento':'Cliente Asociado'})








AlertaCreacionEnrolamiento='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaCreacionEnrolamiento]'''
AlertaCreacionEnrolamiento = pd.read_sql(AlertaCreacionEnrolamiento,conexion_fabogriesgo())
AlertaCreacionEnrolamiento['Documento']=AlertaCreacionEnrolamiento['Documento'].astype(str).str.strip()
AlertaCreacionEnrolamiento = AlertaCreacionEnrolamiento.rename(columns={'Documento':'Cliente Asociado'})





AlertaAltaTransaccionalidad='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaAltaTransaccionalidad]'''
AlertaAltaTransaccionalidad = pd.read_sql(AlertaAltaTransaccionalidad,conexion_fabogriesgo())
AlertaAltaTransaccionalidad['Documento']=AlertaAltaTransaccionalidad['Documento'].astype(str).str.strip()
AlertaAltaTransaccionalidad = AlertaAltaTransaccionalidad.rename(columns={'Documento':'Cliente Asociado'})




AlertaActualizacionRegistroDispositivo='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaActualizacionRegistroDispositivo_historico]'''
AlertaActualizacionRegistroDispositivo = pd.read_sql(AlertaActualizacionRegistroDispositivo,conexion_fabogriesgo())
AlertaActualizacionRegistroDispositivo['Documento']=AlertaActualizacionRegistroDispositivo['Documento'].astype(str).str.strip()
AlertaActualizacionRegistroDispositivo = AlertaActualizacionRegistroDispositivo.rename(columns={'Documento':'Cliente Asociado'})




AlertaRegistroEquipoRecuperacionUsuario='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaRegistroEquipoRecuperacionUsuario]'''
AlertaRegistroEquipoRecuperacionUsuario = pd.read_sql(AlertaRegistroEquipoRecuperacionUsuario,conexion_fabogriesgo())
AlertaRegistroEquipoRecuperacionUsuario['Documento']=AlertaRegistroEquipoRecuperacionUsuario['Documento'].astype(str).str.strip()
AlertaRegistroEquipoRecuperacionUsuario = AlertaRegistroEquipoRecuperacionUsuario.rename(columns={'Documento':'Cliente Asociado'})




AlertaSaldoFlexidigitalAhorro='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaSaldoFlexidigitalAhorroHistorico]'''
AlertaSaldoFlexidigitalAhorro = pd.read_sql(AlertaSaldoFlexidigitalAhorro,conexion_fabogriesgo())
AlertaSaldoFlexidigitalAhorro['Documento']=AlertaSaldoFlexidigitalAhorro['Documento'].astype(str).str.strip()
AlertaSaldoFlexidigitalAhorro = AlertaSaldoFlexidigitalAhorro.rename(columns={'Documento':'Cliente Asociado'})






EntradasFlexiDigitalplus8smmlv='''Select distinct Documento
from [AlertasFraude].[dbo].[EntradasFlexiDigitalplus8smmlvHistorico]'''
EntradasFlexiDigitalplus8smmlv = pd.read_sql(EntradasFlexiDigitalplus8smmlv,conexion_fabogriesgo())
EntradasFlexiDigitalplus8smmlv['Documento']=EntradasFlexiDigitalplus8smmlv['Documento'].astype(str).str.strip()
EntradasFlexiDigitalplus8smmlv = EntradasFlexiDigitalplus8smmlv.rename(columns={'Documento':'Cliente Asociado'})



AlertaCupoLimiteExcedido='''Select distinct Documento
from [AlertasFraude].[dbo].[AlertaCupoLimiteExcedido_historico]'''
AlertaCupoLimiteExcedido = pd.read_sql(AlertaCupoLimiteExcedido,conexion_fabogriesgo())
AlertaCupoLimiteExcedido['Documento']=AlertaCupoLimiteExcedido['Documento'].astype(str).str.strip()
AlertaCupoLimiteExcedido = AlertaCupoLimiteExcedido.rename(columns={'Documento':'Cliente Asociado'})



AlertamientoCupoSuperado='''Select distinct DocumentoCliente
from [AlertasFraude].[dbo].[AlertamientoCupoSuperado]'''
AlertamientoCupoSuperado = pd.read_sql(AlertamientoCupoSuperado,conexion_fabogriesgo())
AlertamientoCupoSuperado['DocumentoCliente']=AlertamientoCupoSuperado['DocumentoCliente'].astype(str).str.strip()
AlertamientoCupoSuperado = AlertamientoCupoSuperado.rename(columns={'DocumentoCliente':'Cliente Asociado'})



# BASE DE FRAUDES 

fraudes='''Select*
from Fabogcubox.[Finandina_Cartera].[dbo].[00 Fraudes_Bco_FA]'''
fraudes = pd.read_sql(fraudes,conexion_fabogriesgo())



# fraudes ultimos 3 meses


# fraudes.columns

# Supongamos que tenemos un DataFrame llamado df con una columna 'fecha' que contiene las fechas de los registros

# Convertir la columna 'fecha' al tipo de dato datetime si no está en ese formato
fraudes['Fecha'] = pd.to_datetime(fraudes['Fecha'])

# Obtener la fecha actual
fecha_actual = datetime.now()

# Calcular la fecha límite de tres meses atrás
fecha_limite = fecha_actual - timedelta(days=90)

# Filtrar los registros que están dentro del rango de los últimos tres meses
df_filtrado = fraudes[fraudes['Fecha'] >= fecha_limite]



fraudes=df_filtrado




## REVISIÓN CON BASE FRAUDES 

fraudes['Cliente Asociado']=fraudes['Cliente Asociado'].astype(str).str.strip()

c1 = pd.merge(AlertaAhorroRegistroCelularSimilitud,fraudes,on='Cliente Asociado',how='inner')

c2 = pd.merge(AlertaAhorroRegistroCorreoSimilitud,fraudes,on='Cliente Asociado',how='inner')

c3 = pd.merge(AlertaTDCCondicionSimilitud,fraudes,on='Cliente Asociado',how='inner')

c4 = pd.merge(AlertaTDCReexpedicion,fraudes,on='Cliente Asociado',how='inner')

c5 = pd.merge(AlertaDebitoReexpedicion,fraudes,on='Cliente Asociado',how='inner')

c6 = pd.merge(AlertaCreacionActualizacion,fraudes,on='Cliente Asociado',how='inner')

c7 = pd.merge(AlertaDisparidadIP,fraudes,on='Cliente Asociado',how='inner')

c8 = pd.merge(AlertaRegistrosAtipicos,fraudes,on='Cliente Asociado',how='inner')

c9 = pd.merge(AlertaTransaccionesOlvido,fraudes,on='Cliente Asociado',how='inner')

c10 = pd.merge(AlertaTransaccionesActualizacion,fraudes,on='Cliente Asociado',how='inner')

c11 = pd.merge(AlertaTransaccionesIngresosTDC,fraudes,on='Cliente Asociado',how='inner')

c12 = pd.merge(AlertaTransaccionesInactividad,fraudes,on='Cliente Asociado',how='inner')

c13 = pd.merge(AlertaVariacionTransaccional,fraudes,on='Cliente Asociado',how='inner')

c14 = pd.merge(AlertaClienteActualiza,fraudes,on='Cliente Asociado',how='inner')


c15 = pd.merge(AlertaIPMultiplesUsuarios,fraudes,on='Cliente Asociado',how='inner')

c16 = pd.merge(AlertaFlexiAltaSalidas,fraudes,on='Cliente Asociado',how='inner')

c17 = pd.merge(AlertaFlexiAltasEntradas,fraudes,on='Cliente Asociado',how='inner')

c18 = pd.merge(AlertaCorreoRiesgo,fraudes,on='Cliente Asociado',how='inner')

c19 = pd.merge(AlertaRecuperacionContraseñaMultiple,fraudes,on='Cliente Asociado',how='inner')

c20 = pd.merge(AlertaActualizacionRiesgo,fraudes,on='Cliente Asociado',how='inner')

c21 = pd.merge(AlertaSimilitudIPCreacion,fraudes,on='Cliente Asociado',how='inner')

c22 = pd.merge(AlertaIntentosVinculacion,fraudes,on='Cliente Asociado',how='inner')

c23 = pd.merge(AlertaMultipleDispositivo,fraudes,on='Cliente Asociado',how='inner')

c24 = pd.merge(AlertaIntentosEvidente,fraudes,on='Cliente Asociado',how='inner')

c25 = pd.merge(AlertaDisparidadCIIU,fraudes,on='Cliente Asociado',how='inner')

c26 = pd.merge(AlertaSospechaTransfiya,fraudes,on='Cliente Asociado',how='inner')

c27 = pd.merge(AlertaCreacionEnrolamiento,fraudes,on='Cliente Asociado',how='inner')

c28 = pd.merge(AlertaAltaTransaccionalidad,fraudes,on='Cliente Asociado',how='inner')

c29 = pd.merge(AlertaActualizacionRegistroDispositivo,fraudes,on='Cliente Asociado',how='inner')

c30 = pd.merge(AlertaRegistroEquipoRecuperacionUsuario,fraudes,on='Cliente Asociado',how='inner')

c31 = pd.merge(AlertaSaldoFlexidigitalAhorro,fraudes,on='Cliente Asociado',how='inner')

c32 = pd.merge(EntradasFlexiDigitalplus8smmlv,fraudes,on='Cliente Asociado',how='inner')

c33 = pd.merge(AlertaCupoLimiteExcedido,fraudes,on='Cliente Asociado',how='inner')


reporte = pd.DataFrame({'Alertamiento': ['AlertaAhorroRegistroCelularSimilitud',
                            'AlertaAhorroRegistroCorreoSimilitud',
                            'AlertaTDCCondicionSimilitud',
                            'AlertaTDCReexpedicion',
                            'AlertaDebitoReexpedicion',
                            'AlertaCreacionActualizacion',
                            'AlertaDisparidadIP',
                            'AlertaRegistrosAtipicos',
                            'AlertaTransaccionesOlvido',
                            'AlertaTransaccionesActualizacion',
                            'AlertaTransaccionesIngresosTDC',
                            'AlertaTransaccionesInactividad',
                            'AlertaVariacionTransaccional',
                            'AlertaClienteActualiza',
                            'AlertaIPMultiplesUsuarios',
                            'AlertaFlexiAltaSalidas',
                            'AlertaFlexiAltasEntradas',
                            'AlertaCorreoRiesgo',
                            'AlertaRecuperacionContraseñaMultiple',
                            'AlertaActualizacionRiesgo',
                            'AlertaSimilitudIPCreacion',
                            'AlertaIntentosVinculacion',
                            'AlertaMultipleDispositivo',
                            'AlertaIntentosEvidente',
                            'AlertaDisparidadCIIU',
                            'AlertaSospechaTransfiya',
                            'AlertaCreacionEnrolamiento',
                            'AlertaAltaTransaccionalidad',
                            'AlertaActualizacionRegistroDispositivo',
                            'AlertaRegistroEquipoRecuperacionUsuario',
                            'AlertaSaldoFlexidigitalAhorro',
                            'EntradasFlexiDigitalplus8smmlv',
                            'AlertaCupoLimiteExcedido'],
           'Cantidad Alertamientos': [len(AlertaAhorroRegistroCelularSimilitud),
                                       len(AlertaAhorroRegistroCorreoSimilitud),
                                       len(AlertaTDCCondicionSimilitud),
                                       len(AlertaTDCReexpedicion),
                                       len(AlertaDebitoReexpedicion),
                                       len(AlertaCreacionActualizacion),
                                       len(AlertaDisparidadIP),
                                       len(AlertaRegistrosAtipicos),
                                       len(AlertaTransaccionesOlvido),
                                       len(AlertaTransaccionesActualizacion),
                                       len(AlertaTransaccionesIngresosTDC),
                                       len(AlertaTransaccionesInactividad),
                                       len(AlertaVariacionTransaccional),
                                       len(AlertaClienteActualiza),
                                       len(AlertaIPMultiplesUsuarios),
                                       len(AlertaFlexiAltaSalidas),
                                       len(AlertaFlexiAltasEntradas),
                                       len(AlertaCorreoRiesgo),
                                       len(AlertaRecuperacionContraseñaMultiple),
                                       len(AlertaActualizacionRiesgo),
                                       len(AlertaSimilitudIPCreacion),
                                       len(AlertaIntentosVinculacion),
                                       len(AlertaMultipleDispositivo),
                                       len(AlertaIntentosEvidente),
                                       len(AlertaDisparidadCIIU),
                                       len(AlertaSospechaTransfiya),
                                       len(AlertaCreacionEnrolamiento),
                                       len(AlertaAltaTransaccionalidad),
                                       len(AlertaActualizacionRegistroDispositivo),
                                       len(AlertaRegistroEquipoRecuperacionUsuario),
                                       len(AlertaSaldoFlexidigitalAhorro),
                                       len(EntradasFlexiDigitalplus8smmlv),
                                       len(AlertaCupoLimiteExcedido)],
           'Validacion Fraudes': [len(c1),
                                   len(c2),
                                   len(c3),
                                   len(c4),
                                   len(c5),
                                   len(c6),
                                   len(c7),
                                   len(c8),
                                   len(c9),
                                   len(c10),
                                   len(c11),
                                   len(c12),
                                   len(c13),
                                   len(c14),
                                   len(c15),
                                   len(c16),
                                   len(c17),
                                   len(c18),
                                   len(c19),
                                   len(c20),
                                   len(c21),
                                   len(c22),
                                   len(c23),
                                   len(c24),
                                   len(c25),
                                   len(c26),
                                   len(c27),
                                   len(c28),
                                   len(c29),
                                   len(c30),
                                   len(c31),
                                   len(c32),
                                   len(c33)]})

reporte['Efectividad'] = reporte['Validacion Fraudes']/reporte['Cantidad Alertamientos']

reporte['Observacion'] = np.where(reporte['Cantidad Alertamientos']>=100,'Sobrealertamiento','NO Sobrealertamiento')


### entraga de reporte 


# Exportar el DataFrame a un archivo Excel
reporte.to_excel('//192.168.60.149/desarrollo y ciencia de datos/024_Jose_Gomez/reporteAlertasClientesUnicos.xlsx', index=False)









































































































