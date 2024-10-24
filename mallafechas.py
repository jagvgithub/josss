# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:12:56 2023

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
FROM {tabla_datos}'''.format(tabla_datos='[FABOGRIESGO\RIESGODB].[Productos y transaccionalidad].[dbo].[ConsolidadoNaturalDemografia]')
datos_demograficos_limpio=load_data(query_datos_demograficos,config_db_riesgo[1],model_logger=logger)

####################################################################################

import pandas as pd
import numpy as np
import datetime as dt


year= dt.datetime.now().year

# historico
df = pd.DataFrame(np.array([[year,  '01', year-1,1,'0','C:/Users/josgom/Desktop/historicoSARLAFT/'], 
                             [year, '02', year,2,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '03', year,3,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '04', year,4,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '05', year,5,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '06', year,6,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '07', year,7,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '08', year,8,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '09', year,9,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '10', year,10,'0' ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '11', year,11,'0' ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '12', year,12,'0' ,'C:/Users/josgom/Desktop/historicoSARLAFT/']]),
                    columns=['year', 'month', 'yy','nummonth','mesbase','prueba'])

# reporte 
df2 = pd.DataFrame(np.array([[year,  '01', year-1,1,'0','C:/Users/josgom/Desktop/historicoSARLAFT/'], 
                             [year, '02', year,2,'1'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '03', year,3,'1'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '04', year,4,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '05', year,5,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '06', year,6,'1'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '07', year,7,'1'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '08', year,8,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '09', year,9,'0'  ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '10', year,10,'0' ,'C:/Users/josgom/Desktop/historicoSARLAFT/'],
                             [year, '11', year,11,'1' ,'C:/Users/josgom/Desktop/historicoSARLAFT/']]),
                    columns=['year', 'month', 'yy','nummonth','mesbase','prueba'])




import pandas as pd

from datetime import datetime



df = pd.read_excel('C:/Users/josgom/Desktop/MALLA/03012023.xlsx')

df2 = pd.read_excel('C:/Users/josgom/Desktop/MALLA/base2.xlsx')



#### ejercicio malla 

p1=df[df['Documento']==1020836017]
p2=df2[df2['Documento']==1020836017]
base=pd.concat([p1,p2])
data=base.transpose() 
data['Status'] = np.where(data.iloc[:,0]==data.iloc[:,1], data.iloc[:,0], data.iloc[:,1])
data['Status2'] = np.where(data.iloc[:,0]==data.iloc[:,1], data.iloc[:,0]+data.iloc[:,0], data.iloc[:,1])
data['Status2']=data['Status2'].replace(1, datetime.now())
data['Status2']=data['Status2'].replace(2, 1)
data['Status2']=data['Status2'].replace(data.iloc[0,1]*2, data.iloc[0,1])
data = data.iloc[:, -1] # Última columna
data=pd.DataFrame(data.transpose()) 
data=data.transpose()



##### DEFINIR FUNCION 

## nuevos registros > registros anteriores 


df = pd.read_excel('C:/Users/josgom/Desktop/MALLA/03012023.xlsx')

## ultimo registro

df2 = pd.read_excel('C:/Users/josgom/Desktop/MALLA/04012023.xlsx')


## registro que fueron nuevos 

prueba = pd.merge(df2, df, on='Documento', how='outer')



d = df2[~(df['Documento'].isin(df2['Documento'])
          
prueba = df2[df2['Documento']!= df['Documento']          

             

aaa= df2[~df2['Documento'].isin(df['Documento'])]


416023-415593













#base=pd.concat([pd.DataFrame(pd.concat([p1,p2])),pd.DataFrame(p1==p2)]).replace(False,0, regex=True)

    




res = p1==p2


base = pd.concat([p1,p2])

base = pd.concat([base,res])


prueba= if base[0,]:
            statement(s)
        else:
            statement(s)
            
            

data=base.transpose()     

base.iloc[0,:]

data.iloc[:,0]



if base.iloc[0] == 0 & base.iloc[1] == 1 :
   base.iloc[1]
else:
    base.iloc[0] 


res = p1==p2


l1 = [1,2,3]
l2 = [3,2,5]
for i in p1:
    for j in p2:
        if(i==j):
            print('true')
            else print('false')
            break






## cruce

df3=pd.merge(df,df2, how='outer')


datos_cantidad_cuentas_asociado_celular=pd.DataFrame(df3.groupby(by=['nummonth']).size(),columns=['registros']).reset_index()

datos_cantidad_cuentas_asociado_celular=datos_cantidad_cuentas_asociado_celular[datos_cantidad_cuentas_asociado_celular['registros']>1]


p1=df[df['Documento']==1020836017]

p2=df2[df2['Documento']==1020836017]













def saludo(primer_nombre,apellido):
    print(f"Hola, {primer_nombre} {apellido}")









df4=pd.merge(df3,datos_cantidad_cuentas_asociado_celular, how='left',on="nummonth")

a=df4[df4['registros'].isnull()].iloc[:,0:6]

b=df4[df4['registros'].notnull()]

b=b[b['mesbase']!='0'].iloc[:,0:6]


resultados=pd.concat([a,b])

from datetime import datetime
fecha_actual=datetime.now()

resultados['mesbase'] = resultados['mesbase'].replace('1', fecha_actual)















df3.duplicated(keep='last')








#############



df1 = pd.DataFrame([['GH_1', 10, 'Hidro'],
                    ['GH_2', 20, 'Hidro'],
                    ['GH_3', 30, 'Hidro']],
                    columns= ['name','p_mw','type'])

df2 = pd.DataFrame([['GT_1', 40, 'Termo'],
                    ['GT_2', 50, 'Termo'],
                    ['GF_1', 10, 'Fict']],
                    columns= ['name','p_mw','type'])



df3 = pd.DataFrame([[150,57,110,20,10],
                    [120,66,110,20,0],
                    [90,40,105,20,0],
                    [60,40,90,20,0]],
                    columns= ['GH_1', 'GH_2', 'GH_3', 'GT_1', 'GT_2'])

act = df3.iloc[:,1]

#actualizamos el df1
df1["p_mw"] = df1.apply(lambda x: act[x["name"]] if x["name"] in act else x["p_mw"],axis=1)

#actualizamos el df2
df2["p_mw"] = df2.apply(lambda x: act[x["name"]] if x["name"] in act else x["p_mw"],axis=1)
print(df2)
print(df1)




##### 

act = df2.iloc[-1,:]

df0["mesbase"] = df0.apply(lambda x: act[x["mesbase"]] if x["mesbase"] in act else x["mesbase"],axis=1)



































