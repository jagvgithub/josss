# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 18:18:04 2023

@author: josgom
"""

################################# MODELO DE SEGMETANCIÓN SARLAFT RIESGO INHERENTE ############################




%%time
import connectorx as cx
import pandas as pd
import pandas as pd
import pyodbc
import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans

import pyodbc

server_name = 'FABOGSQL01\\AUDITORIA,52715'
database_name = 'AUDITORIA_COMPARTIDA'
integrated_security = 'yes'  # Para autenticación de Windows


def convertir_fecha_juliana(fecha_juliana):
    try:
        anno = int(fecha_juliana[:4])
        dia_del_anno = int(fecha_juliana[4:])
        fecha_estandar = datetime(anno, 1, 1) + timedelta(days=dia_del_anno - 1)
        return fecha_estandar.strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return None  # Maneja los valoconsolidado incorrectos proporcionando un valor nulo




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


def convertir_fecha(fecha_str):
    try:
        fecha_obj = datetime.strptime(fecha_str, "%d%m%Y")
        return fecha_obj.strftime("%Y-%m-%d")
    except ValueError:
        return None


def actualizar_columnas(df, columnas):
    for columna in columnas:
        df[columna] = np.where(pd.notnull(df[columna]), df[columna], df[f'{columna}_actualizado'])



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









## tiempo de carga cup003  6 minutos


consolidado_cup = f'''select * from openquery(DB2400_182,'select cuna1,cuna2,cuna3,cussnr,cuclph,cuopdt,cuinc,cuema1,cumtnd,cucens from BNKPRD01.cup003 ')'''

# Ejecuta la consulta y guarda los resultados en una lista
consolidado_cup = cargue_openquery(conn, consolidado_cup)

consolidado_cup['CUCENS'] = consolidado_cup['CUCENS'].astype(str).str.replace('.', '')

# Convierte los valores a enteros
consolidado_cup['CUCENS'] = consolidado_cup['CUCENS'].astype(int)













from datetime import datetime, timedelta

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






SQL_SERVER_RIESGO= "fabogriesgo:49505"
SQL_DB_RIESGO = "AlertasFraude"
sql_connection = f"mssql://{SQL_SERVER_RIESGO}/{SQL_DB_RIESGO}?trusted_connection=true"
ultimo_producto_aperturado = '''SELECT DocumentoCliente,max(FechaApertura) as FechaApertura,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
 FROM [Productos y transaccionalidad].[dbo].[ConsolidadoProductos]			 
WHERE EstadoCuenta='Activa'
             GROUP BY DocumentoCliente'''
ultimo_producto_aperturado = (cx.read_sql(conn = sql_connection, query = ultimo_producto_aperturado, partition_on="Rank", partition_num=10,  return_type="pandas")).groupby('DocumentoCliente').first().reset_index().drop(columns=['Rank'])



consolidado_productos = '''SELECT DocumentoCliente		
 ,CASE WHEN TipoProducto='TDC' AND LineaProducto LIKE ('%Virtual%') THEN 'TDC Digital'
			      WHEN TipoProducto='TDC' AND LineaProducto NOT LIKE ('%Virtual%') THEN 'TDC Física'
			      ELSE LineaProducto
             END LineaProducto,TipoProducto,EstadoCuenta,1 as cantidad,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	  		 FROM [Productos y transaccionalidad].[dbo].[ConsolidadoProductos]
			 WHERE EstadoCuenta='Activa' '''
consolidado_productos = cx.read_sql(conn = sql_connection, query = consolidado_productos, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])




consolidado_pj = '''SELECT distinct CAST(NIT AS VARCHAR(255)) AS DocumentoCliente,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
  FROM [ModelosSARLAFT].[dbo].[CONSOLIDADOPJ]'''
consolidado_pj = cx.read_sql(conn = sql_connection, query = consolidado_pj, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])



consolidado_pn = '''SELECT distinct CAST([numero id] AS VARCHAR(255)) AS DocumentoCliente,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
  FROM [ModelosSARLAFT].[dbo].[CONSOLIDADOPN]'''
consolidado_pn = cx.read_sql(conn = sql_connection, query = consolidado_pn, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])



demografico_pj = '''SELECT DocumentoCliente AS DocumentoCliente
       ,NombreCliente
	   ,TipoPersona
	   ,CASE WHEN CiudadActual IS NOT NULL THEN dbo.RemoveNonAlphaCharacters(REPLACE(REPLACE(REPLACE(UPPER(LEFT(CiudadActual,CHARINDEX('(',CiudadActual+'(')-1)),'D.C',''),'FLORIDA BLANCA','FLORIDABLANCA'),'SANTAFE DE BOGOTA DC','BOGOTA')) COLLATE SQL_Latin1_General_CP1253_CI_AI
	         ELSE CiudadActual
        END CiudadActual
	   ,CanalEntrada
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

# Aplica el formato año-mes-día a la columna 'Fecha'



demografico_pn = '''SELECT DocumentoCliente AS DocumentoCliente
      ,NombreCliente
	   ,TipoPersona
	   ,CASE WHEN CiudadActual IS NOT NULL THEN dbo.RemoveNonAlphaCharacters(REPLACE(REPLACE(REPLACE(UPPER(LEFT(CiudadActual,CHARINDEX('(',CiudadActual+'(')-1)),'D.C',''),'FLORIDA BLANCA','FLORIDABLANCA'),'SANTAFE DE BOGOTA DC','BOGOTA')) COLLATE SQL_Latin1_General_CP1253_CI_AI
	         ELSE CiudadActual
        END CiudadActual
	   ,CanalEntrada
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




# limpieza 


consolidado_productos['DocumentoCliente']=consolidado_productos['DocumentoCliente'].str.strip().astype(str)
consolidado_pj['DocumentoCliente']=consolidado_pj['DocumentoCliente'].str.strip().astype(str)



# Supongamos que ya tienes el DataFrame 'consolidado_productos'

productos_pivoteado = (
    consolidado_productos
    .pivot_table(index='DocumentoCliente', columns='LineaProducto', values='cantidad', aggfunc='sum')
    .fillna(0)
    .astype('int64')
    .reset_index(drop=False)
    .loc[(consolidado_productos['DocumentoCliente'] != "0") & (consolidado_productos['DocumentoCliente'] != '')]
)


# a=pivoteado_df[pivoteado_df['DocumentoCliente']=='1020836017']



#### base persona juridica 



# Cantidad de productos agregados a la consulta inicial PJ

cruce = (pd.merge(consolidado_pj, productos_pivoteado,on='DocumentoCliente',how = 'left')).fillna(0)

cruce2 = pd.merge(cruce, ultimo_producto_aperturado,on='DocumentoCliente',how = 'left')





# Concatena los DataFrames a lo largo del eje de las columnas (axis=1)
demografico = (pd.concat([demografico_pj, demografico_pn], axis=0)).groupby('DocumentoCliente').first().reset_index()

cruce3 = pd.merge(cruce2, demografico,on='DocumentoCliente',how = 'left')






documentos_a_buscar_en_cup003 = (pd.DataFrame(cruce3[cruce3['NombreCliente'].isnull()]['DocumentoCliente']))


documentos_unicos = documentos_a_buscar_en_cup003['DocumentoCliente'].unique().astype(str)


#################





# Llama a la función con tu lista de documentos únicos y guarda el consolidadoultado en una variable
consultacupPJ = (split_and_execute_queries(documentos_unicos)).rename(columns={'CUNA2':'DireccionActual'
                                                                          ,'CUNA3':'CiudadActual'
                                                                          ,'CUSSNR':'DocumentoCliente'
                                                                          ,'CUCLPH':'Celular'
                                                                          ,'CUOPDT':'FechaVinculacion'
                                                                          ,'CUINC':'MontoIngresos'
                                                                          ,'CUEMA1':'Correo'
                                                                          ,'CUMTND':'FechaUltimaActualizacionCore'
                                                                          ,'CUNA1':'NombreCliente'})

# Ahora puedes usar df_consolidado para trabajar con el DataFrame consolidadoultante



###################



consultacupPJ['DocumentoCliente'] = consultacupPJ['DocumentoCliente'].astype(str).str.rstrip('.0')
consultacupPJ['FechaUltimaActualizacionCore'] = consultacupPJ['FechaUltimaActualizacionCore'].astype(str).str.rstrip('.0')
consultacupPJ['FechaUltimaActualizacionCore'] = consultacupPJ['FechaUltimaActualizacionCore'].apply(convertir_fecha_juliana)



consultacupPJ['FechaVinculacion'] = consultacupPJ['FechaVinculacion'].astype(str).str.rstrip('.0')
consultacupPJ['FechaVinculacion']=consultacupPJ['FechaVinculacion'].apply(agregar_cero)
consultacupPJ['FechaVinculacion']=consultacupPJ['FechaVinculacion'].apply(agregar_20)
consultacupPJ['FechaVinculacion'] = np.where(consultacupPJ['FechaVinculacion'].str.len() == 8, consultacupPJ['FechaVinculacion'], np.nan)
consultacupPJ['FechaVinculacion'] = consultacupPJ['FechaVinculacion'].astype(str)
consultacupPJ['FechaVinculacion']=consultacupPJ['FechaVinculacion'].apply(lambda x: convertir_fecha(x))


consultacupPJ=consultacupPJ.groupby('DocumentoCliente').first().reset_index()


# multiplicamos los ingconsolidadoos del cup003 por mil para manejar la misma escala 
consultacupPJ['MontoIngresos'] *= 1000

# [col for col in consultacupPJ.columns if 'Monto' in col]


consolidado = cruce3.merge(consultacupPJ, on='DocumentoCliente', how='left', suffixes=('', '_actualizado'))



# mejora de completitud de información


# Ejemplo de uso:
columnas_a_actualizar = ['DireccionActual', 'CiudadActual', 'Celular', 'FechaVinculacion', 'MontoIngresos', 'Correo', 'FechaUltimaActualizacionCore', 'NombreCliente']

# Llama a la función para actualizar las columnas en el DataFrame consolidado
actualizar_columnas(consolidado, columnas_a_actualizar)
columnas_a_eliminar = [columna for columna in consolidado.columns if 'actualizado' in columna]
consolidado = consolidado.drop(columns=columnas_a_eliminar)







query_tx_ahorro_entrada = '''select DocumentoCliente,FechaTransaccionEfectiva,MontoTransaccion,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesAhorro]
where CaracterTransaccion = 'Entrada'
and DescripcionTransaccional4 
in ('DEPOSITO DE CUENTA DE AHORROS SIN LIBRETA',
'DEPOSITO DEL CLIENTE',
'MEMO DE CREDITO') '''
ahorro_entradas = cx.read_sql(conn=sql_connection, query=query_tx_ahorro_entrada, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])




ahorro_entradas['FechaTransaccionEfectiva'] = ahorro_entradas['FechaTransaccionEfectiva'].astype(str).str.rstrip('.0')

ahorro_entradas['DocumentoCliente'] = ahorro_entradas['DocumentoCliente'].astype(str).str.rstrip('.0')

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




activo_entradas['FechaPublicacion'] = activo_entradas['FechaPublicacion'].astype(str).str.rstrip('.0')

activo_entradas['DocumentoCliente'] = activo_entradas['DocumentoCliente'].astype(str).str.rstrip('.0')

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



cdt_entradas['FechaEfectiva'] = cdt_entradas['FechaEfectiva'].astype(str).str.rstrip('.0')

cdt_entradas['DocumentoCliente'] = cdt_entradas['DocumentoCliente'].astype(str).str.rstrip('.0')

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



query_tx_tdc_entrada = f'''SELECT *    FROM OPENQUERY(DB2400_182,' select PersonalTDC.NUMDOC, TransaccionesTDC.IMPFAC,TransaccionesTDC.FECFAC,CAST(PersonalTDC.NUMDOC as integer) as Rank
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
tdc_entradas = (cx.read_sql(conn=sql_connection, query=query_tx_tdc_entrada, partition_on='Rank', partition_num=2, return_type='pandas')).rename(columns={'NUMDOC':'DocumentoCliente'})




# query_tx_tdc_entrada = f'''SELECT *    FROM OPENQUERY(DB2400_182,' select PersonalTDC.NUMDOC, TransaccionesTDC.IMPFAC,TransaccionesTDC.FECFAC
#   from INTTARCRE.SATMOVEXT as TransaccionesTDC
# 								 left join INTTARCRE.SATTARJET as TarjetaTDC
# 								 ON TransaccionesTDC.CUENTAME=TarjetaTDC.CUENTA and
# 								    TransaccionesTDC.PANME=TarjetaTDC.PAN
# 								 left join INTTARCRE.SATBENEFI as CuentaTDC
# 							     on TarjetaTDC.CUENTA=CuentaTDC.CUENTA and
# 								    TarjetaTDC.NUMBENCTA=CuentaTDC.NUMBENCTA
# 								 left join INTTARCRE.SATDACOPE as PersonalTDC
# 								 on CuentaTDC.IDENTCLI=PersonalTDC.IDENTCLI 
# 								 where TransaccionesTDC.TIPOFAC in (''67'',''253'')  and TransaccionesTDC.FECFAC >= ''{fecha_seis_meses_antes}'' and PersonalTDC.NUMDOC is NOT NULL
# 								 ' ) '''
# # tdc_entradas = (cx.read_sql(conn=sql_connection, query=query_tx_tdc_entrada, partition_on='Rank', partition_num=2, return_type='pandas')).rename(columns={'NUMDOC':'DocumentoCliente'})
# tdc_entradas = (cargue_openquery(conn, query_tx_tdc_entrada)).rename(columns={'NUMDOC':'DocumentoCliente'})



tdc_entradas['FECFAC'] = (pd.to_datetime(tdc_entradas['FECFAC'])).dt.strftime("%Y-%m-%d")
tdc_entradas['DocumentoCliente'] = tdc_entradas['DocumentoCliente'].astype(str).str.rstrip('.0')



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







################# estimación de salidas 



query_tx_ahorro_salida = '''select DocumentoCliente,FechaTransaccionEfectiva,MontoTransaccion,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank from [Productos y transaccionalidad].[dbo].[TransaccionesAhorro]
where CaracterTransaccion = 'Salida'
 '''
ahorro_salidas = cx.read_sql(conn=sql_connection, query=query_tx_ahorro_salida, partition_on='Rank', partition_num=50, return_type='pandas').drop(columns=['Rank'])



ahorro_salidas['FechaTransaccionEfectiva'] = ahorro_salidas['FechaTransaccionEfectiva'].astype(str).str.rstrip('.0')

ahorro_salidas['DocumentoCliente'] = ahorro_salidas['DocumentoCliente'].astype(str).str.rstrip('.0')

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



activo_salidas['FechaPublicacion'] = activo_salidas['FechaPublicacion'].astype(str).str.rstrip('.0')

activo_salidas['DocumentoCliente'] = activo_salidas['DocumentoCliente'].astype(str).str.rstrip('.0')

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



cdt_salidas['FechaEfectiva'] = cdt_salidas['FechaEfectiva'].astype(str).str.rstrip('.0')

cdt_salidas['DocumentoCliente'] = cdt_salidas['DocumentoCliente'].astype(str).str.rstrip('.0')

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

query_tx_tdc_salida = f'''SELECT NUMDOC as DocumentoCliente,FECFACOC,IMPFACOC    FROM OPENQUERY(DB2400_182,'select C.NUMDOC,A.FECFACOC,A.IMPFACOC 
  FROM  INTTARCRE.SATOPECUO A 
  LEFT join INTTARCRE.SATBENEFI B
  on A.CUENTAOC = B.CUENTA
  LEFT JOIN  INTTARCRE.SATDACOPE C
  ON B.IDENTCLI = C.IDENTCLI
  where  A.FECFACOC >= ''{fecha_seis_meses_antes}'' ') '''


tdc_salidas =cargue_openquery(conn,query_tx_tdc_salida)



tdc_salidas['DocumentoCliente'] = tdc_salidas['DocumentoCliente'].astype(str).str.rstrip('.0')


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



consolidado = pd.merge(consolidado, consolidado_movimientos,on='DocumentoCliente',how='left')



############################################





######## consulta canal de entrada 



canal_entrada_pj = '''select DocumentoCliente
	          ,DescripcionSucursal
	          ,CASE WHEN DescripcionSucursal LIKE '%APP%' THEN 'APP'
			   	    WHEN DescripcionSucursal like '%Concesionario%' THEN 'Concesionario'
			   	    WHEN DescripcionSucursal like '%canal especializado%' THEN 'CVD'
			   	    WHEN DescripcionSucursal LIKE '%Oficina%' THEN 'Oficina'
			   	    WHEN DescripcionSucursal like '%Fuerza%' THEN 'CVD'     
			   	    WHEN DescripcionSucursal like '%FMV%' THEN 'CVD'  
			   	    WHEN DescripcionSucursal like '%DIRECCION GENERAL%' THEN 'Oficina'  
			   	    WHEN DescripcionSucursal like '%SUCURSAL VIRTUAL%' THEN 'Internet'  
			   	    WHEN DescripcionSucursal like '%ATM TELLER DEFAULT%' THEN 'Internet'  
			   	    WHEN DescripcionSucursal LIKE '%CENTRO ANDINO%' THEN 'Oficina'
			   	    WHEN DescripcionSucursal like '%CORRETAJE%' THEN 'Concesionario'
		       ELSE 'Oficina' END CanalEntradaVF,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
 FROM [Productos y transaccionalidad].[dbo].[DemografiaJuridicaCore]'''		 

canal_entrada_pj = (cx.read_sql(conn = sql_connection, query = canal_entrada_pj, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])



canal_entrada_pn = '''select DocumentoCliente
	          ,DescripcionSucursal
	          ,CASE WHEN DescripcionSucursal LIKE '%APP%' THEN 'APP'
			   	    WHEN DescripcionSucursal like '%Concesionario%' THEN 'Concesionario'
			   	    WHEN DescripcionSucursal like '%canal especializado%' THEN 'CVD'
			   	    WHEN DescripcionSucursal LIKE '%Oficina%' THEN 'Oficina'
			   	    WHEN DescripcionSucursal like '%Fuerza%' THEN 'CVD'     
			   	    WHEN DescripcionSucursal like '%FMV%' THEN 'CVD'  
			   	    WHEN DescripcionSucursal like '%DIRECCION GENERAL%' THEN 'Oficina'  
			   	    WHEN DescripcionSucursal like '%SUCURSAL VIRTUAL%' THEN 'Internet'  
			   	    WHEN DescripcionSucursal like '%ATM TELLER DEFAULT%' THEN 'Internet'  
			   	    WHEN DescripcionSucursal LIKE '%CENTRO ANDINO%' THEN 'Oficina'
			   	    WHEN DescripcionSucursal like '%CORRETAJE%' THEN 'Concesionario'
		       ELSE 'Oficina' END CanalEntradaVF,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
 FROM [Productos y transaccionalidad].[dbo].[DemografiaNaturalCore]'''		 

canal_entrada_pn = (cx.read_sql(conn = sql_connection, query = canal_entrada_pn, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])


consolidado_canal_entrada = pd.concat([canal_entrada_pj, canal_entrada_pn], axis=0).groupby('DocumentoCliente').first().reset_index()

consolidado_canal_entrada['DocumentoCliente'] = consolidado_canal_entrada['DocumentoCliente'].astype(str).str.rstrip('.0')
consolidado_canal_entrada = consolidado_canal_entrada.drop_duplicates(subset=['DocumentoCliente'])


consolidado=pd.merge(consolidado,consolidado_canal_entrada,on='DocumentoCliente',how='left')






### agregación catalogo departamento reisgo jurisdicción 

catalogo_jurisdiccion = '''select Municipio,[Vulnerabilidad lavado de activos] as VulnerabilidadLavadoActivos
,[Vulnerabilidad terrorismo] as VulnerabilidadTerrorismo 
,[Valor de riesgo jurisdicción] as RiesgoJurisdiccion
,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank FROM [Catalogos].[dbo].[CatalogoSegmentacionJurisdiccion]'''		 

catalogo_jurisdiccion = (cx.read_sql(conn = sql_connection, query = catalogo_jurisdiccion, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])




# Reemplazar 'BOGOTA D.C.' con 'BOGOTA' en la columna 'Municipio'
catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].replace('BOGOTA D.C.', 'BOGOTA')

# Reemplazar 'FLORIDA BLANCA' con 'FLORIDABLANCA' en la columna 'Municipio'
catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].replace('FLORIDA BLANCA', 'FLORIDABLANCA')

# Reemplazar 'SANTAFE DE BOGOTA DC' con 'BOGOTA' en la columna 'Municipio'
catalogo_jurisdiccion['Municipio'] = catalogo_jurisdiccion['Municipio'].replace('SANTAFE DE BOGOTA DC', 'BOGOTA')

catalogo_jurisdiccion = catalogo_jurisdiccion.sort_values(by='RiesgoJurisdiccion', ascending=False)
# Paso 2: Elimina duplicados en la columna A, manteniendo el primero de cada grupo
catalogo_jurisdiccion = (catalogo_jurisdiccion.drop_duplicates(subset='Municipio', keep='first')).rename(columns={'Municipio':'CiudadActual'})







consolidado['CiudadActual'] = consolidado['CiudadActual'].str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
consolidado['CiudadActual'] = consolidado['CiudadActual'].str.replace(r'\s*\([^)]*\)', '', regex=True).str.strip()
consolidado['CiudadActual'] = consolidado['CiudadActual'].replace('BOGOTA D.C', 'BOGOTA')



consolidado = pd.merge(consolidado, catalogo_jurisdiccion, on=['CiudadActual'], how='left')

consolidado['VulnerabilidadLavadoActivos'] = consolidado['VulnerabilidadLavadoActivos'].replace('<NA>', np.nan, regex=True)
consolidado['VulnerabilidadTerrorismo'] = consolidado['VulnerabilidadTerrorismo'].replace('<NA>', np.nan, regex=True)
consolidado['RiesgoJurisdiccion'] = consolidado['RiesgoJurisdiccion'].replace('<NA>', np.nan, regex=True)


mapeo = {'APP': 3, 'Internet': 3, 'CVD': 2, 'Oficina': 1, 'Concesionario': 1}

# Aplica el mapeo utilizando la función map() para crear la nueva columna 'Valoconsolidado'
consolidado['RiesgoCanal'] = consolidado['CanalEntradaVF'].map(mapeo)

# Aplica la función a la columna 'A' para asignar valoconsolidado
consolidado['RiesgoProducto'] = consolidado.apply(riesgo_producto, axis=1)
consolidado['TipoPersona'] = 'Juridica'

consolidado['FechaUltimaActualizacionCore'] = consolidado.apply(lambda row: row['FechaApertura'] if pd.isnull(row['FechaUltimaActualizacionCore']) or row['FechaUltimaActualizacionCore'] == "0" else row['FechaUltimaActualizacionCore'], axis=1)





## generación de alertamientos 



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


consolidado = pd.merge(consolidado, perfil_tx_salidas_ahorro[['DocumentoCliente','AlertaPerfiltxahorrosalidas']],on='DocumentoCliente',how='left')


# alertamiento perfil TDC 


perfil_tx_tdc = '''select Documento as DocumentoCliente,[Cantidad de tx promedio al mes],[SD Cantidad de tx al mes],[Monto promedio tx al mes],[SD monto tx al mes],ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
  FROM [Productos y transaccionalidad].[dbo].[PerfilTransaccionalTDC_Facturacion]'''
perfil_tx_tdc = (cx.read_sql(conn = sql_connection, query = perfil_tx_tdc, partition_on="Rank", partition_num=10,  return_type="pandas")).drop(columns=['Rank'])



perfil_tx_tdc = perfil_tx_tdc.groupby('DocumentoCliente')[['Cantidad de tx promedio al mes','SD Cantidad de tx al mes', 'Monto promedio tx al mes','SD monto tx al mes']].quantile(0.9).reset_index()


perfil_tx_tdc['limitecantidadmes'] = perfil_tx_tdc['Cantidad de tx promedio al mes'] + 3 * perfil_tx_tdc['SD Cantidad de tx al mes']
perfil_tx_tdc['limitemontomes'] = perfil_tx_tdc['Monto promedio tx al mes'] + 3 * perfil_tx_tdc['SD monto tx al mes']



perfil_tx_tdc['DocumentoCliente'] = perfil_tx_tdc['DocumentoCliente'].astype(str).str.rstrip('.0')

resultados_salidas_corta_tdc['DocumentoCliente'] = resultados_salidas_corta_tdc['DocumentoCliente'].str.strip()


perfil_tx_tdc=pd.merge(perfil_tx_tdc, resultados_salidas_corta_tdc[['DocumentoCliente','MontoSalidasCortatdc','CantidadSalidasCortatdc']],on='DocumentoCliente',how='left')

perfil_tx_tdc['AlertaPerfiltxtdc'] = np.where(
    (perfil_tx_tdc['MontoSalidasCortatdc'] > perfil_tx_tdc['limitemontomes']) | 
    (perfil_tx_tdc['CantidadSalidasCortatdc'] > perfil_tx_tdc['limitecantidadmes']), 1, 0
)


consolidado = pd.merge(consolidado, perfil_tx_tdc[['DocumentoCliente','AlertaPerfiltxtdc']],on='DocumentoCliente',how='left')

# hasta acá nice 





# alertamiento prepagos

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

prepagos['DocumentoCliente'] = prepagos['DocumentoCliente'].astype(str).str.rstrip('.0')

prepagos = prepagos.groupby('DocumentoCliente')[['AlertaVCRec', 'AlertaDSrest']].sum().reset_index()

consolidado = pd.merge(consolidado, prepagos,on='DocumentoCliente',how='left')




### validacion ultima tx y si el cliente es activo 

consolidado_tx = ''' select DocumentoCliente,FechaTransaccion,CaracterTransaccion,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank  FROM [Productos y transaccionalidad].[dbo].[ConsolidadoTransacciones]'''
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


consolidado = pd.merge(consolidado, consolidado_tx,on='DocumentoCliente',how='left')


def calcular_ahorro(row):
    if (row['Ahorro de la red'] > 0) or (row['FlexiDigital'] > 0) or (row['Nomina Finandina'] > 0) or (row['Otros ahorros'] > 0):
        return 1
    elif row['Cuenta corriente'] > 0:
        return 2
    elif row['CDT'] > 0:
        return 4
    else:
        return 0

# Aplica la función a cada fila y crea la columna 'ahorro' en el DataFrame
consolidado['Ahorro'] = consolidado.apply(calcular_ahorro, axis=1)







atributos_ahorro = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[AtributosAhorro]'''
atributos_ahorro = cx.read_sql(conn = sql_connection, query = atributos_ahorro, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

atributos_ahorro['Ahorro'] = atributos_ahorro['Ahorro'].astype(int)
consolidado = pd.merge(consolidado, atributos_ahorro,on='Ahorro',how='left')


consolidado['MaximoAhorro'] = consolidado[['producto Atributo A', 'producto Atributo B', 'producto Atributo C',
                   'producto Atributo D', 'producto Atributo E', 'producto Atributo F',
                   'producto Atributo G']].max(axis=1)



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

# Aplicar la función a cada fila y crear la nueva columna 'NuevaColumna'
consolidado['Credito'] = consolidado.apply(calcular_credito, axis=1)


atributos_credito = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[AtributosCredito]'''
atributos_credito = cx.read_sql(conn = sql_connection, query = atributos_credito, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])

atributos_credito['Credito'] = atributos_credito['Credito'].astype(int)
consolidado = pd.merge(consolidado, atributos_credito,on='Credito',how='left')


consolidado['MaximoCredito'] = consolidado[['Atributo credito A', 'Atributo credito B', 'Atributo credito C','Atributo credito D', 'Atributo credito E', 'Atributo credito F']].max(axis=1)

consolidado['RIESGO PRODUCTO'] = consolidado[['MaximoAhorro','MaximoCredito' ]].max(axis=1)





#### riesgo jurisdiccion se debe sacar el departamento 





dpto_cup = '''select distinct cussnr as DocumentoCliente,cuna3 as ciudad from openquery(DB2400_182,'select cussnr,cuna3 from BNKPRD01.cup003 ') '''
dpto_cup =cargue_openquery(conn,dpto_cup)



dpto_cup['Departamento'] = dpto_cup['ciudad'].str.upper()
dpto_cup['Departamento'] = dpto_cup['Departamento'].str.strip()

dpto_cup['DocumentoCliente'] = dpto_cup['DocumentoCliente'].astype(str).str.rstrip('.0')





def validar_departamento(dpto):
    if "TOL" in dpto or "TOL." in dpto or "TOLIMA" in dpto:
        return "TOLIMA"
    elif "CUND" in dpto or "CUN." in dpto or "CUNDINAMARCA" in dpto or "CHIA" in dpto or "GIRARDOT" in dpto or "CAJAMARCA" in dpto or "SOACHA" in dpto or "BOJACA" in dpto or "FUSAGASUG" in dpto:
        return "CUNDINAMARCA"
    elif "NAR" in dpto or "NAR." in dpto or "NARIÑO" in dpto:
        return "NARIÑO"
    elif "MAG" in dpto or "MAG." in dpto or "MAGDALENA" in dpto:
        return "MAGDALENA"
    elif "ATL" in dpto or "ATL." in dpto or "ATLANTICO" in dpto or "BARRANQUILLA" in dpto:
        return "ATLANTICO"
    elif "MET" in dpto or "MET." in dpto or "META" in dpto:
        return "META"
    elif "QUIND" in dpto or "QUIND." in dpto or "QUINDIO" in dpto:
        return "QUINDIO"
    elif "ANT" in dpto or "ANT." in dpto or "ANTIOQUIA" in dpto  or "MEDELLIN" in dpto  or "ITAGUI" in dpto or "ENVIGADO" in dpto:
        return "ANTIOQUIA"
    elif "ANT" in dpto or "ANT." in dpto or "ANTIOQUIA" in dpto or "SABANETA" in dpto:
        return "ANTIOQUIA"
    elif "CAS" in dpto or "CAS." in dpto or "CASANARE" in dpto or "YOPAL" in dpto:
        return "CASANARE"
    elif "BOG" in dpto or "BOG." in dpto or "BOGOTA" in dpto or "FONTIBON" in dpto:
        return "BOGOTA"
    elif "RIS" in dpto or "RIS." in dpto or "RISARALDA" in dpto or "PEREIRA" in dpto:
        return "RISARALDA"
    elif "BOY" in dpto or "BOY." in dpto or "BOYACA" in dpto:
        return "BOYACA"
    elif "COR" in dpto or "COR." in dpto or "CORDOBA" in dpto:
        return "CORDOBA"
    elif "BOL" in dpto or "BOL." in dpto or "BOLIVAR" in dpto or "CARTAGENA" in dpto:
        return "BOLIVAR"
    elif "VALLE" in dpto or "CALI" in dpto or "VALLE." in dpto or "B/VENTURA" in dpto or "PALMIRA" in dpto:
        return "VALLE"
    elif "CHOC" in dpto or "CHOC." in dpto or "CHOCO" in dpto:
        return "CHOCO"
    elif "CAUCA" in dpto:
        return "CAUCA"
    elif "CESAR" in dpto:
        return "CESAR"
    elif "GUAJIRA" in dpto:
        return "LA GUAJIRA"
    elif "CALDAS" in dpto or "MANIZALES" in dpto or "LA DORADA" in dpto:
        return "CALDAS"
    elif "NEIVA" in dpto or "HUILA" in dpto:
        return "HUILA"
    elif "CUCUTA" in dpto or "N.S." in dpto or "N.S" in dpto:
        return "NORTE DE SANTANDER "
    elif "CAQ." in dpto or "CAQ" in dpto:
        return "CAQUETA"
    elif "BUCARAMANGA" in dpto or "GIRON" in dpto:
        return "SANTANDER"
    else:
        return np.nan
    


dpto_cup['Departamento'] =dpto_cup['Departamento'].apply(validar_departamento)

dpto_cup=dpto_cup.groupby('DocumentoCliente').first().reset_index()










consolidado=pd.merge(consolidado, dpto_cup[['DocumentoCliente','Departamento']],on='DocumentoCliente',how = 'left')


consolidado['Departamento'] = np.where(consolidado['CiudadActual']=='BOGOTA',consolidado['CiudadActual'],consolidado['Departamento'])

















atributos_jurisdiccion = '''SELECT *,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
	    FROM [ModelosSARLAFT].[dbo].[BaseJurisdiccion]'''
atributos_jurisdiccion = cx.read_sql(conn = sql_connection, query = atributos_jurisdiccion, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])





consolidado = pd.merge(consolidado, atributos_jurisdiccion,on='Departamento', how='left')



## medias informalidad 


cuantiles_informalidad = consolidado['Informalidad'].quantile([0.25, 0.5, 0.75])

rq_informalidad =  cuantiles_informalidad.loc[0.75] - cuantiles_informalidad.loc[0.25]

MEDIAQ1jur = cuantiles_informalidad.loc[0.25] / 2


import numpy as np

# Supongamos que tienes los valoconsolidado DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_informalidad = [
    (consolidado['Informalidad'] <= MEDIAQ1jur),
    (consolidado['Informalidad'] > MEDIAQ1jur) & (consolidado['Informalidad'] <= cuantiles_informalidad.loc[0.25]),
    (consolidado['Informalidad'] > cuantiles_informalidad.loc[0.25]) & (consolidado['Informalidad'] <= cuantiles_informalidad.loc[0.5]),
    (consolidado['Informalidad'] > cuantiles_informalidad.loc[0.5]) & (consolidado['Informalidad'] <= cuantiles_informalidad.loc[0.75]),
    (consolidado['Informalidad'] > cuantiles_informalidad.loc[0.75])
]

# Crea una lista de valoconsolidado para asignar a cada condición
valores_informalidad = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado corconsolidadopondientes
consolidado['Nivel Riesgo informalidad Jurisdiccion'] = np.select(condiciones_informalidad, valores_informalidad, default=0)




## medidas desempleo  




cuantiles_desempleo = consolidado['Desempleo'].quantile([0.25, 0.5, 0.75])

rq_desempleo =  cuantiles_desempleo.loc[0.75] - cuantiles_desempleo.loc[0.25]

MEDIAQ1desempleo = cuantiles_desempleo.loc[0.25] / 2


import numpy as np

# Supongamos que tienes los valoconsolidado DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_desempleo = [
    (consolidado['Desempleo'] <= MEDIAQ1desempleo),
    (consolidado['Desempleo'] > MEDIAQ1desempleo) & (consolidado['Desempleo'] <= cuantiles_desempleo.loc[0.25]),
    (consolidado['Desempleo'] > cuantiles_desempleo.loc[0.25]) & (consolidado['Desempleo'] <= cuantiles_desempleo.loc[0.5]),
    (consolidado['Desempleo'] > cuantiles_desempleo.loc[0.5]) & (consolidado['Desempleo'] <= cuantiles_desempleo.loc[0.75]),
    (consolidado['Desempleo'] > cuantiles_desempleo.loc[0.75])
]

# Crea una lista de valoconsolidado para asignar a cada condición
valores_desempleo = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado corconsolidadopondientes
consolidado['Nivel Riesgo desempleo Jurisdiccion'] = np.select(condiciones_desempleo, valores_desempleo, default=0)






## medidas ITD  




cuantiles_itd = consolidado['ITD'].quantile([0.25, 0.5, 0.75])

rq_itd =  cuantiles_itd.loc[0.75] - cuantiles_itd.loc[0.25]

MEDIAQ1itd = cuantiles_itd.loc[0.25] / 2


import numpy as np

# Supongamos que tienes los valoconsolidado DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_itd = [
    (consolidado['ITD'] <= MEDIAQ1itd),
    (consolidado['ITD'] > MEDIAQ1itd) & (consolidado['ITD'] <= cuantiles_itd.loc[0.25]),
    (consolidado['ITD'] > cuantiles_itd.loc[0.25]) & (consolidado['ITD'] <= cuantiles_itd.loc[0.5]),
    (consolidado['ITD'] > cuantiles_itd.loc[0.5]) & (consolidado['ITD'] <= cuantiles_itd.loc[0.75]),
    (consolidado['ITD'] > cuantiles_itd.loc[0.75])
]

# Crea una lista de valoconsolidado para asignar a cada condición
valores_idt = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado corconsolidadopondientes
consolidado['Nivel Riesgo ITD Jurisdiccion'] = np.select(condiciones_itd, valores_idt, default=0)




## medidas PIB




cuantiles_pib = consolidado['PIB departamental2'].quantile([0.25, 0.5, 0.75])

rq_pib =  cuantiles_pib.loc[0.75] - cuantiles_pib.loc[0.25]

MEDIAQ1pib = cuantiles_pib.loc[0.25] / 2


import numpy as np

# Supongamos que tienes los valoconsolidado DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_pib = [
    (consolidado['PIB departamental2'] <= MEDIAQ1pib),
    (consolidado['PIB departamental2'] > MEDIAQ1pib) & (consolidado['PIB departamental2'] <= cuantiles_pib.loc[0.25]),
    (consolidado['PIB departamental2'] > cuantiles_pib.loc[0.25]) & (consolidado['PIB departamental2'] <= cuantiles_pib.loc[0.5]),
    (consolidado['PIB departamental2'] > cuantiles_pib.loc[0.5]) & (consolidado['PIB departamental2'] <= cuantiles_pib.loc[0.75]),
    (consolidado['PIB departamental2'] > cuantiles_pib.loc[0.75])
]

# Crea una lista de valoconsolidado para asignar a cada condición
valores_pib = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado corconsolidadopondientes
consolidado['Nivel Riesgo PIB departamental Jurisdiccion'] = np.select(condiciones_pib, valores_pib, default=0)



consolidado['Nivel de riesgo JURISDICCION NACIONAL'] = consolidado['Nivel Riesgo informalidad Jurisdiccion'] * 0.35 + consolidado['Nivel Riesgo desempleo Jurisdiccion']* 0.2 + consolidado['Nivel Riesgo ITD Jurisdiccion'] * 0.1 + consolidado['Nivel Riesgo PIB departamental Jurisdiccion'] * 0.35







###### informacion GLPI



informacion_GLPI = '''select A.NumId as DocumentoCliente,B.Ingresos,C.WECodCIIU,ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Rank
from [fabogsqlclu].[LineaProduccion].[dbo].[DB_PERSONA] A
left join [fabogsqlclu].[LineaProduccion].[dbo].[DB_ESTADO_FINANCIERO] B
on A.CodPersona = B.CodPersona
left join  [fabogsqlclu].[LineaProduccion].[dbo].[TMP_PERSONA_ICBS] C
on A.NumId = C.WENroIde'''
informacion_GLPI = cx.read_sql(conn = sql_connection, query = informacion_GLPI, partition_on="Rank", partition_num=10,  return_type="pandas").drop(columns=['Rank'])



informacion_GLPI['DocumentoCliente'] = informacion_GLPI['DocumentoCliente'].astype(str).str.rstrip('.0')

informacion_GLPI['Ingresos'] = informacion_GLPI['Ingresos'].fillna(0)


# Supongamos que tienes un DataFrame llamado 'informacion_GLPI' con columnas 'DocumentoCliente' y 'Ingresos'

# Elimina filas con valores nulos en 'DocumentoCliente'
informacion_GLPI = informacion_GLPI.dropna(subset=['DocumentoCliente'])


# Encuentra los índices de las filas con el valor máximo de 'Ingresos' para cada documento
indices_maximos = informacion_GLPI.groupby('DocumentoCliente')['Ingresos'].idxmax()

# Filtra el DataFrame 'informacion_GLPI' utilizando los índices de las filas máximas
informacion_GLPI = informacion_GLPI.loc[indices_maximos]


consolidado = pd.merge(consolidado, informacion_GLPI,on='DocumentoCliente', how='left')

consolidado['MontoIngresos'] = np.where(consolidado['MontoIngresos'].isnull(), consolidado['Ingresos'], consolidado['MontoIngresos'])

consolidado['CodigoCIIU'] = np.where(consolidado['CodigoCIIU'].isnull(), consolidado['WECodCIIU'], consolidado['CodigoCIIU'])




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



### 


consolidado['CodigoCIIU'] = consolidado['CodigoCIIU'].astype(str).str[:2]


consolidado = pd.merge(consolidado, atributos_actividad_economica,on='CodigoCIIU', how='left')


## validacion Nivel Riesgo Efectivo
cuantiles_efectivo = consolidado['Efectivo'].quantile([0.25, 0.5, 0.75])

rq_efectivo =  cuantiles_efectivo.loc[0.75] - cuantiles_efectivo.loc[0.25]

MEDIAQ1efectivo = cuantiles_efectivo.loc[0.25] / 2


import numpy as np

# Supongamos que tienes los valoconsolidado DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_efectivo = [
    (consolidado['Efectivo'] <= MEDIAQ1efectivo),
    (consolidado['Efectivo'] > MEDIAQ1efectivo) & (consolidado['Efectivo'] <= cuantiles_efectivo.loc[0.25]),
    (consolidado['Efectivo'] > cuantiles_efectivo.loc[0.25]) & (consolidado['Efectivo'] <= cuantiles_efectivo.loc[0.5]),
    (consolidado['Efectivo'] > cuantiles_efectivo.loc[0.5]) & (consolidado['Efectivo'] <= cuantiles_efectivo.loc[0.75]),
    (consolidado['Efectivo'] > cuantiles_efectivo.loc[0.75])
]

# Crea una lista de valoconsolidado para asignar a cada condición
valores_efectivo = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado corconsolidadopondientes
consolidado['Nivel Riesgo Efectivo'] = np.select(condiciones_efectivo, valores_efectivo, default=0)





## validacion exportaciones   Nivel Riesgo CE



cuantiles_CE = consolidado['Comercio Exterior (Exportaciones + Importaciones)'].quantile([0.25, 0.5, 0.75])

rq_CE =  cuantiles_CE.loc[0.75] - cuantiles_CE.loc[0.25]

MEDIAQ1CE = cuantiles_CE.loc[0.25] / 2


import numpy as np

# Supongamos que tienes los valoconsolidado DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_CE = [
    (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] <= MEDIAQ1CE),
    (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] > MEDIAQ1CE) & (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] <= cuantiles_CE.loc[0.25]),
    (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] > cuantiles_CE.loc[0.25]) & (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] <= cuantiles_CE.loc[0.5]),
    (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] > cuantiles_CE.loc[0.5]) & (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] <= cuantiles_CE.loc[0.75]),
    (consolidado['Comercio Exterior (Exportaciones + Importaciones)'] > cuantiles_CE.loc[0.75])
]

# Crea una lista de valoconsolidado para asignar a cada condición
valores_CE = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado corconsolidadopondientes
consolidado['Nivel Riesgo CE'] = np.select(condiciones_CE, valores_CE, default=0)





## medias informalidad AE


cuantiles_informalidad2 = consolidado['Informalidad2'].quantile([0.25, 0.5, 0.75])

rq_informalidad2 =  cuantiles_informalidad2.loc[0.75] - cuantiles_informalidad2.loc[0.25]

MEDIAQ1inf2 = cuantiles_informalidad2.loc[0.25] / 2


import numpy as np

# Supongamos que tienes los valoconsolidado DMEDIAQ1jur, D0.25-quantile(Informalidad), DMedian(Informalidad), y D0.75-quantile(Informalidad)

# Crea una serie de condiciones
condiciones_informalidad2 = [
    (consolidado['Informalidad2'] <= MEDIAQ1inf2),
    (consolidado['Informalidad2'] > MEDIAQ1inf2) & (consolidado['Informalidad2'] <= cuantiles_informalidad2.loc[0.25]),
    (consolidado['Informalidad2'] > cuantiles_informalidad2.loc[0.25]) & (consolidado['Informalidad2'] <= cuantiles_informalidad2.loc[0.5]),
    (consolidado['Informalidad2'] > cuantiles_informalidad2.loc[0.5]) & (consolidado['Informalidad2'] <= cuantiles_informalidad2.loc[0.75]),
    (consolidado['Informalidad2'] > cuantiles_informalidad2.loc[0.75])
]

# Crea una lista de valoconsolidado para asignar a cada condición
valores_informalidad2 = [1, 2, 3, 4, 5]

# Aplica las condiciones y asigna los valoconsolidado corconsolidadopondientes
consolidado['Nivel Riesgo informalidad2'] = np.select(condiciones_informalidad2, valores_informalidad2, default=0)

consolidado['Nivel de Riesgo Actividades Alto Riesgo'] = np.where(consolidado['Actividades de alto riesgo'] =="SI",5,1)
consolidado['Nivel de Riesgo Sarlaft'] = np.where(consolidado['Sarlaft'] =="SI",5,1)

consolidado['PonderacionAtributosAE'] = (
    consolidado['Nivel Riesgo CE'] * 0.2 +
    consolidado['Nivel Riesgo informalidad2'] * 0.2 +
    consolidado['Nivel de Riesgo Actividades Alto Riesgo'] * 0.15 +
    consolidado['Nivel de Riesgo Sarlaft'] * 0.1 +
    consolidado['Nivel Riesgo Efectivo'] * 0.35
)



## riesgo canal 


condicionesRC = [
    consolidado['CanalEntrada'].isna(),
    (consolidado['CanalEntrada'] == "Internet") | (consolidado['CanalEntrada'] == "APP"),
    (consolidado['CanalEntrada'] == "Oficina") | (consolidado['CanalEntrada'] == "Concesionario")
]

valoresRC = [4, 3, 1]

# Utiliza np.where para aplicar las condiciones y asignar valores a la nueva columna 'Resultado'
consolidado['RiesgoCanal'] = np.select(condicionesRC, valoresRC, default=2)




# estimación riesgo inherente 

consolidado['Riesgo inherente del cliente']  = (
    consolidado['RIESGO PRODUCTO'] * 0.2 +
    consolidado['PonderacionAtributosAE'] * 0.35 +
    consolidado['Nivel de riesgo JURISDICCION NACIONAL'] * 0.15 +
    consolidado['RiesgoCanal'] * 0.3
)




##################### 

columnas_a_contar = [
    'CDT', 'Ahorro de la red', 'Cuenta corriente', 'FlexiDigital',
    'Nomina Finandina', 'Otros ahorros', 'TDC Digital', 'TDC Física',
    'Crédito hipotecario', 'Crédito libre inversión', 'Libranza',
    'Crédito vehículo', 'Leasing vehículo', 'Maquina agrícola',
    'Plan mayor', 'Redescuentos', 'Cartera vendida', 'Castigados',
    'Otros activos','Riesgo inherente del cliente'
]



conteos = (pd.DataFrame({'Columna': consolidado[columnas_a_contar].columns, 'Conteo': consolidado[columnas_a_contar].apply(lambda col: (col > 0).sum())}).reset_index(drop=True)).T
conteos.columns = conteos.iloc[0]
conteos = conteos.iloc[1:]
conteos.columns = ['np' + col for col in conteos.columns]






import pandas as pd

# Supongamos que 'consolidado' es tu DataFrame y 'columnas_a_contar' es la lista de columnas que deseas contar y sumar 'Riesgo inherente del cliente' si cumplen con la condición

# Crear un diccionario para almacenar los resultados
resultados = {}

# Iterar sobre las columnas y calcular la suma de 'Riesgo inherente del cliente' para cada una
for columna in columnas_a_contar:
    suma_de_riesgo = consolidado[consolidado[columna] > 0]['Riesgo inherente del cliente'].sum()
    resultados[f'i_p_{columna}'] = suma_de_riesgo

# Crear un DataFrame a partir del diccionario de resultados
sumas_riesgo = pd.DataFrame(resultados, index=[0])


# list(conteos.columns)
# list(sumas_riesgo.columns)

consolidado['RipCDT'] = np.where(consolidado['CDT']>0, (sumas_riesgo['i_p_CDT'].iloc[0]  / conteos['npCDT'].iloc[0]),0)
consolidado['RipAhorro de la red'] = np.where(consolidado['Ahorro de la red']>0, (sumas_riesgo['i_p_Ahorro de la red'].iloc[0]  / conteos['npAhorro de la red'].iloc[0]),0)
consolidado['RipCuenta corriente'] = np.where(consolidado['Cuenta corriente']>0, (sumas_riesgo['i_p_Cuenta corriente'].iloc[0]  / conteos['npCuenta corriente'].iloc[0]),0)
consolidado['RipFlexiDigital'] = np.where(consolidado['FlexiDigital']>0, (sumas_riesgo['i_p_FlexiDigital'].iloc[0]  / conteos['npFlexiDigital'].iloc[0]),0)
consolidado['RipNomina Finandina'] = np.where(consolidado['Nomina Finandina']>0, (sumas_riesgo['i_p_Nomina Finandina'].iloc[0]  / conteos['npNomina Finandina'].iloc[0]),0)
consolidado['RipOtros ahorros'] = np.where(consolidado['Otros ahorros']>0, (sumas_riesgo['i_p_Otros ahorros'].iloc[0]  / conteos['npOtros ahorros'].iloc[0]),0)
consolidado['RipTDC Digital'] = np.where(consolidado['TDC Digital']>0, (sumas_riesgo['i_p_TDC Digital'].iloc[0]  / conteos['npTDC Digital'].iloc[0]),0)
consolidado['RipTDC Física'] = np.where(consolidado['TDC Física']>0, (sumas_riesgo['i_p_TDC Física'].iloc[0]  / conteos['npTDC Física'].iloc[0]),0)
consolidado['RipCrédito hipotecario'] = np.where(consolidado['Crédito hipotecario']>0, (sumas_riesgo['i_p_Crédito hipotecario'].iloc[0]  / conteos['npCrédito hipotecario'].iloc[0]),0)
consolidado['RipCrédito libre inversión'] = np.where(consolidado['Crédito libre inversión']>0, (sumas_riesgo['i_p_Crédito libre inversión'].iloc[0]  / conteos['npCrédito libre inversión'].iloc[0]),0)
consolidado['RipLibranza'] = np.where(consolidado['Libranza']>0, (sumas_riesgo['i_p_Libranza'].iloc[0]  / conteos['npLibranza'].iloc[0]),0)
consolidado['RipCrédito vehículo'] = np.where(consolidado['Crédito vehículo']>0, (sumas_riesgo['i_p_Crédito vehículo'].iloc[0]  / conteos['npCrédito vehículo'].iloc[0]),0)
consolidado['RipLeasing vehículo'] = np.where(consolidado['Leasing vehículo']>0, (sumas_riesgo['i_p_Leasing vehículo'].iloc[0]  / conteos['npLeasing vehículo'].iloc[0]),0)
consolidado['RipMaquina agrícola'] = np.where(consolidado['Maquina agrícola']>0, (sumas_riesgo['i_p_Maquina agrícola'].iloc[0]  / conteos['npMaquina agrícola'].iloc[0]),0)
consolidado['RipPlan mayor'] = np.where(consolidado['Plan mayor']>0, (sumas_riesgo['i_p_Plan mayor'].iloc[0]  / conteos['npPlan mayor'].iloc[0]),0)
consolidado['RipRedescuentos'] = np.where(consolidado['Redescuentos']>0, (sumas_riesgo['i_p_Redescuentos'].iloc[0]  / conteos['npRedescuentos'].iloc[0]),0)
consolidado['RipCartera vendida'] = np.where(consolidado['Cartera vendida']>0, (sumas_riesgo['i_p_Cartera vendida'].iloc[0]  / conteos['npCartera vendida'].iloc[0]),0)
consolidado['RipCastigados'] = np.where(consolidado['Castigados']>0, (sumas_riesgo['i_p_Castigados'].iloc[0]  / conteos['npCastigados'].iloc[0]),0)
consolidado['RipOtros activos'] = np.where(consolidado['Otros activos']>0, (sumas_riesgo['i_p_Otros activos'].iloc[0]  / conteos['npOtros activos'].iloc[0]),0)



# Calcular la suma de las columnas 'Rip' para cada registro y crear una nueva columna 'Suma_Rip'
consolidado['Riesgo inherente compuesto por producto'] = consolidado[[col for col in consolidado.columns if col.startswith('Rip')]].sum(axis=1)


parametro_sturges_pj = math.ceil(math.log(consolidado.shape[0]) / math.log(2))

import pandas as pd
from sklearn.cluster import KMeans

# Supongamos que tienes un DataFrame llamado "jose" y deseas clusterizar la columna "MontoEntradasLarga"
# Asegúrate de que "jose" sea un DataFrame y "MontoEntradasLarga" sea una columna válida en ese DataFrame.

# Llenar los valores faltantes en la columna "MontoEntradasLarga" con ceros
consolidado['MontoEntradasLarga'].fillna(0, inplace=True)
consolidado['MontoEntradasMedia'].fillna(0, inplace=True)
consolidado['MontoEntradasCorta'].fillna(0, inplace=True)

consolidado['MontoSalidasLarga'].fillna(0, inplace=True)
consolidado['MontoSalidasMedia'].fillna(0, inplace=True)
consolidado['MontoSalidasCorta'].fillna(0, inplace=True)




# Número de clusters que deseas crear
n_clusters = parametro_sturges_pj  # Cambia esto según tus necesidades

# Crear y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
consolidado['Clasificacion MontoEntradasLarga'] = kmeans.fit_predict(consolidado[['MontoEntradasLarga']])
consolidado['Clasificacion MontoEntradasMedia'] = kmeans.fit_predict(consolidado[['MontoEntradasMedia']])
consolidado['Clasificacion MontoEntradasCorta'] = kmeans.fit_predict(consolidado[['MontoEntradasCorta']])
consolidado['Clasificacion MontoSalidasLarga'] = kmeans.fit_predict(consolidado[['MontoSalidasLarga']])
consolidado['Clasificacion MontoSalidasMedia'] = kmeans.fit_predict(consolidado[['MontoSalidasMedia']])
consolidado['Clasificacion MontoSalidasCorta'] = kmeans.fit_predict(consolidado[['MontoSalidasCorta']])



entradas_columnas = [ 'Clasificacion MontoEntradasLarga','Clasificacion MontoEntradasMedia','Clasificacion MontoEntradasCorta']

consolidado['Entradas'] = consolidado[entradas_columnas].max(axis=1)

salidas_columnas = ['Clasificacion MontoSalidasLarga','Clasificacion MontoSalidasMedia','Clasificacion MontoSalidasCorta']

consolidado['Salidas'] = consolidado[salidas_columnas].max(axis=1)


tx_columnas = ['Entradas','Salidas']

consolidado['Comportamiento transaccional'] = consolidado[tx_columnas].max(axis=1)


def cuadrante(row):
    RI = row['Riesgo inherente del cliente']
    TX = row['Comportamiento transaccional']
    
    linea1 = int(parametro_sturges_pj / 3)
    linea2 = li_pj * 2
    li_ri= 1.5
    lm_ri = 2.5
    
    if RI <= li_ri:
        if TX <= linea1:
            return 1
        elif TX < linea2:
            return 4
        else:
            return 7
    elif li_ri < RI <= lm_ri:
        if TX <= linea1:
            return 2
        elif TX < linea2:
            return 5
        else:
            return 8
    else:
        if TX <= linea1:
            return 3
        elif TX < linea2:
            return 6
        else:
            return 9




# Aplicar la función a cada fila del DataFrame "consolidado"
consolidado['Cuadrante'] = consolidado.apply(cuadrante, axis=1)



conteo_por_categoria = consolidado['Cuadrante'].value_counts()


list(consolidado.columns)




# validacion de fraudes 





















parametro_sturges_pj = math.ceil(math.log(consolidado.shape[0]) / math.log(2))


# Definir las columnas a clasificar y agregar en una lista
columnas_clasificacion = ['MontoIngresos', 'MontoEntradasLarga', 'MontoEntradasMedia', 'MontoEntradasCorta',
                          'MontoSalidasLarga', 'MontoSalidasMedia', 'MontoSalidasCorta']

# Recorre las columnas y realiza la clasificación
for columna in columnas_clasificacion:
    consolidado[f'Clasificacion {columna}'] = (pd.cut(consolidado[columna], bins=parametro_sturges_pj, labels=False)).fillna(0)

# Clasificación de entradas y salidas
entradas_columnas = [ 'Clasificacion MontoEntradasLarga','Clasificacion MontoEntradasMedia','Clasificacion MontoEntradasCorta']

consolidado['Entradas'] = consolidado[entradas_columnas].max(axis=1)

salidas_columnas = ['Clasificacion MontoSalidasLarga','Clasificacion MontoSalidasMedia','Clasificacion MontoSalidasCorta']

consolidado['Salidas'] = consolidado[salidas_columnas].max(axis=1)


tx_columnas = ['Entradas','Salidas']

consolidado['Comportamiento transaccional'] = consolidado[tx_columnas].max(axis=1)



def cuadrante(row):
    RI = row['Riesgo inherente del cliente']
    TX = row['Comportamiento transaccional']
    
    linea1 = int(parametro_sturges_pj / 3)
    linea2 = linea1 * 2
    li_ri= 1.5
    lm_ri = 2.5
    
    if RI <= li_ri:
        if TX <= linea1:
            return 1
        elif TX < linea2:
            return 4
        else:
            return 7
    elif li_ri < RI <= lm_ri:
        if TX <= linea1:
            return 2
        elif TX < linea2:
            return 5
        else:
            return 8
    else:
        if TX <= linea1:
            return 3
        elif TX < linea2:
            return 6
        else:
            return 9




# Aplicar la función a cada fila del DataFrame "jose"
consolidado['Cuadrante'] = consolidado.apply(cuadrante, axis=1)



list(consolidado.columns)



### hasta acá codigo nice 









##########################

import pyodbc
import pandas as pd

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



auditoria['DocumentoCliente'] = auditoria['DocumentoCliente'].astype(str).str.rstrip('.0')


auditoria['tipologia']=auditoria['tipologia'].str.replace('[^a-zA-Z ]', '', regex=True).str.upper().str.strip()


import pandas as pd
import numpy as np

# Supongamos que tienes un DataFrame 'auditoria' con una columna 'tipologia'

# Lista de correcciones a realizar
correcciones = [
    ('SOSPECHA POR INCUMPLIMIENTO EN PAGO', 'SOSPECHA POR INCUMPLIMIENTO EN PAGOS'),
    ('DOCUMENTACIN FALSA EN RADICACIN', 'DOCUMENTACIN FALSA EN RADICACION'),
    ('DOCUMENTACION FALSA EN RADICACIN', 'DOCUMENTACIN FALSA EN RADICACION'),
    ('SUPLANTACIN','SUPLANTACION')
]

# Realiza las correcciones en un bucle
for antigua, nueva in correcciones:
    auditoria['tipologia'] = np.where(auditoria['tipologia'].str.contains(antigua, case=False, na=False), nueva, auditoria['tipologia'])



# Utiliza la función pivot_table para pivotear el DataFrame
auditoria = auditoria.pivot_table(auditoria, index='DocumentoCliente', columns='tipologia', aggfunc='size', fill_value=0).reset_index()

import pandas as pd


list(auditoria.columns)
# Supongamos que tienes un DataFrame df con las columnas mencionadas

# Lista de las columnas en las que deseas verificar si algún valor es mayor que 0
columnas_verificar = [
    'DOCUMENTACIN FALSA EN RADICACION',
    'DOCUMENTACION FALSA',
    'DOCUMENTACION FALSA INCOCREDITO',
    'EMPRESA FACHADA',
    'PREVENCION',
    'SOSPECHA POR INCUMPLIMIENTO EN PAGOS',
    'SUPLANTACION'
]

# Verificar si al menos una columna es mayor que 0 para cada fila
auditoria['ValidacionFraudes'] = auditoria[columnas_verificar].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)

consolidado = pd.merge(consolidado,auditoria,on='DocumentoCliente',how='left')

import pandas as pd


jose=consolidado


# Supongamos que tienes un DataFrame llamado 'df' con todas las columnas mencionadas

# Lista de columnas que deseas mantener
columnas_deseadas = [
    'DocumentoCliente',
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
    'TipoPersona',
    'CiudadActual',
    'CanalEntrada',
    'FechaVinculacion',
    'MontoIngresos',
    'CodigoCIIU',
    'Celular',
    'Correo',
    'DireccionActual',
    'ActividadEconomica',
    'FechaUltimaActualizacionCore',
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
    'DescripcionSucursal',
    'CanalEntradaVF',
    'VulnerabilidadLavadoActivos',
    'VulnerabilidadTerrorismo',
    'RiesgoJurisdiccion',
    'RiesgoCanal',
    'RiesgoProducto',
    'AlertaPerfiltxahorrosalidas',
    'AlertaPerfiltxtdc',
    'AlertaVCRec',
    'AlertaDSrest',
    'Activo',
    'dias_de_ultima_tx_entrada',
    'dias_de_ultima_tx_salida',
    'dias_de_ultima_tx_otros',
    'RIESGO PRODUCTO',
    'Nivel de riesgo JURISDICCION NACIONAL',
    'Ingresos',
    'Grupo general actividad economica',
    'PonderacionAtributosAE',
    'Riesgo inherente del cliente',
    'Riesgo inherente compuesto por producto',
    'Clasificacion MontoEntradasLarga',
    'Clasificacion MontoEntradasMedia',
    'Clasificacion MontoEntradasCorta',
    'Clasificacion MontoSalidasLarga',
    'Clasificacion MontoSalidasMedia',
    'Clasificacion MontoSalidasCorta',
    'Comportamiento transaccional',
    'Cuadrante',
    'DOCUMENTACIN FALSA EN RADICACION',
    'DOCUMENTACION FALSA',
    'DOCUMENTACION FALSA INCOCREDITO',
    'EMPRESA FACHADA',
    'PREVENCION',
    'SOSPECHA POR INCUMPLIMIENTO EN PAGOS',
    'SUPLANTACION'
]

# Filtra el DataFrame para incluir solo las columnas deseadas
df_filtrado = jose[columnas_deseadas]













####################

































jose = consolidado[['Riesgo inherente del cliente','MontoIngresos', 'MontoEntradasLarga', 'MontoEntradasMedia', 'MontoEntradasCorta',
                          'MontoSalidasLarga', 'MontoSalidasMedia', 'MontoSalidasCorta','Clasificacion MontoEntradasLarga','Clasificacion MontoEntradasMedia','Clasificacion MontoEntradasCorta','Entradas','Clasificacion MontoSalidasLarga','Clasificacion MontoSalidasMedia','Clasificacion MontoSalidasCorta','Salidas']]

 

import pandas as pd
from sklearn.cluster import KMeans

# Supongamos que tienes un DataFrame llamado "jose" y deseas clusterizar la columna "MontoEntradasLarga"
# Asegúrate de que "jose" sea un DataFrame y "MontoEntradasLarga" sea una columna válida en ese DataFrame.

# Llenar los valores faltantes en la columna "MontoEntradasLarga" con ceros
jose['MontoEntradasLarga'].fillna(0, inplace=True)
jose['MontoEntradasMedia'].fillna(0, inplace=True)
jose['MontoEntradasCorta'].fillna(0, inplace=True)

jose['MontoSalidasLarga'].fillna(0, inplace=True)
jose['MontoSalidasMedia'].fillna(0, inplace=True)
jose['MontoSalidasCorta'].fillna(0, inplace=True)




# Número de clusters que deseas crear
n_clusters = 13  # Cambia esto según tus necesidades

# Crear y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
jose['Clasificacion MontoEntradasLarga'] = kmeans.fit_predict(jose[['MontoEntradasLarga']])
jose['Clasificacion MontoEntradasMedia'] = kmeans.fit_predict(jose[['MontoEntradasMedia']])
jose['Clasificacion MontoEntradasCorta'] = kmeans.fit_predict(jose[['MontoEntradasCorta']])
jose['Clasificacion MontoSalidasLarga'] = kmeans.fit_predict(jose[['MontoSalidasLarga']])
jose['Clasificacion MontoSalidasMedia'] = kmeans.fit_predict(jose[['MontoSalidasMedia']])
jose['Clasificacion MontoSalidasCorta'] = kmeans.fit_predict(jose[['MontoSalidasCorta']])



entradas_columnas = [ 'Clasificacion MontoEntradasLarga','Clasificacion MontoEntradasMedia','Clasificacion MontoEntradasCorta']

jose['Entradas'] = jose[entradas_columnas].max(axis=1)

salidas_columnas = ['Clasificacion MontoSalidasLarga','Clasificacion MontoSalidasMedia','Clasificacion MontoSalidasCorta']

jose['Salidas'] = jose[salidas_columnas].max(axis=1)

jose.columns
tx_columnas = ['Entradas','Salidas']

jose['Comportamiento transaccional'] = jose[tx_columnas].max(axis=1)


def cuadrante(row):
    RI = row['Riesgo inherente del cliente']
    TX = row['Comportamiento transaccional']
    
    linea1 = int(parametro_sturges_pj / 3)
    linea2 = linea1 * 2
    li_ri= 1.5
    lm_ri = 2.5
    
    if RI <= li_ri:
        if TX <= linea1:
            return 1
        elif TX < linea2:
            return 4
        else:
            return 7
    elif li_ri < RI <= lm_ri:
        if TX <= linea1:
            return 2
        elif TX < linea2:
            return 5
        else:
            return 8
    else:
        if TX <= linea1:
            return 3
        elif TX < linea2:
            return 6
        else:
            return 9




# Aplicar la función a cada fila del DataFrame "jose"
jose['Cuadrante'] = jose.apply(cuadrante, axis=1)



conteo_por_categoria = jose['Cuadrante'].value_counts()





data = jose[jose['Cuadrante']==9]




prueba = jose[['Riesgo inherente del cliente','Comportamiento transaccional','Cuadrante']]



ruta_completa = r'C:\Users\josgom\Desktop\BORRAR\pruebacuadrante.xlsx'  # Usar 'r' para interpretar la cadena como una ruta cruda

# Exportar el DataFrame a la ubicación deseada
prueba.to_excel(ruta_completa, index=False)




import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que ya tienes un DataFrame "prueba" que contiene las columnas 'Riesgo inherente del cliente',
# 'Comportamiento transaccional' y 'Cuadrante'.

# Crear un gráfico de dispersión y colorear los puntos por cuadrante
plt.figure(figsize=(10, 6))

# Definir colores para cada cuadrante (puedes personalizarlos según tus preferencias)
colores = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 6: 'orange', 7: 'pink', 9: 'gray'}

for cuadrante, color in colores.items():
    subset = prueba[prueba['Cuadrante'] == cuadrante]
    plt.scatter(subset['Riesgo inherente del cliente'], subset['Comportamiento transaccional'], label=f'Cuadrante {cuadrante}', color=color)

# Configurar etiquetas de ejes y título
plt.xlabel('Riesgo inherente del cliente')
plt.ylabel('Comportamiento transaccional')
plt.title('Gráfico de Dispersión con Colores por Cuadrante')

# Mostrar leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()































conteo_por_categoria = consolidado['Comportamiento transaccional'].value_counts()











import pandas as pd
from sklearn.cluster import KMeans

# Supongamos que tienes un DataFrame llamado 'df' con las columnas A, B y C que deseas utilizar para K-Means

# Seleciona las columnas A, B y C para el clustering
datos_clustering = consolidado[['Riesgo inherente del cliente', 'Entradas', 'Salidas']]


# Crea el modelo K-Means
modelo_kmeans = KMeans(n_clusters=parametro_sturges_pj, random_state=0)
# modelo_kmeans = KMeans(n_clusters=3, random_state=0)
# Ajusta el modelo a tus datos y obtén las etiquetas de clúster para cada fila
etiquetas_clusters = modelo_kmeans.fit_predict(datos_clustering)

# Agrega las etiquetas de clúster al DataFrame original
consolidado['Cluster'] = etiquetas_clusters



from sklearn import metrics

inercia = modelo_kmeans.inertia_

# Calcula el índice de Davies-Bouldin
davies_bouldin = metrics.davies_bouldin_score(datos_clustering, etiquetas_clusters)

# Calcula la puntuación Silhouette
silhouette = metrics.silhouette_score(datos_clustering, etiquetas_clusters)

# Muestra las métricas
print(f'Inercia: {inercia}')
print(f'Índice de Davies-Bouldin: {davies_bouldin}')
print(f'Puntuación Silhouette: {silhouette}')



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

import pandas as pd
from sklearn.decomposition import PCA

# Supongamos que tienes un DataFrame llamado 'prueba' con las columnas mencionadas

# Selecciona las columnas 'Entradas' y 'Salidas' para la reducción de dimensionalidad
columnas_reducir = ['Entradas', 'Salidas']
datos_reducir = consolidado[columnas_reducir]

# Crea un objeto PCA con 1 componente principal (una dimensión)
pca = PCA(n_components=1)

# Ajusta el PCA a tus datos
reduccion_dimensional = pca.fit_transform(datos_reducir)

# Agrega una nueva columna 'ReduccionDimensional' al DataFrame original con la reducción
consolidado['ReduccionDimensional'] = abs(reduccion_dimensional)

colores = sns.color_palette("husl", len(consolidado['Cluster'].unique()))

# Graficar Riesgo Inherente vs. Reducción de Dimensionalidad con colores por Cluster
plt.figure(figsize=(10, 6))

for cluster, color in zip(consolidado['Cluster'].unique(), colores):
    subset = consolidado[consolidado['Cluster'] == cluster]
    plt.scatter(subset['Riesgo inherente del cliente'], subset['ReduccionDimensional'], label=f'Cluster {cluster}', c=[color], alpha=0.5)

plt.title('Riesgo Inherente vs. Reducción de Dimensionalidad por Cluster')
plt.xlabel('Riesgo Inherente del Cliente')
plt.ylabel('Reducción de Dimensionalidad')
plt.legend()
plt.grid(True)
plt.show()
list(consolidado.columns)



# medidas estimacion cuadrantes 

consolidado['Riesgo inherente del cliente'].quantile(0.25)
consolidado['Riesgo inherente del cliente'].quantile(0.5)
consolidado['Riesgo inherente del cliente'].quantile(0.75)

    
consolidado['ReduccionDimensional'].quantile(0.25)
consolidado['ReduccionDimensional'].quantile(0.5)
consolidado['ReduccionDimensional'].quantile(0.75)
    















conteo_clusters = consolidado['Cluster'].value_counts()

# Imprime el resultado
print(conteo_clusters)




import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame llamado 'df' con una columna 'A'

# Graficar un histograma de la columna 'A'
plt.figure(figsize=(8, 6))
plt.hist(consolidado['ReduccionDimensional'], bins=20, edgecolor='k')  # Puedes ajustar el número de bins según tus preferencias
plt.title('Histograma de la columna A')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()









max(consolidado['ReduccionDimensional'])










import numpy as np

# Supongamos que tienes las listas de columnas sumas_riesgo.columns y conteos.columns
columnas_sumas_riesgo = sumas_riesgo.columns
columnas_conteos = conteos.columns

# Crear un diccionario para almacenar los resultados
resultados = {}

# Iterar sobre las columnas en sumas_riesgo y conteos
for columna in columnas_sumas_riesgo:
    if columna in columnas_conteos:
        valor_sumas_riesgo = sumas_riesgo[columna].iloc[0]
        valor_conteos = conteos[columna].iloc[0]
        resultados[f'prueba_{columna}'] = np.where(jose['CDT'] > 0, valor_sumas_riesgo / valor_conteos, np.nan)

# Convertir el diccionario de resultados en un DataFrame
df_resultados = pd.DataFrame(resultados)

# Agregar los resultados al DataFrame jose
jose = pd.concat([jose, df_resultados], axis=1)












df.loc[0, 'i_p_CDT']






sum(validacion['Riesgo inherente del cliente'])





import pandas as pd

# Supongamos que 'consolidado' es tu DataFrame y 'columnas_a_contar' es la lista de columnas que deseas contar y sumar 'RI' si cumplen con la condición

# Creamos una función que toma una columna y devuelve la suma de 'RI' si la condición se cumple
def suma_ri_si_condicion(col):
    if (col > 0).any():
        return col['Riesgo inherente del cliente'].sum()
    else:
        return 0

# Aplicamos la función a cada columna y obtenemos el resultado en un DataFrame
conteos2 = pd.DataFrame({
    'Columna': consolidado[columnas_a_contar].columns,
    'Conteo': consolidado[columnas_a_contar].apply(lambda col: (col > 0).sum()),
    'Suma_RI': consolidado[columnas_a_contar].apply(suma_ri_si_condicion)
}).reset_index(drop=True)

conteos2 = conteos2.T  # Transponemos el DataFrame si es necesario


import pandas as pd

# Supongamos que 'consolidado' es tu DataFrame y 'columnas_a_contar' es la lista de columnas que deseas contar y sumar 'RI' si cumplen con la condición

# Creamos una función que toma una columna y devuelve la suma de 'RI' si la condición se cumple
def suma_ri_si_condicion(col):
    if (col > 0).any():
        return col['RI'].sum()
    else:
        return 0

# Aplicamos la función a cada columna y obtenemos el resultado en un DataFrame
conteos2 = pd.DataFrame({
    'Columna': consolidado[columnas_a_contar].columns,
    'Conteo': consolidado[columnas_a_contar].apply(lambda col: (col > 0).sum()),
    'Suma_RI': consolidado[columnas_a_contar].apply(suma_ri_si_condicion)
}).reset_index(drop=True)

conteos2 = conteos2.T  # Transponemos el DataFrame si es necesario




















import pyodbc
import pandas as pd

# Configurar la cadena de conexión (como se mencionó en respuestas anteriores)

connection_string = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};Integrated Security={integrated_security}'

# Intentar establecer la conexión
try:
    conn = pyodbc.connect(connection_string)
    print('Conexión exitosa a SQL Server')
    
    # Consulta SQL para realizar el pivote y contar la cantidad de registros por categoría
    query = """
        SELECT *
        FROM (
            SELECT id_cliente, tipologia
            FROM [AUDITORIA_COMPARTIDA].[PV].[Bd_Fraudes]
        ) AS SourceTable
        PIVOT (
            COUNT(tipologia)
            FOR tipologia IN ([Categoria1], [Categoria2], [Categoria3], ...) -- Reemplaza las categorías
        ) AS PivotTable;
    """
    
    # Ejecutar la consulta y cargar los resultados en un DataFrame
    df = pd.read_sql(query, conn)
    
    # Imprimir los resultados (opcional)
    # print(df)
    
    # No olvides cerrar la conexión cuando hayas terminado
    conn.close()

except Exception as e:
    print(f'Error al conectar a SQL Server: {str(e)}')



















atributos_actividad_economica.columns











jose = consolidado 
vvvv= jose[['CodigoCIIU','CodigoCIIU2']]


list(jose.columns)
[col for col in jose.columns if 'CIIU' in col ]


a=perfil_tx_tdc[perfil_tx_tdc['DocumentoCliente']=='1000002323']





[col for col in perfil_tx_tdc.columns if 'mes' in col ]




















resultados_entradas_corta_ahorro.columns

consolidado2 =  consolidado 


a=paso[paso['DocumentoCliente']=='1020836017']



paso1 = pd.merge(consolidado2, paso[['DocumentoCliente','limitecantidadmes','limitemontomes']],on='DocumentoCliente',how='left')


paso1['AlertaPerfiltxahorrosalidas'] = np.where(paso1['MontoEntradasCorta']>paso1['limitemontomes'] |paso1['CantidadEntradasCorta']>paso1['limitecantidadmes'],1,0 )

[col for  col in  paso1.columns if 'Corta' in col ]


















consolidado.columns

# [col for col in consolidado.columns if 'Monto' in col]

[col for col in consolidado.columns if 'Monto' in col]
