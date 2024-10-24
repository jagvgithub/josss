# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:05:56 2024

@author: josgom
"""




'''# Cargue archivo de Riesgo # '''

import pandas as pd

# Ruta al archivo Excel
ruta_archivo = r'\\192.168.60.149\desarrollo y ciencia de datos\RIESGO\Alertamiento_bloqueo_TDC\ALERTAMIENTO_HPAN_CLARO_MES02_28.xlsx'

# Especificar el nombre de la hoja (por ejemplo, 'Sheet1' o 0 para la primera hoja)
nombre_hoja = 'Hoja1'

# Cargar el archivo Excel en un DataFrame de pandas con la hoja especificada
df2 = pd.read_excel(ruta_archivo, sheet_name=nombre_hoja)