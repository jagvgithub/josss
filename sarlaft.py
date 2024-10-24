# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:46:10 2022

@author: josgom
"""

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

sns.set() 
pd.set_option('mode.chained_assignment',None)
#pd.options.mode.chained_assignment=None

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


# creación de carpeta donde se alojarán los resultados 

# mkdir(str(df[(df.nummonth ==str( dt.datetime.now().month)) & (df.yy == str(dt.datetime.now().year))].iloc[0, 9]))



import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

me = "jose.gomezv@bancofinandina.com"
my_password = r"mpwohrvzgbstkbvk"

#you=["jose.gomezv@bancofinandina.com" , "jesus.alvear@bancofinandina.com","astrid.bermudez@bancofinandina.com"]
you=["jose.gomezv@bancofinandina.com"]
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



