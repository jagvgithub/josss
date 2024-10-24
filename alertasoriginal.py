# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:22:03 2023

@author: josgom
"""


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











# skipped your comments for readability 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

me = "jose.gomezv@bancofinandina.com"
my_password = r"mpwohrvzgbstkbvk"

you=["jose.gomezv@bancofinandina.com" , "jesus.alvear@bancofinandina.com","astrid.bermudez@bancofinandina.com"]
# you=["jose.gomezv@bancofinandina.com"]
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