"""Genera la documentación ficticia de operaciones y el dataset de evaluación.

Escribe 32 documentos en `data/docs/` (29 Markdown y 3 PDF) y 40 preguntas en
`data/eval_questions.jsonl`. Al final carga y trocea los documentos con el pipeline real
y verifica que cada `expected_chunk_id` exista.

Uso:
    uv run scripts/generate_docs.py
"""

from __future__ import annotations

import html
import io
import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest import chunk_documents, load_directory  # noqa: E402

DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "data" / "eval_questions.jsonl"
PDF_IDS = {"RB-11", "SLA-C", "POL-07"}

# (id, título, tipo, fecha, [(encabezado, cuerpo), ...])
DOCS: list[tuple[str, str, str, str, list[tuple[str, str]]]] = [
    (
        "RB-01",
        "Caída del ERP Orion",
        "runbook",
        "2026-01-15",
        [
            (
                "Síntomas",
                "Los usuarios reciben el error ORN-503 al iniciar sesión en el ERP Orion o "
                "las pantallas de pedidos se quedan cargando más de 60 segundos. El monitor de Zabbix "
                "dispara la alerta ERP-HEALTH en rojo.",
            ),
            (
                "Diagnóstico",
                "Verifica el estado del servicio orion-app en los nodos erp-app-01 y "
                "erp-app-02 con systemctl status orion-app. Revisa el pool de conexiones a la base de "
                "datos: si supera 180 de 200 conexiones, el cuello de botella es la base y no la "
                "aplicación.",
            ),
            (
                "Recuperación",
                "Reinicia primero el nodo con más memoria usada, nunca ambos a la vez. "
                "Si tras dos reinicios el error persiste, activa el modo de contingencia: los pedidos "
                "se capturan en la hoja de respaldo y se cargan con el job ORN-IMPORT al restablecerse "
                "el servicio.",
            ),
            (
                "Escalamiento",
                "Si la caída dura más de 30 minutos, se declara P1 y se abre caso con "
                "OrionSoft según el SLA-D. El gerente de guardia avisa a ventas y a los centros de "
                "distribución.",
            ),
        ],
    ),
    (
        "RB-02",
        "Falla del enlace MPLS principal",
        "runbook",
        "2026-02-03",
        [
            (
                "Síntomas",
                "Las sucursales pierden acceso al ERP y al WMS al mismo tiempo, mientras "
                "internet local sigue funcionando. El router de borde rt-core-01 muestra la interfaz "
                "Gi0/1 en estado down.",
            ),
            (
                "Diagnóstico",
                "Haz ping al gateway de NetLink 10.20.0.1 y ejecuta traceroute desde "
                "rt-core-01. Si el último salto que responde es el equipo del proveedor, la falla es "
                "del lado de NetLink.",
            ),
            (
                "Conmutación al enlace secundario",
                "El tráfico pasa por el túnel IPsec sobre el enlace "
                "de internet de Telcom Express (SLA-A). La conmutación es automática con BGP en menos "
                "de 3 minutos; si no ocurre, baja manualmente la prioridad de la ruta MPLS con el "
                "comando documentado en el anexo de red.",
            ),
            (
                "Reporte al proveedor",
                "Abre ticket con NetLink al 800-555-0142 indicando el ID de "
                "circuito MPLS-NL-7781. El contrato SLA-B obliga a NetLink a responder en 30 minutos "
                "para fallas totales.",
            ),
        ],
    ),
    (
        "RB-03",
        "Reinicio del WMS Atlas",
        "runbook",
        "2026-01-28",
        [
            (
                "Cuándo aplicar",
                "Cuando las terminales de radiofrecuencia de los andenes no pueden "
                "confirmar surtidos o el WMS Atlas tarda más de 10 segundos en responder a un escaneo.",
            ),
            (
                "Procedimiento",
                "Avisa a los supervisores de turno para pausar el surtido. Detén el "
                "servicio atlas-core, limpia la cola de mensajes con el script purge_queue.sh y vuelve "
                "a arrancar. El reinicio completo toma unos 7 minutos.",
            ),
            (
                "Validación",
                "Confirma que la cola de tareas pendientes vuelve a cero y realiza un "
                "escaneo de prueba en el andén 1. Registra la hora de inicio y fin en el ticket.",
            ),
        ],
    ),
    (
        "RB-04",
        "Restauración de la base de datos de facturación",
        "runbook",
        "2026-02-20",
        [
            (
                "Alcance",
                "Aplica a la base PostgreSQL db-fact-01 que guarda facturas electrónicas y "
                "notas de crédito.",
            ),
            (
                "Respaldos disponibles",
                "Hay un respaldo completo diario a las 02:00 y archivado de WAL "
                "cada 5 minutos en CloudVault (SLA-I). Esto permite recuperar hasta un punto en el "
                "tiempo con pérdida máxima de 5 minutos.",
            ),
            (
                "Procedimiento de restauración",
                "Restaura el último respaldo completo en un servidor "
                "limpio, aplica los WAL hasta la hora deseada con recovery_target_time y valida conteos "
                "de facturas contra el portal del SAT antes de redirigir la aplicación.",
            ),
            (
                "Autorización",
                "Toda restauración en producción requiere aprobación del líder de "
                "finanzas y un cambio de emergencia registrado según POL-03.",
            ),
        ],
    ),
    (
        "RB-05",
        "Cuenta comprometida o phishing",
        "runbook",
        "2026-03-02",
        [
            (
                "Señales",
                "Un usuario reporta un correo sospechoso en el que ingresó su contraseña, o "
                "el sistema de correo detecta inicios de sesión desde otro país.",
            ),
            (
                "Contención inmediata",
                "Deshabilita la cuenta, revoca todas las sesiones activas y "
                "fuerza el restablecimiento de contraseña y del segundo factor. Bloquea el remitente y "
                "el dominio del correo malicioso en el filtro.",
            ),
            (
                "Investigación",
                "Revisa reglas de reenvío creadas en el buzón, accesos de los últimos "
                "14 días y descargas de archivos compartidos. Conserva la evidencia en la carpeta del "
                "incidente.",
            ),
            (
                "Notificación",
                "Si hubo acceso a información clasificada como Confidencial según "
                "POL-08, el oficial de seguridad notifica a dirección en un plazo máximo de 24 horas.",
            ),
        ],
    ),
    (
        "RB-06",
        "Falla de impresoras de etiquetas en andenes",
        "runbook",
        "2026-01-09",
        [
            (
                "Síntomas",
                "Las impresoras térmicas Zebra ZT411 imprimen etiquetas en blanco, "
                "desfasadas o no reciben trabajos desde el WMS.",
            ),
            (
                "Solución rápida",
                "Recalibra el sensor de medios manteniendo presionados los botones "
                "de pausa y cancelar durante 2 segundos. Si imprime en blanco, revisa que el rollo sea "
                "de transferencia térmica y no térmico directo.",
            ),
            (
                "Cola de impresión",
                "Si no llegan trabajos, revisa la cola PRN-ANDEN en el servidor de "
                "impresión y reinicia el servicio de spooler. Las impresoras de reemplazo están en el "
                "almacén de TI, estante 4.",
            ),
        ],
    ),
    (
        "RB-07",
        "Desincronización de inventario entre WMS y ERP",
        "runbook",
        "2026-03-10",
        [
            (
                "Descripción del problema",
                "El inventario disponible en el ERP Orion no coincide con "
                "las existencias del WMS Atlas. Es el problema conocido registrado como T-1042 y suele "
                "aparecer después de cierres de mes con alto volumen.",
            ),
            (
                "Causa raíz conocida",
                "El job de interfaz INV-SYNC se detiene cuando recibe un SKU con "
                "más de 18 caracteres y deja de procesar los movimientos siguientes sin marcar error.",
            ),
            (
                "Corrección",
                "Identifica el SKU problemático en el log de INV-SYNC, corrígelo en el "
                "maestro de artículos y relanza el job con la opción --desde-ultimo-ok. Después ejecuta "
                "el reporte de conciliación CONC-02 para confirmar que las diferencias son cero.",
            ),
            (
                "Prevención",
                "Desde marzo de 2026 el maestro de artículos valida la longitud del SKU. "
                "El ticket T-1042 sigue abierto hasta que el proveedor libere el parche definitivo.",
            ),
        ],
    ),
    (
        "RB-08",
        "Renovación de certificados TLS",
        "runbook",
        "2026-02-11",
        [
            (
                "Alertas",
                "El monitoreo avisa 30 días antes del vencimiento de cualquier certificado "
                "público y vuelve a avisar a los 7 días con prioridad P2.",
            ),
            (
                "Renovación",
                "Los certificados de los portales de clientes se emiten con la autoridad "
                "DigiCert mediante la cuenta corporativa. Genera el CSR con llave RSA de 2048 bits como "
                "mínimo y súbelo al portal del proveedor.",
            ),
            (
                "Instalación",
                "Instala el certificado y la cadena intermedia en el balanceador "
                "lb-ext-01 y verifica con una prueba SSL externa que la calificación sea A o superior.",
            ),
        ],
    ),
    (
        "RB-09",
        "Disco lleno en servidores de aplicaciones",
        "runbook",
        "2026-01-21",
        [
            ("Umbrales", "Se alerta al 80 % de uso del disco y se considera crítico al 90 %."),
            (
                "Limpieza segura",
                "Comprime y mueve a almacenamiento frío los logs de más de 7 días de "
                "/var/log/apps. Nunca borres archivos de /data/spool porque contienen transacciones "
                "pendientes de enviar.",
            ),
            (
                "Ampliación",
                "Si después de la limpieza el uso sigue arriba del 80 %, solicita "
                "ampliación del volumen al equipo de infraestructura con un cambio estándar.",
            ),
        ],
    ),
    (
        "RB-10",
        "Falla de la VPN de acceso remoto",
        "runbook",
        "2026-02-26",
        [
            (
                "Síntomas",
                "Los usuarios remotos no pueden conectarse al concentrador GW-VPN-02 o se "
                "desconectan cada pocos minutos.",
            ),
            (
                "Diagnóstico",
                "Revisa si el número de túneles activos llegó al límite de licencia de "
                "250 usuarios simultáneos y si el certificado del portal VPN sigue vigente.",
            ),
            (
                "Solución temporal",
                "Habilita el concentrador alterno GW-VPN-03 y comunica a los "
                "usuarios la dirección alterna vpn2.operadora.example. Prioriza a finanzas y a la mesa "
                "de servicio.",
            ),
        ],
    ),
    (
        "RB-11",
        "Corte eléctrico en el centro de datos",
        "runbook",
        "2026-03-05",
        [
            (
                "Respuesta inicial",
                "Al detectar el corte, el UPS sostiene la carga hasta 15 minutos. "
                "El generador diésel debe arrancar automáticamente en menos de 30 segundos; si no "
                "arranca, el técnico de sitio de DataCenter Monterrey lo enciende de forma manual.",
            ),
            (
                "Apagado ordenado",
                "Si el generador no está disponible y el UPS baja del 40 % de carga, "
                "apaga primero los ambientes de pruebas, luego las aplicaciones y al final las bases de "
                "datos.",
            ),
            (
                "Recuperación",
                "Al regresar la energía, enciende en orden inverso: almacenamiento, "
                "bases de datos, aplicaciones y por último servicios de usuario. Valida la integridad "
                "de las bases antes de abrir operaciones.",
            ),
        ],
    ),
    (
        "RB-12",
        "Alta de accesos para personal nuevo",
        "runbook",
        "2026-01-05",
        [
            (
                "Solicitud",
                "Recursos humanos envía el formato ALTA-01 con al menos 3 días hábiles de "
                "anticipación a la fecha de ingreso.",
            ),
            (
                "Accesos base",
                "Todo colaborador recibe correo, VPN y acceso de solo lectura al portal "
                "de documentación. Los accesos al ERP Orion se otorgan por rol y requieren firma del "
                "jefe directo.",
            ),
            (
                "Entrega",
                "La mesa de servicio entrega las credenciales en persona o por llamada, "
                "nunca por correo. El usuario debe activar MFA el primer día según POL-02.",
            ),
        ],
    ),
    (
        "SLA-A",
        "Telcom Express, enlace de internet secundario",
        "sla",
        "2025-11-01",
        [
            (
                "Alcance del servicio",
                "Telcom Express provee un enlace de internet dedicado de "
                "200 Mbps en el centro de datos, usado como respaldo del MPLS de NetLink.",
            ),
            (
                "Disponibilidad",
                "Disponibilidad mensual comprometida de 99.5 %. Las ventanas de "
                "mantenimiento se avisan con 5 días de anticipación.",
            ),
            (
                "Tiempos de atención",
                "Respuesta en 1 hora y resolución en 8 horas para fallas totales. "
                "Soporte telefónico 24x7 al 800-555-0199.",
            ),
        ],
    ),
    (
        "SLA-B",
        "NetLink, enlaces MPLS",
        "sla",
        "2025-10-15",
        [
            (
                "Alcance del servicio",
                "NetLink provee la red MPLS que conecta el centro de datos con "
                "14 sucursales y 3 centros de distribución.",
            ),
            (
                "Disponibilidad",
                "Disponibilidad mensual comprometida de 99.7 % por circuito, medida "
                "por el monitoreo del proveedor y validada contra el nuestro.",
            ),
            (
                "Tiempos de respuesta y resolución",
                "Falla total: respuesta en 30 minutos y resolución "
                "en 4 horas. Degradación: respuesta en 2 horas y resolución en 12 horas.",
            ),
            (
                "Penalizaciones",
                "Por cada hora completa por encima del tiempo de resolución se "
                "descuenta el 5 % de la renta mensual del circuito, con un tope de 30 %. La reclamación "
                "debe hacerse dentro de los 15 días naturales siguientes al cierre del mes.",
            ),
        ],
    ),
    (
        "SLA-C",
        "DataCenter Monterrey, colocation",
        "sla",
        "2025-09-01",
        [
            (
                "Infraestructura",
                "Dos racks de 42U con energía redundante A+B, UPS y generador con "
                "autonomía de 48 horas de diésel.",
            ),
            (
                "Acceso físico",
                "El acceso requiere registro previo en el portal del proveedor con "
                "24 horas de anticipación; para emergencias, el gerente de guardia puede autorizar por "
                "teléfono.",
            ),
            (
                "Compromisos",
                "Disponibilidad de energía de 99.98 %. Temperatura del pasillo frío entre "
                "18 y 27 grados. Técnico en sitio con respuesta de 15 minutos para manos remotas.",
            ),
        ],
    ),
    (
        "SLA-D",
        "OrionSoft, soporte del ERP Orion",
        "sla",
        "2025-12-01",
        [
            (
                "Cobertura",
                "Soporte de aplicación y base de datos para el ERP Orion; incluye parches y "
                "actualizaciones menores.",
            ),
            (
                "Prioridades y tiempos",
                "P1: respuesta en 1 hora, 24x7. P2: respuesta en 4 horas "
                "hábiles. P3 y P4: siguiente día hábil.",
            ),
            (
                "Canal de contacto",
                "Los casos se abren en el portal soporte.orionsoft.example con el "
                "número de cliente OR-55120. Los P1 además se confirman por teléfono.",
            ),
        ],
    ),
    (
        "SLA-E",
        "HelpDesk Plus, mesa de servicio externa",
        "sla",
        "2025-08-20",
        [
            (
                "Horario",
                "Atención de lunes a sábado de 7:00 a 22:00. Fuera de ese horario, las "
                "llamadas se desvían al ingeniero de guardia interno.",
            ),
            (
                "Métricas",
                "Primer contacto resuelto en al menos 70 % de los casos y satisfacción "
                "mínima de 4.2 sobre 5 en la encuesta mensual.",
            ),
            (
                "Alcance",
                "Restablecimiento de contraseñas, configuración de equipos, impresoras de "
                "oficina y altas de usuarios. No atiende fallas de servidores ni de red.",
            ),
        ],
    ),
    (
        "SLA-F",
        "MontaServ, mantenimiento de montacargas",
        "sla",
        "2025-07-01",
        [
            (
                "Mantenimiento preventivo",
                "Cada montacargas recibe servicio preventivo cada 250 horas "
                "de operación o cada 3 meses, lo que ocurra primero.",
            ),
            (
                "Fallas",
                "Para un montacargas detenido, MontaServ debe llegar en 4 horas en zona "
                "metropolitana y 8 horas en centros foráneos. Si la reparación excede 48 horas, debe "
                "proveer un equipo de reemplazo.",
            ),
            (
                "Penalizaciones",
                "Cada día sin reemplazo después de las 48 horas genera una "
                "penalización de 2,000 MXN por equipo.",
            ),
        ],
    ),
    (
        "SLA-G",
        "RutaVeloz, transporte de última milla",
        "sla",
        "2025-10-01",
        [
            (
                "Tiempos de entrega",
                "Entregas locales en 24 horas y foráneas en 48 horas desde la "
                "recolección en el centro de distribución.",
            ),
            (
                "Indicadores",
                "Entregas a tiempo de al menos 95 % mensual y tasa de daño menor a 0.5 % "
                "de los paquetes.",
            ),
            (
                "Incumplimiento",
                "Si el indicador de entregas a tiempo baja de 95 % dos meses seguidos, "
                "se activa una revisión del contrato según POL-09.",
            ),
        ],
    ),
    (
        "SLA-H",
        "VigiPro, seguridad física",
        "sla",
        "2025-06-15",
        [
            (
                "Cobertura",
                "Guardias 24x7 en los tres centros de distribución y rondines cada 2 horas "
                "en el estacionamiento de tráileres.",
            ),
            (
                "Control de accesos",
                "Todo visitante se registra con identificación oficial y recibe "
                "gafete temporal. Los transportistas esperan en la caseta hasta que el andén asignado "
                "esté libre.",
            ),
            (
                "Reporte de incidentes",
                "VigiPro entrega reporte escrito de cualquier incidente en "
                "menos de 2 horas y un resumen mensual con estadísticas.",
            ),
        ],
    ),
    (
        "SLA-I",
        "CloudVault, respaldos en la nube",
        "sla",
        "2025-11-20",
        [
            (
                "Servicio",
                "Almacenamiento de respaldos cifrados con AES-256 en dos regiones "
                "geográficas distintas.",
            ),
            (
                "Retención",
                "Los respaldos diarios se conservan 35 días y los mensuales 12 meses, "
                "alineado con POL-05.",
            ),
            (
                "Restauración",
                "Tiempo de inicio de descarga de un respaldo menor a 1 hora. Pruebas de "
                "restauración trimestrales con evidencia firmada por ambas partes.",
            ),
        ],
    ),
    (
        "POL-01",
        "Gestión de incidentes",
        "politica",
        "2025-09-15",
        [
            (
                "Prioridades",
                "P1: operación detenida en uno o más centros de distribución o sistema "
                "crítico caído. P2: degradación severa con alternativa manual. P3: afecta a un usuario "
                "o área sin detener la operación. P4: solicitudes y consultas.",
            ),
            (
                "Tiempos objetivo internos",
                "Atención de P1 en 15 minutos y comunicación de estado "
                "cada 30 minutos. P2 en 1 hora. P3 en 8 horas hábiles.",
            ),
            (
                "Roles",
                "Cada P1 tiene un coordinador de incidente que dirige la llamada y un "
                "responsable de comunicación con el negocio. Ninguno de los dos debe ejecutar cambios "
                "técnicos.",
            ),
            (
                "Cierre",
                "Un incidente P1 o P2 se cierra solo cuando el usuario confirma el servicio y "
                "se agenda el postmortem según POL-11.",
            ),
        ],
    ),
    (
        "POL-02",
        "Contraseñas y MFA",
        "politica",
        "2025-08-01",
        [
            (
                "Requisitos de contraseña",
                "Mínimo 14 caracteres, sin caducidad periódica salvo "
                "sospecha de compromiso. La cuenta se bloquea 15 minutos tras 10 intentos fallidos.",
            ),
            (
                "Segundo factor",
                "MFA obligatorio para correo, VPN, ERP Orion y consolas de "
                "administración. Se aceptan aplicación autenticadora o llave física; no se permite SMS.",
            ),
            (
                "Cuentas de servicio",
                "Las cuentas de servicio usan contraseñas de 32 caracteres "
                "guardadas en la bóveda de secretos y se rotan cada 180 días.",
            ),
        ],
    ),
    (
        "POL-03",
        "Gestión de cambios",
        "politica",
        "2025-10-10",
        [
            (
                "Tipos de cambio",
                "Estándar: preaprobado y de bajo riesgo. Normal: requiere aprobación "
                "del comité CAB. Emergencia: lo aprueba el gerente de guardia y se documenta en las "
                "siguientes 48 horas.",
            ),
            (
                "Ventanas de mantenimiento",
                "Los cambios normales se ejecutan los martes y jueves de "
                "23:00 a 03:00. No se permiten cambios del día 28 al 3 de cada mes por el cierre "
                "contable.",
            ),
            (
                "Comité CAB",
                "El CAB se reúne los lunes a las 10:00. Las solicitudes deben registrarse "
                "antes del viernes a las 14:00 con plan de reversa probado.",
            ),
        ],
    ),
    (
        "POL-04",
        "Guardias on-call",
        "politica",
        "2025-07-20",
        [
            (
                "Rotación",
                "Las guardias son semanales, de lunes 09:00 a lunes 09:00, con un ingeniero "
                "primario y uno secundario.",
            ),
            (
                "Tiempos de reacción",
                "El primario debe reconocer la alerta en 10 minutos. Si no lo "
                "hace, la plataforma escala automáticamente al secundario y después al gerente de "
                "guardia.",
            ),
            (
                "Compensación",
                "Cada semana de guardia se paga con un bono fijo equivalente a 2 días "
                "de salario, más horas extra por intervenciones fuera de horario mayores a 30 minutos. "
                "Se otorga un día libre después de una noche con más de 3 horas de trabajo.",
            ),
        ],
    ),
    (
        "POL-05",
        "Retención de respaldos",
        "politica",
        "2025-09-01",
        [
            (
                "Periodos de retención",
                "Diarios: 35 días. Mensuales: 12 meses. Anuales de información fiscal: 5 años.",
            ),
            (
                "Pruebas",
                "Cada trimestre se restaura al menos un sistema crítico completo en un "
                "ambiente aislado y se documenta el tiempo real de recuperación.",
            ),
            (
                "Responsables",
                "El equipo de infraestructura opera los respaldos; el dueño de cada "
                "sistema valida la prueba de restauración.",
            ),
        ],
    ),
    (
        "POL-06",
        "Uso aceptable de equipos",
        "politica",
        "2025-06-01",
        [
            (
                "Equipos asignados",
                "Las laptops son propiedad de la empresa y se entregan con disco "
                "cifrado. Está prohibido instalar software sin licencia o desactivar el antivirus.",
            ),
            (
                "Uso personal",
                "Se permite uso personal razonable siempre que no afecte la operación "
                "ni implique almacenar información de la empresa en servicios personales en la nube.",
            ),
            (
                "Pérdida o robo",
                "Reporta la pérdida o robo a la mesa de servicio en menos de 4 horas "
                "para bloquear y borrar el equipo de forma remota.",
            ),
        ],
    ),
    (
        "POL-07",
        "Viáticos para técnicos en campo",
        "politica",
        "2025-12-15",
        [
            (
                "Montos",
                "Alimentos hasta 650 MXN por día en zona metropolitana y 900 MXN en viajes "
                "foráneos. Hospedaje hasta 1,800 MXN por noche.",
            ),
            (
                "Comprobación",
                "Los gastos se comprueban con factura electrónica a nombre de la "
                "empresa dentro de los 5 días hábiles siguientes al regreso.",
            ),
            (
                "Excepciones",
                "Los gastos por encima del tope requieren autorización previa del "
                "gerente de operaciones por correo.",
            ),
        ],
    ),
    (
        "POL-08",
        "Clasificación de información",
        "politica",
        "2025-05-10",
        [
            (
                "Niveles",
                "Pública, Interna, Confidencial y Restringida. Los contratos con proveedores "
                "y los datos de clientes son Confidenciales; la nómina y las llaves criptográficas son "
                "Restringidas.",
            ),
            (
                "Manejo",
                "La información Confidencial solo se comparte con cuentas corporativas y "
                "nunca en memorias USB sin cifrar. La Restringida requiere aprobación del dueño de la "
                "información para cada acceso.",
            ),
            (
                "Etiquetado",
                "Todo documento Confidencial o Restringido lleva la etiqueta del nivel en "
                "el encabezado.",
            ),
        ],
    ),
    (
        "POL-09",
        "Gestión de proveedores",
        "politica",
        "2025-11-05",
        [
            (
                "Evaluación trimestral",
                "Cada proveedor crítico se evalúa cada trimestre con una "
                "calificación de 1 a 5 en cumplimiento de SLA, calidad y comunicación.",
            ),
            (
                "Proveedores críticos",
                "Se consideran críticos NetLink, OrionSoft, DataCenter "
                "Monterrey, CloudVault y RutaVeloz.",
            ),
            (
                "Plan de mejora",
                "Una calificación menor a 3 obliga a un plan de mejora de 60 días. "
                "Dos trimestres seguidos debajo de 3 habilitan la búsqueda de un proveedor alterno.",
            ),
        ],
    ),
    (
        "POL-10",
        "Continuidad del negocio",
        "politica",
        "2025-10-20",
        [
            (
                "Objetivos por sistema",
                "ERP Orion: RTO 4 horas y RPO 15 minutos. WMS Atlas: RTO "
                "2 horas y RPO 5 minutos. Correo: RTO 8 horas y RPO 1 hora.",
            ),
            (
                "Sitio alterno",
                "El sitio alterno en la nube se activa por decisión del director de "
                "operaciones cuando se estima que el centro de datos estará fuera más de 6 horas.",
            ),
            (
                "Simulacros",
                "Se realiza un simulacro de continuidad cada semestre con participación "
                "de TI, operaciones y finanzas.",
            ),
        ],
    ),
    (
        "POL-11",
        "Postmortems",
        "politica",
        "2025-09-25",
        [
            (
                "Cuándo se requiere",
                "Todo incidente P1 y todo P2 con duración mayor a 4 horas requiere postmortem.",
            ),
            (
                "Plazo y formato",
                "El documento se publica en un máximo de 5 días hábiles e incluye "
                "línea de tiempo, causa raíz, impacto y acciones con responsable y fecha.",
            ),
            (
                "Enfoque sin culpables",
                "El análisis se centra en procesos y sistemas, no en personas. "
                "Las acciones correctivas se revisan en la junta semanal de operaciones hasta su "
                "cierre.",
            ),
        ],
    ),
]

# (pregunta, chunk_ids esperados, tipo). `no_answer` no tiene chunk esperado.
QUESTIONS: list[tuple[str, list[str], str]] = [
    ("¿Qué significa el error ORN-503?", ["RB-01#sintomas"], "lexical"),
    (
        "¿Qué es el ticket T-1042?",
        ["RB-07#descripcion-del-problema", "RB-07#prevencion"],
        "lexical",
    ),
    (
        "¿Qué hace el job INV-SYNC cuando recibe un SKU largo?",
        ["RB-07#causa-raiz-conocida"],
        "lexical",
    ),
    (
        "¿A qué teléfono se reporta el circuito MPLS-NL-7781?",
        ["RB-02#reporte-al-proveedor"],
        "lexical",
    ),
    (
        "¿Dónde se abren los casos con el número de cliente OR-55120?",
        ["SLA-D#canal-de-contacto"],
        "lexical",
    ),
    (
        "¿Qué hago si GW-VPN-02 no acepta conexiones?",
        ["RB-10#solucion-temporal", "RB-10#diagnostico", "RB-10#sintomas"],
        "lexical",
    ),
    ("¿Para qué sirve el reporte CONC-02?", ["RB-07#correccion"], "lexical"),
    ("¿Qué es el formato ALTA-01?", ["RB-12#solicitud"], "lexical"),
    ("¿Qué penalización aplica el SLA-B?", ["SLA-B#penalizaciones"], "lexical"),
    ("¿Cómo se recalibra una Zebra ZT411?", ["RB-06#solucion-rapida"], "lexical"),
    ("¿Qué contiene la carpeta /data/spool?", ["RB-09#limpieza-segura"], "lexical"),
    ("¿Qué es la cola PRN-ANDEN?", ["RB-06#cola-de-impresion"], "lexical"),
    ("¿Cuál es el tiempo de respuesta P1 en el SLA-D?", ["SLA-D#prioridades-y-tiempos"], "lexical"),
    ("¿Cada cuánto debo cambiar mi contraseña?", ["POL-02#requisitos-de-contrasena"], "semantic"),
    ("Perdí mi laptop en el aeropuerto, ¿qué hago?", ["POL-06#perdida-o-robo"], "semantic"),
    (
        "¿Cuánto me pagan por estar disponible fuera de horario una semana?",
        ["POL-04#compensacion"],
        "semantic",
    ),
    (
        "¿Puedo hacer cambios en producción a fin de mes?",
        ["POL-03#ventanas-de-mantenimiento"],
        "semantic",
    ),
    (
        "¿Cuánto puedo gastar en comida cuando viajo a otra ciudad por trabajo?",
        ["POL-07#montos"],
        "semantic",
    ),
    (
        "Las sucursales no ven el sistema pero sí tienen internet, ¿qué reviso?",
        ["RB-02#sintomas", "RB-02#diagnostico"],
        "semantic",
    ),
    (
        "Alguien cayó en un correo falso y dio su clave, ¿cuál es el primer paso?",
        ["RB-05#contencion-inmediata"],
        "semantic",
    ),
    (
        "¿Cuánto tiempo aguanta el centro de datos sin luz antes de que arranque la planta?",
        ["RB-11#respuesta-inicial"],
        "semantic",
    ),
    (
        "¿En qué orden se vuelven a prender los equipos cuando regresa la energía?",
        ["RB-11#recuperacion"],
        "semantic",
    ),
    (
        "¿Cuánto tiempo se guardan las copias de seguridad diarias?",
        ["POL-05#periodos-de-retencion", "SLA-I#retencion"],
        "semantic",
    ),
    ("¿Quién coordina una falla grave y qué no debe hacer?", ["POL-01#roles"], "semantic"),
    (
        "¿Cuánto tiempo máximo puede estar caído el sistema del almacén según el plan de "
        "continuidad?",
        ["POL-10#objetivos-por-sistema"],
        "semantic",
    ),
    ("¿Qué hago si las etiquetas salen sin nada impreso?", ["RB-06#solucion-rapida"], "semantic"),
    (
        "¿Con cuánta anticipación debe avisar recursos humanos que entra alguien nuevo?",
        ["RB-12#solicitud"],
        "semantic",
    ),
    (
        "¿Qué pasa si un montacargas se descompone por más de dos días?",
        ["SLA-F#fallas", "SLA-F#penalizaciones"],
        "semantic",
    ),
    (
        "¿Cuál es el SLA del proveedor de enlaces?",
        ["SLA-B#tiempos-de-respuesta-y-resolucion", "SLA-B#disponibilidad"],
        "mixed",
    ),
    ("¿Qué disponibilidad garantiza Telcom Express?", ["SLA-A#disponibilidad"], "mixed"),
    (
        "¿Cada cuánto se rotan las contraseñas de las cuentas de servicio?",
        ["POL-02#cuentas-de-servicio"],
        "mixed",
    ),
    ("¿Cuándo se reúne el CAB?", ["POL-03#comite-cab"], "mixed"),
    ("¿Qué RPO tiene el ERP Orion?", ["POL-10#objetivos-por-sistema"], "mixed"),
    ("¿Cuál es el plazo para publicar un postmortem?", ["POL-11#plazo-y-formato"], "mixed"),
    ("¿Qué proveedores se consideran críticos?", ["POL-09#proveedores-criticos"], "mixed"),
    ("¿Cuántos usuarios simultáneos soporta la VPN?", ["RB-10#diagnostico"], "mixed"),
    ("¿Cuál es la política de trabajo remoto para el personal de oficina?", [], "no_answer"),
    ("¿Cuál es el SLA de Amazon Web Services?", [], "no_answer"),
    ("¿Cuántos días de vacaciones corresponden por año?", [], "no_answer"),
    ("¿Cómo configuro la impresora 3D del laboratorio?", [], "no_answer"),
]


def to_markdown(doc_id: str, title: str, doc_type: str, date: str, sections) -> str:
    lines = ["---", f"id: {doc_id}", f"titulo: {title}", f"tipo: {doc_type}", f"fecha: {date}"]
    lines += ["---", "", f"# {doc_id} · {title}", ""]
    for heading, body in sections:
        lines += [f"## {heading}", "", body, ""]
    return "\n".join(lines)


def to_pdf(path: Path, doc_id: str, title: str, doc_type: str, date: str, sections) -> None:
    parts = [f"<h1>{html.escape(f'{doc_id} · {title}')}</h1>"]
    for heading, body in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2><p>{html.escape(body)}</p>")
    story = pymupdf.Story(html="".join(parts), user_css="body {font-family: sans-serif;}")
    mediabox = pymupdf.paper_rect("letter")
    where = mediabox + (54, 54, -54, -54)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    with pymupdf.open(stream=buffer.getvalue(), filetype="pdf") as pdf:
        pdf.set_metadata({"title": title, "subject": doc_type, "keywords": date})
        pdf.save(path)


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for old in DOCS_DIR.glob("*"):
        if old.suffix in {".md", ".pdf"}:
            old.unlink()
    for doc_id, title, doc_type, date, sections in DOCS:
        if doc_id in PDF_IDS:
            to_pdf(DOCS_DIR / f"{doc_id}.pdf", doc_id, title, doc_type, date, sections)
        else:
            text = to_markdown(doc_id, title, doc_type, date, sections)
            (DOCS_DIR / f"{doc_id}.md").write_text(text, encoding="utf-8")

    with EVAL_PATH.open("w", encoding="utf-8") as f:
        for index, (question, expected, kind) in enumerate(QUESTIONS, start=1):
            row = {
                "id": f"Q{index:02d}",
                "question": question,
                "expected_chunk_ids": expected,
                "kind": kind,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    chunk_ids = {c.chunk_id for c in chunk_documents(load_directory(DOCS_DIR))}
    missing = sorted({cid for _, expected, _ in QUESTIONS for cid in expected} - chunk_ids)
    if missing:
        print(f"chunk_ids esperados que no existen: {missing}", file=sys.stderr)
        return 1
    print(
        f"{len(DOCS)} documentos en {DOCS_DIR} · {len(chunk_ids)} chunks · "
        f"{len(QUESTIONS)} preguntas en {EVAL_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
