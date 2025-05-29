import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time # Optional: for potential loading simulation or checks

# Necesitas instalar deap si no lo tienes: pip install deap
from deap import base, creator, tools, algorithms

# ----------------------- Configuración Inicial DEAP -----------------------
# Estos deben crearse una sola vez. Streamlit ejecuta el script completo en cada interacción,
# por lo que verificamos si ya existen para evitar errores.
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizar un objetivo
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)  # Individuo = lista de números

# ----------------------- Funciones Auxiliares (Adaptadas para recibir parámetros) -----------------------

@st.cache_data # Cache the data loading result
def load_data(uploaded_file):
    """Carga y valida el archivo CSV."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Validar columnas
            if 'Numero' in df.columns and 'Atraso' in df.columns:
                df['Numero'] = df['Numero'].astype(str)
                # Convert Atraso to numeric, coerce errors to NaN, fill NaN with -1, then to int
                df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce').fillna(-1).astype(int)
                st.success("Archivo cargado exitosamente.")
                st.dataframe(df.head())
                return df
            else:
                 st.error("El archivo CSV debe contener las columnas 'Numero' y 'Atraso'.")
                 return None
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            return None
    return None


def generar_combinaciones_con_restricciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones):
    """
    Genera combinaciones con restricciones de atraso, usando un muestreo basado en probabilidad.
    Adaptada para recibir datos y restricciones como argumentos.
    """
    valores = list(distribucion_probabilidad.keys())
    # probabilidades = list(distribucion_probabilidad.values()) # No se usan directamente en el bucle de selección

    combinaciones = []

    for _ in range(n_combinaciones):
        seleccionados = []
        atrasos_seleccionados = Counter()
        usados = set() # Conjunto para rastrear valores ya seleccionados en esta combinación

        # Iterar hasta seleccionar n_selecciones números o hasta que no haya más opciones válidas
        while len(seleccionados) < n_selecciones:
            valores_posibles = []
            probabilidades_posibles = [] # Probabilidades correspondientes a valores_posibles

            # Identificar valores disponibles que cumplen con las restricciones
            for valor, prob in distribucion_probabilidad.items():
                 if valor not in usados: # Solo considerar números no seleccionados aún
                     atraso = numero_a_atraso.get(valor)
                     # Verificar si el atraso es válido (-1 indica error) y si la restricción lo permite
                     # Usamos str(atraso) porque las claves de restricciones pueden ser strings
                     if atraso is not None and atraso != -1 and atrasos_seleccionados.get(str(atraso), 0) < restricciones_atraso.get(str(atraso), n_selecciones):
                          valores_posibles.append(valor)
                          probabilidades_posibles.append(prob)

            # Si no hay valores posibles que cumplan, romper el bucle para esta combinación
            if not valores_posibles:
                break

            # Normalizar probabilidades de los valores posibles
            total_probabilidades_posibles = sum(probabilidades_posibles)
            if total_probabilidades_posibles == 0: # Evitar división por cero
                break
            probabilidades_posibles_normalized = [p / total_probabilidades_posibles for p in probabilidades_posibles]


            # Elegir un nuevo valor basado en las probabilidades normalizadas
            nuevo_valor = random.choices(valores_posibles, weights=probabilidades_posibles_normalized, k=1)[0]

            # Añadir el valor seleccionado a la combinación y marcarlo como usado
            seleccionados.append(nuevo_valor)
            usados.add(nuevo_valor)

            # Actualizar el contador de atrasos seleccionados para esta combinación
            atraso = numero_a_atraso.get(nuevo_valor)
            if atraso is not None and atraso != -1:
                atrasos_seleccionados[str(atraso)] += 1 # Usar string key

        # Si se completó una combinación válida (con el número correcto de selecciones)
        if len(seleccionados) == n_selecciones:
            seleccionados.sort()
            combinaciones.append(tuple(seleccionados))
        # Si no, la combinación es incompleta y se descarta automáticamente

    # Procesar las combinaciones válidas generadas para contar frecuencias y calcular probabilidades
    conteo_combinaciones = Counter(combinaciones)
    probabilidad_combinaciones = {}
    for combinacion, frecuencia in conteo_combinaciones.items():
        # Calcular la probabilidad de la combinación como el producto de las probabilidades individuales
        prob_comb = 1.0
        try:
            prob_comb = np.prod([distribucion_probabilidad.get(val, 0) for val in combinacion])
        except Exception:
            # En caso de error al calcular la probabilidad (ej: valor no encontrado), asignar 0
            prob_comb = 0

        probabilidad_combinaciones[combinacion] = (frecuencia, prob_comb)

    # Ordenar las combinaciones por probabilidad (descendente) y luego por frecuencia (descendente)
    combinaciones_ordenadas = sorted(
        probabilidad_combinaciones.items(),
        key=lambda x: (-x[1][1], -x[1][0])
    )

    return combinaciones_ordenadas

# @st.cache_resource # Potentially cache executor, but might interfere with streamlit thread model
def procesar_combinaciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, n_ejecuciones):
    """Ejecuta la generación de combinaciones en paralelo."""
    resultados_por_ejecucion = []
    task_args = (distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones)

    # Usar ProcessPoolExecutor para paralelizar
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *task_args) for _ in range(n_ejecuciones)]

        # Monitorear progreso (opcional)
        # No es trivial mostrar un progreso perfecto con as_completed en Streamlit sin actualizar la UI constantemente
        # Pero podemos esperar a que terminen.
        for i, future in enumerate(as_completed(futures)):
            resultados_por_ejecucion.append(future.result())
            # st.text(f"Completada ejecución {i+1}/{n_ejecuciones}") # Esto refrescaría mucho la UI

    return resultados_por_ejecucion


def encontrar_combinaciones_coincidentes(resultados_por_ejecucion):
    """Encuentra combinaciones que aparecen en todas las ejecuciones."""
    if not resultados_por_ejecucion:
        return {}

    # Diccionario para guardar las combinaciones y en cuáles ejecuciones aparecieron (usando un set para eficiencia)
    combinaciones_encontradas = {}
    for i, resultado in enumerate(resultados_por_ejecucion):
         # resultado es una lista de (combinacion_tuple, (frecuencia, probabilidad))
         for combinacion_tuple, _ in resultado:
              if combinacion_tuple not in combinaciones_encontradas:
                  combinaciones_encontradas[combinacion_tuple] = {i}
              else:
                  combinaciones_encontradas[combinacion_tuple].add(i)

    # Filtrar combinaciones que aparecieron en todas las ejecuciones
    num_total_ejecuciones = len(resultados_por_ejecucion)
    combinaciones_coincidentes = {
            comb: sorted(list(ejecuciones_set)) # Convertir set a lista ordenada para la salida
            for comb, ejecuciones_set in combinaciones_encontradas.items() if len(ejecuciones_set) == num_total_ejecuciones
    }

    return combinaciones_coincidentes

# --- Funciones para el Algoritmo Genético (Adaptadas para recibir parámetros) ---

def generar_individuo_deap(distribucion_prob, num_atraso, restr_atraso, n_sel):
    """Genera un individuo (combinación) válido para DEAP."""
    valores = list(distribucion_prob.keys())
    combinacion = []
    atrasos_seleccionados = Counter()
    usados = set() # Usar set para rastrear números usados

    while len(combinacion) < n_sel:
        valores_posibles = []
        # En GA, la generación inicial puede ser uniforme entre los posibles
        # La evaluación y selección se encargan de la probabilidad

        for valor in valores:
            if valor not in usados:
                atraso = num_atraso.get(valor)
                # Verificar si atraso es válido y si la restricción lo permite
                if atraso is not None and atraso != -1 and atrasos_seleccionados.get(str(atraso), 0) < restr_atraso.get(str(atraso), n_sel):
                    valores_posibles.append(valor)

        if not valores_posibles:
            # No se pueden añadir más números que cumplan las restricciones
            break

        # Elegir uniformemente entre los valores posibles
        nuevo_valor = random.choice(valores_posibles)

        combinacion.append(nuevo_valor)
        usados.add(nuevo_valor)

        atraso = num_atraso.get(nuevo_valor)
        if atraso is not None and atraso != -1:
            atrasos_seleccionados[str(atraso)] += 1

    # Asegurarse de que el individuo tenga el tamaño correcto (si no, no será válido)
    # y convertirlo al tipo Individual de DEAP.
    # Si el bucle se rompió antes de n_sel, devuelve una lista incompleta que será penalizada.
    return creator.Individual(sorted(combinacion))


def evaluar_individuo_deap(individuo, distribucion_prob, num_atraso, restr_atraso, n_sel):
    """Función de evaluación (fitness) para DEAP."""
    # Penalizar individuos que no tienen el tamaño correcto
    if not isinstance(individuo, list) or len(individuo) != n_sel:
        return (0,)

    # Verificar restricciones de atraso
    atrasos_seleccionados = Counter([num_atraso.get(val) for val in individuo if num_atraso.get(val) is not None and num_atraso.get(val) != -1])
    for atraso_str, cantidad in atrasos_seleccionados.items():
        # Usar str(atraso) para coincidir con las claves de restricciones
        if cantidad > restr_atraso.get(atraso_str, n_sel):
            return (0,)  # Penalizar si no cumple las restricciones

    # Calcular la probabilidad de la combinación (fitness)
    probabilidad = 1.0
    try:
        # Usar .get con valor por defecto 0 en caso de que algún número no esté en la distribución
        probabilidad = np.prod([distribucion_prob.get(val, 0) for val in individuo])
    except Exception:
        # En caso de error (ej: datos inconsistentes), penalizar
        probabilidad = 0

    return (probabilidad,)  # Devolver la probabilidad como fitness (tupla)


def ejecutar_algoritmo_genetico(n_generaciones, n_poblacion, cxpb, mutpb, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones=6):
    """Ejecuta el algoritmo genético con DEAP."""

    # Configurar el toolbox *antes* de la ejecución, usando los parámetros actuales
    toolbox = base.Toolbox()
    toolbox.register("individual", generar_individuo_deap, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo_deap, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)
    toolbox.register("mate", tools.cxTwoPoint)
    # Mutación que mezcla los índices. La función de fitness penalizará si el resultado es inválido.
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crear la población inicial
    # Manejar el caso donde no se puedan generar individuos válidos (ej: restricciones imposibles)
    try:
        population = toolbox.population(n=n_poblacion)
        if not population:
             return None, 0.0, "No se pudo generar una población inicial válida con las restricciones dadas."

    except Exception as e:
        return None, 0.0, f"Error al crear la población inicial: {e}"


    # Ejecutar el algoritmo genético
    # Desactivar verbose para no llenar la consola de Streamlit
    algorithms.eaSimple(population, toolbox,
                          cxpb=cxpb,
                          mutpb=mutpb,
                          ngen=n_generaciones,
                          stats=None, # No mostrar estadísticas detalladas por generación en la UI
                          halloffame=None, # No necesitamos un hall of fame separado para este caso
                          verbose=False) # Silenciar salida de DEAP

    # Obtener el mejor individuo de la población final
    # selBest devuelve una lista, tomamos el primer elemento
    if population:
        best_ind = tools.selBest(population, k=1)[0]
        best_fitness = evaluar_individuo_deap(best_ind, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)[0]
        return best_ind, best_fitness, None # Devolver el mejor, su fitness y sin error
    else:
        return None, 0.0, "La población se volvió vacía durante la ejecución del AG."


# ----------------------- Interfaz de Streamlit -----------------------

st.title("Generador de Combinaciones de Números con Restricciones")
st.write("Esta aplicación te ayuda a encontrar combinaciones de números, considerando sus 'atrasos' y restricciones definidas.")

# --- Carga de Datos ---
st.header("1. Cargar Datos de Atraso")
uploaded_file = st.file_uploader("Sube tu archivo CSV (debe contener las columnas 'Numero' y 'Atraso')", type="csv")

df = load_data(uploaded_file) # Usar la función cacheada

numero_a_atraso = {}
atrasos_disponibles = []
distribucion_probabilidad = {} # Se generará después de cargar los datos

if df is not None:
    # Mapear número a atraso
    # Filtrar filas donde Atraso sea -1 (errores de conversión)
    df_valid = df[df['Atraso'] != -1].copy()
    numero_a_atraso = dict(zip(df_valid['Numero'], df_valid['Atraso']))

    # Obtener atrasos disponibles
    atrasos_disponibles = sorted(list(set(df_valid['Atraso'].tolist())))

    # Generar distribución de probabilidad uniforme para los números válidos
    # Asume que todos los números válidos tienen la misma probabilidad
    numeros_validos = list(numero_a_atraso.keys())
    if numeros_validos:
        prob_por_numero = 1.0 / len(numeros_validos)
        distribucion_probabilidad = {num: prob_por_numero for num in numeros_validos}
        st.info(f"Distribución de probabilidad uniforme generada para {len(numeros_validos)} números válidos encontrados.")
        st.write(f"Atrasos disponibles en los datos: {atrasos_disponibles}")
    else:
        st.warning("No se encontraron números válidos con atraso en el archivo. No se pueden realizar cálculos.")
        numero_a_atraso = {}
        atrasos_disponibles = []
        distribucion_probabilidad = {}


# --- Configuración de Parámetros y Restricciones ---
st.header("2. Configurar Parámetros")

n_selecciones = 6 # Número fijo de selecciones por combinación

st.subheader("Restricciones de Atraso")
st.write(f"Define la cantidad máxima de números permitida para cada valor de 'Atraso' en una combinación de {n_selecciones} números.")

restricciones_atraso_config = {}
if atrasos_disponibles:
    st.info(f"Selecciona los valores de 'Atraso' que quieres restringir y establece un límite (entre 0 y {n_selecciones}).")
    # Puedes pre-seleccionar algunos atrasos comunes o dejarlo vacío
    default_selected_atrasos = [str(a) for a in [0, 2, 4, 6, 8, 10, 12, 14] if a in atrasos_disponibles] # Ejemplo de default
    selected_atrasos_to_restrict = st.multiselect(
        "Selecciona los valores de 'Atraso' a restringir:",
        options=[str(a) for a in atrasos_disponibles], # Asegurarse que las opciones son strings
        default=default_selected_atrasos if default_selected_atrasos else ([str(atrasos_disponibles[0])] if atrasos_disponibles else []) # Ejemplo: seleccionar el primero si hay, o nada
    )

    # Diccionario para las restricciones finales (usamos claves str)
    restricciones_finales = {}
    if selected_atrasos_to_restrict:
         st.write("Define los límites:")
         for atraso_str in selected_atrasos_to_restrict:
            # Intentar obtener un valor por defecto si existe en las restricciones de ejemplo del código original
            default_limit_example = {'0':20, '2':12, '4':2, '6':3, '8':5, '10':2, '12':1, '14':1}.get(atraso_str, n_selecciones)
            limit = st.number_input(
                f"Máximo permitido para Atraso '{atraso_str}':",
                min_value=0,
                max_value=n_selecciones,
                value=default_limit_example,
                step=1,
                key=f"restriction_{atraso_str}" # Clave única para cada input
            )
            restricciones_finales[atraso_str] = limit

    st.write("Restricciones configuradas:", restricciones_finales if restricciones_finales else "Ninguna")

else:
    st.info("Carga un archivo CSV con datos de 'Atraso' para configurar las restricciones.")
    restricciones_finales = {}


# --- Parámetros de Algoritmos ---
st.subheader("Parámetros del Algoritmo Genético")
ga_ngen = st.slider("Número de Generaciones", 10, 500, 100) # Rango ampliado
ga_npob = st.slider("Tamaño de la Población", 100, 2000, 500) # Rango ampliado
ga_cxpb = st.slider("Probabilidad de Cruce (CXPB)", 0.0, 1.0, 0.7, 0.05) # Valor típico 0.5-0.9
ga_mutpb = st.slider("Probabilidad de Mutación (MUTPB)", 0.0, 1.0, 0.1, 0.01) # Valor típico 0.1-0.3


st.subheader("Parámetros de la Simulación Concurrente")
sim_n_combinaciones = st.number_input("Número de Combinaciones por Ejecución", min_value=1000, value=100000, step=5000)
sim_n_ejecuciones = st.number_input("Número de Ejecuciones Concurrentes", min_value=1, value=5, step=1)

# --- Ejecución del Algoritmo Genético ---
st.header("3. Ejecutar Algoritmo Genético")

if df is not None and distribucion_probabilidad and numero_a_atraso:
    if st.button("Ejecutar GA para encontrar la combinación más probable"):
        if not restricciones_finales:
            st.warning("No se han definido restricciones de atraso. Esto podría afectar los resultados.")

        st.info("Ejecutando Algoritmo Genético...")
        with st.spinner(f"Buscando mejor combinación por {ga_ngen} generaciones..."):
            # Pasamos los parámetros y datos necesarios a la función
            mejor_individuo, mejor_fitness, error_msg = ejecutar_algoritmo_genetico(
                n_generaciones=ga_ngen,
                n_poblacion=ga_npob,
                cxpb=ga_cxpb,
                mutpb=ga_mutpb,
                distribucion_prob=distribucion_probabilidad,
                numero_a_atraso=numero_a_atraso,
                restricciones_atraso=restricciones_finales,
                n_selecciones=n_selecciones
            )

        if error_msg:
             st.error(error_msg)
        elif mejor_individuo is not None:
            st.subheader("Mejor Combinación Encontrada por el Algoritmo Genético")
            st.write("Combinación:", " - ".join(map(str, mejor_individuo)))
            st.write("Fitness (Probabilidad Calculada):", f"{mejor_fitness:.12f}") # Mostrar más decimales para probabilidad

            # Mostrar atrasos de la combinación encontrada
            atrasos_best_ind = Counter([numero_a_atraso.get(val) for val in mejor_individuo if numero_a_atraso.get(val) is not None and numero_a_atraso.get(val) != -1])
            st.write("Distribución de Atrasos en esta combinación:", dict(atrasos_best_ind))
        else:
            st.warning("El algoritmo genético no pudo encontrar una combinación válida.")

else:
    st.info("Carga y configura los datos para ejecutar el Algoritmo Genético.")


# --- Ejecución de la Simulación Concurrente ---
st.header("4. Ejecutar Simulación Concurrente")

if df is not None and distribucion_probabilidad and numero_a_atraso:
     if st.button(f"Ejecutar Simulación ({sim_n_ejecuciones} ejecuciones)"):
        if not restricciones_finales:
            st.warning("No se han definido restricciones de atraso. Esto podría afectar los resultados.")

        st.info(f"Ejecutando {sim_n_ejecuciones} simulaciones concurrentes generando {sim_n_combinaciones} combinaciones por ejecución...")
        with st.spinner("Generando y procesando combinaciones en paralelo..."):
            try:
                resultados_por_ejecucion = procesar_combinaciones(
                    distribucion_probabilidad,
                    numero_a_atraso,
                    restricciones_finales,
                    n_selecciones,
                    sim_n_combinaciones,
                    sim_n_ejecuciones
                )

                combinaciones_coincidentes = encontrar_combinaciones_coincidentes(resultados_por_ejecucion)

                st.subheader("Combinaciones Coincidentes en Todas las Simulaciones")

                if combinaciones_coincidentes:
                    # Preparar datos para DataFrame
                    coincident_list = []
                    # Recorrer las combinaciones coincidentes para obtener su probabilidad y frecuencia total (si aplica)
                    # La probabilidad se puede obtener de los resultados de cualquier ejecución
                    # La frecuencia total requeriría sumar las frecuencias de todas las ejecuciones,
                    # pero aquí solo nos interesa que apareció en todas, no cuántas veces *en total*.
                    # La frecuencia mostrada será solo la de la primera ejecución donde se encontró.

                    # Crear un mapa rápido de combinación -> (frecuencia, probabilidad) de la primera ejecución
                    first_run_results_map = dict(resultados_por_ejecucion[0]) if resultados_por_ejecucion else {}


                    for comb_tuple, ejecuciones_list in combinaciones_coincidentes.items():
                        # Obtener la probabilidad y frecuencia de la primera ejecución donde apareció
                        freq_prob = first_run_results_map.get(comb_tuple, (0, 0.0)) # Default a 0 si no se encuentra (no debería pasar si está en coincidentes)
                        frecuencia_en_primera_ejecucion = freq_prob[0]
                        probabilidad_calculada = freq_prob[1]

                        # Calcular atrasos para esta combinación
                        atrasos_comb = Counter([numero_a_atraso.get(val) for val in comb_tuple if numero_a_atraso.get(val) is not None and numero_a_atraso.get(val) != -1])
                        atrasos_str = ", ".join([f"A{k}:{v}" for k, v in atrasos_comb.items()])

                        coincident_list.append({
                            "Combinación": " - ".join(map(str, comb_tuple)),
                            "Probabilidad": probabilidad_calculada,
                            f"Apariciones (de {sim_n_ejecuciones} ejec.)": len(ejecuciones_list),
                            "Atrasos": atrasos_str
                        })

                    # Ordenar por probabilidad descendente
                    coincident_list_sorted = sorted(coincident_list, key=lambda x: x["Probabilidad"], reverse=True)

                    st.dataframe(coincident_list_sorted, height=400) # Mostrar como tabla

                else:
                    st.info("No se encontraron combinaciones que aparecieran en *todas* las simulaciones con las restricciones dadas.")

            except Exception as e:
                 st.error(f"Ocurrió un error durante la ejecución de la Simulación Concurrente: {e}")

else:
     st.info("Carga y configura los datos para ejecutar la Simulación Concurrente.")


# --- Notas/Explicación Adicional ---
st.sidebar.header("Información")
st.sidebar.markdown("""
**Propósito:**
Ayuda a encontrar combinaciones de números (ej. para loterías) basándose en datos históricos de 'atraso' y restricciones definidas por el usuario.

**Métodos:**
1.  **Algoritmo Genético:** Busca una única combinación "óptima" que maximice su probabilidad (basada en la distribución cargada) mientras cumple las restricciones de atraso.
2.  **Simulación Concurrente:** Genera muchas combinaciones aleatorias (respetando restricciones y probabilidad) en paralelo múltiples veces. Identifica combinaciones que son lo suficientemente robustas como para aparecer en *todas* las ejecuciones.

**Datos:**
Requiere un archivo CSV con al menos las columnas 'Numero' y 'Atraso'.
""")