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

# Use session_state for caching to ensure data persists across reruns caused by user interaction
# This is often better than st.cache_data for mutable data or objects like DFs that change state
# Streamlit recommends st.cache_data/resource but session_state can be simpler for this use case
# if you want the DF to be readily available without recalculating or hashing.
# However, for simple loading, st.cache_data is generally preferred if possible.
# Let's stick to st.cache_data for the load function as recommended by Streamlit best practices.
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
                # Using -1 to signify invalid/missing atraso
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

    combinaciones = []

    for _ in range(n_combinaciones):
        seleccionados = []
        atrasos_seleccionados_counter = Counter() # Usamos Counter directamente para los atrasos seleccionados
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
                     # Usamos str(atraso) porque las claves de restricciones son strings
                     # Si un atraso no tiene restricción explícita, el .get() devolverá n_selecciones, permitiendo cualquier cantidad hasta 6.
                     if atraso is not None and atraso != -1 and atrasos_seleccionados_counter.get(str(atraso), 0) < restricciones_atraso.get(str(atraso), n_selecciones):
                          valores_posibles.append(valor)
                          probabilidades_posibles.append(prob)

            # Si no hay valores posibles que cumplan, romper el bucle para esta combinación
            if not valores_posibles:
                break

            # Normalizar probabilidades de los valores posibles
            total_probabilidades_posibles = sum(probabilidades_posibles)
            if total_probabilidades_posibles == 0: # Evitar división por cero
                # This can happen if all remaining possible values have probability 0
                break
            probabilidades_posibles_normalized = [p / total_probabilidades_posibles for p in probabilidades_posibles]


            # Elegir un nuevo valor basado en las probabilidades normalizadas
            # Usar k=1 para obtener solo un elemento y acceder a él con [0]
            try:
                 nuevo_valor = random.choices(valores_posibles, weights=probabilidades_posibles_normalized, k=1)[0]
            except IndexError: # random.choices might return empty list if weights are somehow problematic
                 break # Cannot select, break the loop

            # Añadir el valor seleccionado a la combinación y marcarlo como usado
            seleccionados.append(nuevo_valor)
            usados.add(nuevo_valor)

            # Actualizar el contador de atrasos seleccionados para esta combinación
            atraso = numero_a_atraso.get(nuevo_valor)
            if atraso is not None and atraso != -1:
                atrasos_seleccionados_counter[str(atraso)] += 1 # Usar string key for consistency with restrictions

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
            # Usar .get con valor por defecto 0 en caso de que algún número no esté en la distribución
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
    # max_workers=None usa el número de CPUs
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *task_args) for _ in range(n_ejecuciones)]

        # Monitorear progreso (opcional) - Difficult to update progress bar inside as_completed easily in Streamlit
        # without potential UI blocking or complex threading. Simple spinner is better.
        for future in as_completed(futures):
            # futures.index(future) # This is inefficient to get index inside as_completed
            resultados_por_ejecucion.append(future.result())
            # If you wanted to show progress, you'd need a different approach,
            # like storing futures and checking if done, updating a progress bar outside the loop.

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
    atrasos_seleccionados_counter = Counter()
    usados = set() # Usar set para rastrear números usados

    # Shuffle valores initially to add randomness to the starting point
    random.shuffle(valores)

    while len(combinacion) < n_sel:
        valores_posibles = []
        # Only iterate through values not already used in this individual
        for valor in valores:
            if valor not in usados:
                atraso = num_atraso.get(valor)
                # Verificar si atraso es válido y si la restricción lo permite
                if atraso is not None and atraso != -1 and atrasos_seleccionados_counter.get(str(atraso), 0) < restr_atraso.get(str(atraso), n_sel):
                    valores_posibles.append(valor)

        if not valores_posibles:
            # No se pueden añadir más números que cumplan las restricciones
            break

        # Elegir uniformemente entre los valores posibles para la generación inicial
        nuevo_valor = random.choice(valores_posibles)

        combinacion.append(nuevo_valor)
        usados.add(nuevo_valor)

        atraso = num_atraso.get(nuevo_valor)
        if atraso is not None and atraso != -1:
            atrasos_seleccionados_counter[str(atraso)] += 1

    # Asegurarse de que el individuo tenga el tamaño correcto (si no, no será válido)
    # y convertirlo al tipo Individual de DEAP.
    # Si el bucle se rompió antes de n_sel, devuelve una lista incompleta que será penalizada por la evaluación.
    # Return as a list first, it will be converted to creator.Individual by the toolbox init.
    return sorted(combinacion)


def evaluar_individuo_deap(individuo, distribucion_prob, num_atraso, restr_atraso, n_sel):
    """Función de evaluación (fitness) para DEAP."""
    # Penalizar individuos que no tienen el tamaño correcto
    if not isinstance(individuo, list) or len(individuo) != n_sel:
        return (0,)

    # Verificar restricciones de atraso
    atrasos_seleccionados_counter = Counter([num_atraso.get(val) for val in individuo if num_atraso.get(val) is not None and num_atraso.get(val) != -1])
    for atraso_str, cantidad in atrasos_seleccionados_counter.items():
        # Usar str(atraso) para coincidir con las claves de restricciones
         # Get restriction for this atraso, default to n_sel if not explicitly restricted
        restriction = restr_atraso.get(atraso_str, n_sel)
        if cantidad > restriction:
            return (0,)  # Penalizar si no cumple las restricciones

    # Calcular la probabilidad de la combinación (fitness)
    probabilidad = 1.0
    try:
        # Usar .get con valor por defecto 0 en caso de que algún número no esté en la distribución
        # If any value in the individual is not in the probability distribution, probability is 0.
        if any(val not in distribucion_prob for val in individuo):
             return (0,)
        probabilidad = np.prod([distribucion_prob[val] for val in individuo])
    except Exception:
        # Should not happen if `any` check passes, but as a safeguard
        probabilidad = 0

    return (probabilidad,)  # Devolver la probabilidad como fitness (tupla)


def ejecutar_algoritmo_genetico(n_generaciones, n_poblacion, cxpb, mutpb, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones=6):
    """Ejecuta el algoritmo genético con DEAP."""

    # Configurar el toolbox *antes* de la ejecución, usando los parámetros actuales
    # Register functions using lambda to pass arguments
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: generar_individuo_deap(distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo_deap, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)
    toolbox.register("mate", tools.cxTwoPoint)
    # Mutación que mezcla los índices. La función de fitness penalizará si el resultado es inválido.
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crear la población inicial
    # Manejar el caso donde no se puedan generar individuos válidos (ej: restricciones imposibles)
    population = []
    attempts = 0
    max_attempts = n_poblacion * 10 # Try creating individuals for a limited number of times
    with st.spinner("Generando población inicial para el Algoritmo Genético..."):
        while len(population) < n_poblacion and attempts < max_attempts:
            try:
                ind = toolbox.individual()
                if len(ind) == n_selecciones: # Only add if the individual generator succeeded in creating a full individual
                   population.append(ind)
                # else: # Optionally print why an individual was not valid from generator
                   # print(f"Skipping incomplete individual during init: {ind}")

            except Exception as e:
                 # print(f"Error generating individual: {e}") # For debugging generator issues
                 pass # Silently fail generation if issues
            attempts += 1


    if not population:
        return None, 0.0, "No se pudo generar una población inicial válida con las restricciones dadas. Revisa las restricciones o los datos."

    # Ejecutar el algoritmo genético
    st.info(f"Población inicial generada ({len(population)} individuos válidos). Ejecutando AG...")
    # Desactivar verbose para no llenar la consola de Streamlit
    try:
        algorithms.eaSimple(population, toolbox,
                          cxpb=cxpb,
                          mutpb=mutpb,
                          ngen=n_generaciones,
                          stats=None, # No mostrar estadísticas detalladas por generación en la UI
                          halloffame=None, # No necesitamos un hall of fame separado para este caso
                          verbose=False) # Silenciar salida de DEAP
    except Exception as e:
        return None, 0.0, f"Error durante la ejecución del Algoritmo Genético: {e}"


    # Obtener el mejor individuo de la población final
    # selBest devuelve una lista, tomamos el primer elemento
    if population:
        # Evaluate the final population to ensure fitness values are up-to-date
        # DEAP's eaSimple does this internally for the final population
        best_ind = tools.selBest(population, k=1)[0]
        best_fitness = best_ind.fitness.values[0] if best_ind.fitness.valid else evaluar_individuo_deap(best_ind, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)[0]

        # Double check fitness just in case
        evaluated_fitness = evaluar_individuo_deap(best_ind, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)[0]
        if best_fitness != evaluated_fitness:
             # This indicates an issue, maybe fitness was not updated correctly or evaluation is inconsistent
             # For safety, use the re-evaluated fitness
             best_fitness = evaluated_fitness
             # print(f"Warning: DEAP fitness mismatch. Using re-evaluated fitness: {best_fitness}") # Debugging

        return best_ind, best_fitness, None # Devolver el mejor, su fitness y sin error
    else:
        return None, 0.0, "La población se volvió vacía o inválida durante la ejecución del AG."


# ----------------------- Interfaz de Streamlit -----------------------

st.title("Generador de Combinaciones de Números con Restricciones")
st.write("Esta aplicación te ayuda a encontrar combinaciones de números, considerando sus 'atrasos' y restricciones definidas.")

# --- Carga de Datos ---
st.header("1. Cargar Datos de Atraso")
uploaded_file = st.file_uploader("Sube tu archivo CSV (debe contener las columnas 'Numero' y 'Atraso')", type="csv")

df = load_data(uploaded_file) # Usar la función cacheada

numero_a_atraso = {}
atrasos_disponibles = []
distribucion_probabilidad = {}
atraso_counts = {} # Diccionario para almacenar la cuenta de cada atraso

if df is not None:
    # Mapear número a atraso y filtrar filas inválidas
    df_valid = df[df['Atraso'] != -1].copy()
    numero_a_atraso = dict(zip(df_valid['Numero'], df_valid['Atraso']))

    # Obtener atrasos disponibles (únicos y válidos) y su conteo
    if not df_valid.empty:
        atrasos_disponibles = sorted(list(set(df_valid['Atraso'].tolist())))
        # Calculate counts and convert keys to string for consistency with restrictions
        atraso_counts = df_valid['Atraso'].value_counts().rename(index=str).to_dict()


    # Generar distribución de probabilidad uniforme para los números válidos
    # Asume que todos los números válidos tienen la misma probabilidad
    numeros_validos = list(numero_a_atraso.keys())
    if numeros_validos:
        prob_por_numero = 1.0 / len(numeros_validos)
        distribucion_probabilidad = {num: prob_por_numero for num in numeros_validos}
        st.info(f"Distribución de probabilidad uniforme generada para {len(numeros_validos)} números válidos encontrados.")
        st.write(f"Atrasos disponibles en los datos válidos: {atrasos_disponibles}")
        st.write("Conteo de cada atraso en los datos:", atraso_counts)
    else:
        st.warning("No se encontraron números válidos con atraso en el archivo (verifica columnas 'Numero' y 'Atraso'). No se pueden realizar cálculos.")
        numero_a_atraso = {}
        atrasos_disponibles = []
        distribucion_probabilidad = {}
        atraso_counts = {}


# --- Configuración de Parámetros y Restricciones ---
st.header("2. Configurar Parámetros")

n_selecciones = 6 # Número fijo de selecciones por combinación

st.subheader(f"Restricciones de Atraso (para combinaciones de {n_selecciones} números)")
st.write("Define la cantidad máxima de números permitida en una combinación para cada valor de 'Atraso'.")
st.info("Los valores por defecto se basan en la cantidad de números con cada atraso en tu archivo.")

restricciones_finales = {}
if atrasos_disponibles:
    st.write("Configura los límites para cada atraso disponible:")
    # Aseguramos que los atrasos disponibles en la UI son strings
    atrasos_disponibles_str = sorted([str(a) for a in atrasos_disponibles])

    for atraso_str in atrasos_disponibles_str:
        # Get the default value from the calculated counts, default to 0 if somehow missing
        default_limit = atraso_counts.get(atraso_str, 0)
        limit = st.number_input(
            f"Máximo permitido para Atraso '{atraso_str}':",
            min_value=0,
            # Max value is the total number of selections
            max_value=n_selecciones,
            value=default_limit, # Set the default value based on count
            step=1,
            key=f"restriction_{atraso_str}" # Clave única para cada input
        )
        # Store the user's chosen limit (could be the default or a changed value)
        restricciones_finales[atraso_str] = limit

    st.write("Restricciones configuradas:", restricciones_finales if restricciones_finales else "Ninguna restricción específica aplicada.")

else:
    st.info("Carga un archivo CSV con datos de 'Atraso' para configurar las restricciones.")
    restricciones_finales = {}


# --- Parámetros de Algoritmos ---
st.subheader("Parámetros del Algoritmo Genético")
ga_ngen = st.slider("Número de Generaciones", 10, 1000, 100) # Rango ampliado, 100-500 es común
ga_npob = st.slider("Tamaño de la Población", 100, 5000, 500) # Rango ampliado, 300-1000 es común
ga_cxpb = st.slider("Probabilidad de Cruce (CXPB)", 0.0, 1.0, 0.7, 0.05) # Valor típico 0.5-0.9
ga_mutpb = st.slider("Probabilidad de Mutación (MUTPB)", 0.0, 1.0, 0.1, 0.01) # Valor típico 0.1-0.3


st.subheader("Parámetros de la Simulación Concurrente")
sim_n_combinaciones = st.number_input("Número de Combinaciones por Ejecución", min_value=1000, value=100000, step=5000)
sim_n_ejecuciones = st.number_input("Número de Ejecuciones Concurrentes", min_value=1, value=5, step=1)

# --- Ejecución del Algoritmo Genético ---
st.header("3. Ejecutar Algoritmo Genético")

# Only enable button if necessary data is loaded
if df is not None and distribucion_probabilidad and numero_a_atraso:
    if st.button("Ejecutar GA para encontrar la combinación más probable"):
        if not restricciones_finales:
            st.warning("No se han definido restricciones de atraso. Esto significa que cualquier combinación válida de 6 números será considerada por el AG.")

        st.info("Iniciando Algoritmo Genético...")
        # Pass all necessary parameters and data
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

# Only enable button if necessary data is loaded
if df is not None and distribucion_probabilidad and numero_a_atraso:
     if st.button(f"Ejecutar Simulación ({sim_n_ejecuciones} ejecuciones)"):
        if not restricciones_finales:
            st.warning("No se han definido restricciones de atraso. Esto significa que la simulación generará combinaciones sin tener en cuenta límites por atraso.")

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
                        # .get() will handle cases where the key might not be in the first run's map (shouldn't happen for coincident ones, but safe)
                        freq_prob = first_run_results_map.get(comb_tuple, (0, 0.0))
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
                    st.info(f"No se encontraron combinaciones que aparecieran en *todas* las {sim_n_ejecuciones} simulaciones con las restricciones dadas.")

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
