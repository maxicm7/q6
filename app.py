import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
def load_data_and_counts(uploaded_file):
    """Carga, valida el archivo CSV, calcula las cuentas de atraso y la suma total."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Validar columnas
            if 'Numero' in df.columns and 'Atraso' in df.columns:
                df['Numero'] = df['Numero'].astype(str)
                # Convert Atraso to numeric, coerce errors to NaN, fill NaN with -1, then to int
                df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce').fillna(-1).astype(int)

                # Filter out invalid atrasos before mapping, counting, and summing
                df_valid = df[df['Atraso'] != -1].copy()

                if df_valid.empty:
                     st.warning("No se encontraron filas válidas con 'Numero' y 'Atraso' numérico.")
                     # Retorna valores vacíos y suma 0
                     return None, {}, {}, [], {}, 0

                st.success("Archivo cargado exitosamente.")
                st.dataframe(df.head())

                # Mapear número a atraso para los datos válidos
                numero_a_atraso = dict(zip(df_valid['Numero'], df_valid['Atraso']))

                # Obtener atrasos disponibles (como enteros)
                atrasos_disponibles_int = sorted(list(set(df_valid['Atraso'].tolist())))

                # Generar distribución de probabilidad uniforme para los números válidos
                numeros_validos = list(numero_a_atraso.keys())
                prob_por_numero = 1.0 / len(numeros_validos) if numeros_validos else 0
                distribucion_probabilidad = {num: prob_por_numero for num in numeros_validos}

                # Calcular las cuentas de cada atraso (usando strings como claves)
                atraso_counts = df_valid['Atraso'].astype(str).value_counts().to_dict()
                st.info(f"Distribución de probabilidad uniforme generada para {len(numeros_validos)} números válidos.")
                st.write(f"Atrasos disponibles en los datos: {atrasos_disponibles_int}")
                st.write("Conteo de cada atraso encontrado:", atraso_counts)

                # Calcular la suma total de todos los atrasos en el dataset
                total_atraso_dataset = df_valid['Atraso'].sum()


                # Retornar todos los datos procesados, incluyendo los conteos y la suma total
                return df_valid, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset

            else:
                 st.error("El archivo CSV debe contener las columnas 'Numero' y 'Atraso'.")
                 # Retorna valores vacíos y suma 0 en caso de error
                 return None, {}, {}, [], {}, 0
        except Exception as e:
            st.error(f"Error al leer o procesar el archivo CSV: {e}")
            # Retorna valores vacíos y suma 0 en caso de error
            return None, {}, {}, [], {}, 0
    # Retorna valores vacíos y suma 0 si no hay archivo
    return None, {}, {}, [], {}, 0


def generar_combinaciones_con_restricciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones):
    """
    Genera combinaciones con restricciones de atraso, usando un muestreo basado en probabilidad.
    Adaptada para recibir datos y restricciones como argumentos.
    """
    valores = list(distribucion_probabilidad.keys())
    combinaciones = []

    for _ in range(n_combinaciones):
        seleccionados = []
        atrasos_seleccionados = Counter()
        usados = set()

        while len(seleccionados) < n_selecciones:
            valores_posibles = []
            probabilidades_posibles = []

            for valor, prob in distribucion_probabilidad.items():
                 if valor not in usados:
                     atraso = numero_a_atraso.get(valor)
                     # Verificar si el atraso es válido y si la restricción lo permite
                     if atraso is not None and atraso != -1 and atrasos_seleccionados.get(str(atraso), 0) < restricciones_atraso.get(str(atraso), n_selecciones):
                          valores_posibles.append(valor)
                          probabilidades_posibles.append(prob)

            if not valores_posibles:
                break

            total_probabilidades_posibles = sum(probabilidades_posibles)
            if total_probabilidades_posibles == 0:
                break
            probabilidades_posibles_normalized = [p / total_probabilidades_posibles for p in probabilidades_posibles]

            nuevo_valor = random.choices(valores_posibles, weights=probabilidades_posibles_normalized, k=1)[0]

            seleccionados.append(nuevo_valor)
            usados.add(nuevo_valor)

            atraso = numero_a_atraso.get(nuevo_valor)
            if atraso is not None and atraso != -1:
                atrasos_seleccionados[str(atraso)] += 1

        if len(seleccionados) == n_selecciones:
            seleccionados.sort()
            combinaciones.append(tuple(seleccionados))

    conteo_combinaciones = Counter(combinaciones)
    probabilidad_combinaciones = {}
    for combinacion, frecuencia in conteo_combinaciones.items():
        prob_comb = 1.0
        try:
            # Usar .get(val, 0) en caso de que un valor no se encuentre (aunque con validación previa no debería pasar)
            prob_comb = np.prod([distribucion_probabilidad.get(val, 0) for val in combinacion])
        except Exception:
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

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *task_args) for _ in range(n_ejecuciones)]

        for i, future in enumerate(as_completed(futures)):
            resultados_por_ejecucion.append(future.result())

    return resultados_por_ejecucion


def encontrar_combinaciones_coincidentes(resultados_por_ejecucion):
    """Encuentra combinaciones que aparecen en todas las ejecuciones."""
    if not resultados_por_ejecucion:
        return {}

    combinaciones_encontradas = {}
    for i, resultado in enumerate(resultados_por_ejecucion):
         for combinacion_tuple, _ in resultado:
              if combinacion_tuple not in combinaciones_encontradas:
                  combinaciones_encontradas[combinacion_tuple] = {i}
              else:
                  combinaciones_encontradas[combinacion_tuple].add(i)

    num_total_ejecuciones = len(resultados_por_ejecucion)
    combinaciones_coincidentes = {
            comb: sorted(list(ejecuciones_set))
            for comb, ejecuciones_set in combinaciones_encontradas.items() if len(ejecuciones_set) == num_total_ejecuciones
    }

    return combinaciones_coincidentes

# --- Funciones para el Algoritmo Genético (Adaptadas para recibir parámetros) ---

def generar_individuo_deap(distribucion_prob, num_atraso, restr_atraso, n_sel):
    """Genera un individuo (combinación) válido para DEAP."""
    valores = list(distribucion_prob.keys())
    combinacion = []
    atrasos_seleccionados = Counter()
    usados = set()

    while len(combinacion) < n_sel:
        valores_posibles = []
        for valor in valores:
            if valor not in usados:
                atraso = num_atraso.get(valor)
                if atraso is not None and atraso != -1 and atrasos_seleccionados.get(str(atraso), 0) < restr_atraso.get(str(atraso), n_sel):
                    valores_posibles.append(valor)

        if not valores_posibles:
            break

        if not valores_posibles: # This check is redundant due to the first break, but kept for clarity
             break

        nuevo_valor = random.choice(valores_posibles)

        combinacion.append(nuevo_valor)
        usados.add(nuevo_valor)

        atraso = num_atraso.get(nuevo_valor)
        if atraso is not None and atraso != -1:
            atrasos_seleccionados[str(atraso)] += 1

    return creator.Individual(sorted(combinacion))


def evaluar_individuo_deap(individuo, distribucion_prob, num_atraso, restr_atraso, n_sel):
    """Función de evaluación (fitness) para DEAP."""
    if not isinstance(individuo, list) or len(individuo) != n_sel:
        return (0,)

    atrasos_seleccionados = Counter([num_atraso.get(val) for val in individuo if num_atraso.get(val) is not None and num_atraso.get(val) != -1])
    for atraso_str, cantidad in atrasos_seleccionados.items():
        if cantidad > restr_atraso.get(atraso_str, n_sel):
            return (0,)

    probabilidad = 1.0
    try:
        probabilidad = np.prod([distribucion_prob.get(val, 0) for val in individuo])
    except Exception:
        probabilidad = 0

    return (probabilidad,)


def ejecutar_algoritmo_genetico(n_generaciones, n_poblacion, cxpb, mutpb, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones=6):
    """Ejecuta el algoritmo genético con DEAP."""

    toolbox = base.Toolbox()
    toolbox.register("individual", generar_individuo_deap, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo_deap, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    try:
        population = toolbox.population(n=n_poblacion)
        if not population:
             return None, 0.0, "No se pudo generar una población inicial válida con las restricciones dadas (posiblemente restricciones imposibles)."

    except Exception as e:
        return None, 0.0, f"Error al crear la población inicial: {e}"

    algorithms.eaSimple(population, toolbox,
                          cxpb=cxpb,
                          mutpb=mutpb,
                          ngen=n_generaciones,
                          stats=None,
                          halloffame=None,
                          verbose=False)

    if population:
        best_ind = tools.selBest(population, k=1)[0]
        best_fitness = evaluar_individuo_deap(best_ind, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)[0]
        return best_ind, best_fitness, None
    else:
        return None, 0.0, "La población se volvió vacía durante la ejecución del AG."


# ----------------------- Interfaz de Streamlit -----------------------

st.title("Generador de Combinaciones de Números con Restricciones")
st.write("Esta aplicación te ayuda a encontrar combinaciones de números, considerando sus 'atrasos' y restricciones definidas.")

# --- Carga de Datos ---
st.header("1. Cargar Datos de Atraso")
uploaded_file = st.file_uploader("Sube tu archivo CSV (debe contener las columnas 'Numero' y 'Atraso')", type="csv")

# Modificamos la llamada para obtener los conteos Y la suma total
df, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset = load_data_and_counts(uploaded_file)

n_selecciones = 6 # Número fijo de selecciones por combinación

# Mostrar la suma total del dataset después de cargar
if df is not None and total_atraso_dataset is not None:
     st.info(f"**Suma total de todos los 'Atraso' en el dataset cargado:** {total_atraso_dataset}")


# --- Configuración de Parámetros y Restricciones ---
st.header("2. Configurar Parámetros")

st.subheader("Restricciones de Atraso")
st.write(f"Define la cantidad máxima de números permitida para cada valor de 'Atraso' en una combinación de {n_selecciones} números.")

restricciones_finales = {}
if atrasos_disponibles_int:
    st.info("Selecciona los valores de 'Atraso' a restringir. El valor por defecto es la cantidad de números con ese atraso en tus datos.")
    # Pre-seleccionar *todos* los atrasos disponibles por defecto
    selected_atrasos_to_restrict = st.multiselect(
        "Selecciona los valores de 'Atraso' a restringir:",
        options=[str(a) for a in atrasos_disponibles_int], # Opciones como string
        default=[str(a) for a in atrasos_disponibles_int] # Default a todos como string
    )

    if selected_atrasos_to_restrict:
         st.write("Define los límites para los atrasos seleccionados:")
         for atraso_str in selected_atrasos_to_restrict:
            # El valor por defecto es el conteo de ese atraso en los datos
            # Usamos .get() con 0 como fallback si por alguna razón el atraso seleccionado no tiene conteo (no debería pasar si viene de atraso_counts)
            default_limit = atraso_counts.get(atraso_str, 0)
            limit = st.number_input(
                f"Máximo permitido para Atraso '{atraso_str}' (por defecto: {default_limit}):",
                min_value=0,
                # El máximo lógico para una restricción en una combinación de 6 es 6
                max_value=n_selecciones,
                value=default_limit, # Usar el conteo como valor inicial
                step=1,
                key=f"restriction_{atraso_str}"
            )
            # Asegurarse de guardar la restricción como string key -> int limit
            restricciones_finales[atraso_str] = limit

    st.write("Restricciones configuradas:", restricciones_finales if restricciones_finales else "Ninguna")

else:
    st.info("Carga un archivo CSV con datos de 'Atraso' para configurar las restricciones.")
    restricciones_finales = {}


# --- Parámetros de Algoritmos ---
st.subheader("Parámetros del Algoritmo Genético")
# Ajustar rangos y valores por defecto si es necesario
ga_ngen = st.slider("Número de Generaciones", 10, 1000, 200)
ga_npob = st.slider("Tamaño de la Población", 100, 5000, 1000)
ga_cxpb = st.slider("Probabilidad de Cruce (CXPB)", 0.0, 1.0, 0.7, 0.05)
ga_mutpb = st.slider("Probabilidad de Mutación (MUTPB)", 0.0, 1.0, 0.2, 0.01)


st.subheader("Parámetros de la Simulación Concurrente")
sim_n_combinaciones = st.number_input("Número de Combinaciones por Ejecución", min_value=1000, value=200000, step=10000) # Aumentado por defecto
sim_n_ejecuciones = st.number_input("Número de Ejecuciones Concurrentes", min_value=1, value=8, step=1) # Aumentado por defecto

# --- Ejecución del Algoritmo Genético ---
st.header("3. Ejecutar Algoritmo Genético")

# Solo permitir ejecución si hay datos válidos cargados
if numero_a_atraso and distribucion_probabilidad and total_atraso_dataset is not None:
    if st.button("Ejecutar GA para encontrar la combinación más probable"):
        if not restricciones_finales:
            st.warning("No se han definido restricciones de atraso. Esto podría afectar los resultados.")

        st.info("Ejecutando Algoritmo Genético...")
        with st.spinner(f"Buscando mejor combinación por {ga_ngen} generaciones..."):
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
            st.write("Fitness (Probabilidad Calculada):", f"{mejor_fitness:.12f}")

            # Calcular suma de atrasos para el mejor individuo
            # Asegurarse de que los números del individuo estén en numero_a_atraso antes de sumar
            suma_atrasos_mejor_individuo = sum(numero_a_atraso.get(val, 0) for val in mejor_individuo)
            st.write("Suma de Atrasos de esta combinación:", suma_atrasos_mejor_individuo)

            # Calcular el valor especial
            valor_especial_mejor_individuo = total_atraso_dataset + 40 - suma_atrasos_mejor_individuo
            st.write(f"**Cálculo Especial:** ({total_atraso_dataset} [Suma Atrasos Dataset] + 40 - {suma_atrasos_mejor_individuo} [Suma Atrasos Combinación]) = **{valor_especial_mejor_individuo}**")

            atrasos_best_ind_counts = Counter([numero_a_atraso.get(val) for val in mejor_individuo if numero_a_atraso.get(val) is not None and numero_a_atraso.get(val) != -1])
            st.write("Distribución de Atrasos (Conteos) en esta combinación:", dict(atrasos_best_ind_counts))
        else:
            st.warning("El algoritmo genético no pudo encontrar una combinación válida.")

else:
    st.info("Carga un archivo CSV válido para ejecutar el Algoritmo Genético.")


# --- Ejecución de la Simulación Concurrente ---
st.header("4. Ejecutar Simulación Concurrente")

# Solo permitir ejecución si hay datos válidos cargados
if numero_a_atraso and distribucion_probabilidad and total_atraso_dataset is not None:
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
                    coincident_list = []
                    # Podemos usar los resultados de la primera ejecución para obtener probabilidad y frecuencia
                    # asumiendo que la probabilidad es una propiedad intrínseca de la combinación y los datos
                    first_run_results_map = dict(resultados_por_ejecucion[0]) if resultados_por_ejecucion else {}

                    for comb_tuple, ejecuciones_list in combinaciones_coincidentes.items():
                        # Obtener la probabilidad de la primera ejecución donde apareció
                        freq_prob = first_run_results_map.get(comb_tuple, (0, 0.0))
                        probabilidad_calculada = freq_prob[1]

                        # Calcular suma de atrasos para esta combinación
                        # Asegurarse de que los números de la combinación estén en numero_a_atraso antes de sumar
                        suma_atrasos_comb = sum(numero_a_atraso.get(val, 0) for val in comb_tuple)

                        # Calcular el valor especial para esta combinación
                        valor_especial_comb = total_atraso_dataset + 40 - suma_atrasos_comb

                        # Calcular atrasos (conteos) para esta combinación para mostrar
                        atrasos_comb_counts = Counter([numero_a_atraso.get(val) for val in comb_tuple if numero_a_atraso.get(val) is not None and numero_a_atraso.get(val) != -1])
                        atrasos_str = ", ".join([f"A{k}:{v}" for k, v in atrasos_comb_counts.items()])


                        coincident_list.append({
                            "Combinación": " - ".join(map(str, comb_tuple)),
                            "Probabilidad": probabilidad_calculada,
                            f"Apariciones (de {sim_n_ejecuciones} ejec.)": len(ejecuciones_list),
                            "Suma Atrasos Combinación": suma_atrasos_comb, # Agregar la suma de atrasos de la combinación
                            "Cálculo Especial": valor_especial_comb, # Agregar el cálculo especial
                            "Atrasos (Conteos)": atrasos_str # Mantener los conteos de atraso
                        })

                    # Ordenar por probabilidad descendente (o podrías ordenar por el cálculo especial si fuera relevante)
                    coincident_list_sorted = sorted(coincident_list, key=lambda x: x["Probabilidad"], reverse=True)

                    st.dataframe(coincident_list_sorted, height=400)

                else:
                    st.info("No se encontraron combinaciones que aparecieran en *todas* las simulaciones con las restricciones dadas.")

            except Exception as e:
                 st.error(f"Ocurrió un error durante la ejecución de la Simulación Concurrente: {e}")

else:
     st.info("Carga un archivo CSV válido para ejecutar la Simulación Concurrente.")


# --- Notas/Explicación Adicional ---
st.sidebar.header("Información")
st.sidebar.markdown("""
**Propósito:**
Ayuda a encontrar combinaciones de números (ej. para loterías) basándose en datos históricos de 'atraso' y restricciones definidas por el usuario.

**Datos:**
Requiere un archivo CSV con al menos las columnas 'Numero' y 'Atraso'. Los números en la columna 'Numero' deben ser únicos.

**Suma Total de Atrasos:**
Al cargar el archivo, se calcula y muestra la suma de todos los valores de la columna 'Atraso' en tus datos.

**Cálculo Especial:**
Para cada combinación encontrada, se calcula:
`(Suma Total de Atrasos en Dataset) + 40 - (Suma de Atrasos de esta Combinación)`

**Configuración de Restricciones:**
*   La sección de Restricciones de Atraso ahora **carga automáticamente** todos los valores de 'Atraso' encontrados en tu archivo CSV.
*   Por defecto, todos estos atrasos están **pre-seleccionados** para restringir.
*   El valor inicial mostrado para cada restricción es la **cantidad de veces que ese atraso específico aparece en tu dataset**. Puedes ajustar estos valores si lo deseas.

**Métodos:**
1.  **Algoritmo Genético:** Busca una única combinación "óptima" que maximice su probabilidad (basada en la distribución cargada) mientras cumple las restricciones de atraso.
2.  **Simulación Concurrente:** Genera muchas combinaciones aleatorias (respetando restricciones y probabilidad) en paralelo múltiples veces. Identifica combinaciones que son lo suficientemente robustas como para aparecer en *todas* las ejecuciones.
""")
