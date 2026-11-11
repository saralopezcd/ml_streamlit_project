# Segmentación de Pokémon por estadísticas de combate (K-Means)

**Autora:** Sara Lopez Chavarro
**Ruta:** No supervisada (clustering con K-Means)

## 1. Objetivo
Agrupar Pokémon en perfiles de combate según sus estadísticas base, sin usar
etiquetas previas, y ofrecer una app que asigne un cluster a un nuevo caso.

## 2. Dataset
- **Fuente:** Kaggle, "Complete Pokemon Stats & Attributes Archive", construido
  a partir de la API abierta [PokeAPI](https://pokeapi.co).
  Enlace: https://www.kaggle.com/datasets/shreyashautomation/pokemon-stats-dataset/data
- **Unidad de observación:** un Pokémon (incluye formas alternativas).
- **Tamaño:** 1.351 filas x 20 columnas.
- **Ubicación en el repositorio:** `data/datos.csv`.
- **Limitación:** algunas especies aparecen en varias formas (regionales,
  mega, etc.), por lo que no todas las filas son independientes.

## 3. Variables utilizadas
| Variable | Descripción |
|---|---|
| hp | Puntos de vida base |
| attack | Ataque físico base |
| defense | Defensa física base |
| special_attack | Ataque especial base |
| special_defense | Defensa especial base |
| speed | Velocidad base |

**Excluidas:** `id` (identificador), `name`, `type_1`, `type_2`, `abilities`,
`generation`, `color`, `shape` (categóricas); `is_legendary`, `is_mythical`
(etiquetas, se usan solo para validar los clusters); `base_experience` y
`capture_rate` (muy relacionadas con la fuerza, repetirían información);
`height_dm` y `weight_hg` (medidas físicas con valores extremos, no de combate).

## 4. Preparación de datos
Pipeline de scikit-learn: imputación de faltantes con la mediana ->
estandarización (`StandardScaler`) -> `KMeans`. Todo queda guardado en
`model.joblib`, de modo que la app aplica las mismas transformaciones que el
entrenamiento.

## 5. Modelo y justificación
Usé K-Means porque permite asignar nuevos casos con `predict()`. En el
análisis exploratorio (notebook de Colab) comparé valores de k entre 2 y 8
con el coeficiente de silhouette:

| k | Silhouette |
|---|---|
| 2 | 0.282 |
| 3 | 0.248 |
| 4 | 0.224 |
| 5 | 0.222 |
| 6 | 0.230 |
| 7 | 0.226 |
| 8 | 0.190 |

El silhouette más alto corresponde a k=2, pero esa solución solo separa
Pokémon "débiles" de "fuertes" y aporta poca información. Elegí **k=5**
por interpretabilidad: produce perfiles de combate distintos y explicables,
con un silhouette similar al de k=4 y k=6. En `train.py` el valor queda fijo
(`FORCE_K = 5`) para que el entrenamiento sea reproducible.

## 6. Evaluación
- **Silhouette (k=5): 0.222.** Es bajo en todos los valores de k probados.
  Esto es esperable: las estadísticas de combate forman un continuo y no
  grupos naturalmente separados, por lo que los clusters se solapan en sus
  fronteras. Por eso la interpretación de los perfiles pesa tanto como la
  métrica.
- **Perfiles promedio por cluster:**

| Cluster | Nombre | Casos | hp | attack | defense | sp. atk | sp. def | speed | Total |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Base / bajo nivel | 406 (30%) | 49.0 | 54.4 | 50.6 | 47.3 | 47.8 | 51.6 | 300.7 |
| 1 | Tanques especiales | 203 (15%) | 70.1 | 69.7 | 95.2 | 87.9 | 102.1 | 54.8 | 479.6 |
| 2 | Veloces ofensivos | 290 (21%) | 68.4 | 87.9 | 66.5 | 80.6 | 70.6 | 99.4 | 473.4 |
| 3 | Élite | 186 (14%) | 91.5 | 109.5 | 95.1 | 126.3 | 103.5 | 105.1 | 631.0 |
| 4 | Atacantes físicos pesados | 266 (20%) | 97.2 | 111.7 | 98.2 | 67.6 | 76.2 | 63.3 | 514.2 |

- **Validación externa (no usada para entrenar):** el 44% de los Pokémon del
  cluster 3 son legendarios (81 de 186), y ese cluster concentra cerca del 68%
  de todos los legendarios del dataset (81 de 120). En los demás clusters la
  proporción de legendarios es del 7% o menos. No es una separación perfecta:
  más de la mitad del cluster 3 no son legendarios.

## 7. Interpretación de los clusters
- **Cluster 0 – Base / bajo nivel (30%):** todas las estadísticas bajas
  (total ~301). Casi no tiene legendarios.
- **Cluster 1 – Tanques especiales (15%):** defensa (95) y defensa especial
  (102) altas, velocidad baja (55). Aguantan golpes pero actúan lento.
- **Cluster 2 – Veloces ofensivos (21%):** velocidad muy alta (~99) con buen
  ataque (88) y ataque especial (81), pero defensa media-baja.
- **Cluster 3 – Élite (14%):** todas las estadísticas altas, destacando
  ataque especial (126). Total medio de 631 y mayor concentración de
  legendarios.
- **Cluster 4 – Atacantes físicos pesados (20%):** HP (97), ataque (112) y
  defensa (98) altos, ataque especial bajo (68) y velocidad media-baja (63).

**Limitaciones:** el dataset incluye formas alternativas de una misma especie,
por lo que las filas no son totalmente independientes; el silhouette es
moderado; y el análisis solo usa estadísticas base, sin tipos ni habilidades.

## 8. Estructura del repositorio
| Archivo | Función |
|---|---|
| `train.py` | Entrena el modelo y genera `model.joblib` y `profiles.csv` |
| `app.py` | Aplicación Streamlit (carga `model.joblib`, no reentrena) |
| `model.joblib` | Pipeline entrenado + metadatos y nombres de clusters |
| `profiles.csv` | Perfil promedio de cada cluster |
| `data/datos.csv` | Dataset |
| `requirements.txt` | Dependencias |
| `Dockerfile`, `.dockerignore` | Empaquetado en contenedor |

## 9. Cómo ejecutar

### Obtener el proyecto
```bash
git clone https://github.com/saralopezcd/ml_streamlit_project.git
cd ml_streamlit_project
```

### Entrenar
```bash
pip install -r requirements.txt
python train.py
```
Genera `model.joblib` y `profiles.csv`. Si los perfiles reentrenados no
coinciden con los nombres definidos en `train.py`, el script imprime un aviso
y la app mostrará "Cluster N" en lugar del nombre.

### Streamlit (sin Docker)
```bash
streamlit run app.py
```
Abrir http://localhost:8501

### Docker
```bash
docker build -t ml-streamlit-app .
docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
```
Abrir http://localhost:8501. Nota: la imagen reentrena el modelo al
construirse (`RUN python train.py`), por eso necesita `data/datos.csv`.

Ver logs: `docker logs ml-streamlit`.
Detener y borrar: `docker stop ml-streamlit && docker rm ml-streamlit`.

## 10. Ejemplo de entrada y salida
**Entrada:** hp=70, attack=80, defense=70, special_attack=69, special_defense=70, speed=70

**Salida:** `Cluster asignado: Veloces ofensivos (cluster 2)`
y la descripción: *"290 casos (21% del total). Valores altos en: speed."*

La app también avisa si algún valor ingresado está fuera del rango visto en el
entrenamiento, y permite ver el perfil promedio de cada cluster.

## 11. Despliegue
**Por qué no usé Google Cloud.** No me fue posible desplegar en una VM de Google
Cloud Compute Engine. Mi cuenta ya había gastado el crédito de la prueba
gratuita de Google Cloud en otro momento, mientras estudiaba la plataforma, y
no pude volver a activarla. Sin ese crédito, la plataforma exigía un pago previo
para habilitar el servicio, así que decidí no usarlo para evitar costos.

**Alternativa elegida.** Empaqueté el proyecto con Docker y lo ejecuté en un
computador local, siguiendo los pasos de la sección 9. Esto conserva lo esencial
del reto (construir la imagen, ejecutar el contenedor y exponer la aplicación en
el puerto 8501), y cambia únicamente el lugar donde corre.

Referencia (no ejecutado): en una VM Ubuntu, los pasos serían los mismos más
instalar Docker (`sudo apt install -y docker.io git`), `git clone`, y abrir el
puerto 8501 (`gcloud compute firewall-rules create allow-streamlit-8501
--allow=tcp:8501 --direction=INGRESS --target-tags=streamlit-server`).

**Prueba de despliegue local.** Construí la imagen con `docker build` y ejecuté
el contenedor con `docker run`, siguiendo los pasos de la sección 9. La
aplicación abrió en http://localhost:8501 y asignó clusters correctamente.

## 12. Prueba de reproducibilidad
**Tipo de prueba.** No hice una prueba cruzada con otra pareja, porque trabajé
de forma individual. En su lugar, la hice por mi cuenta en un segundo
computador, con otra cuenta de usuario, partiendo solo del repositorio y
siguiendo las instrucciones de este README.

**Resultado.** En ese segundo entorno construí la imagen Docker, ejecuté el
contenedor y la aplicación funcionó correctamente, sin depender de archivos ni
configuraciones de mi computador de desarrollo.

**Limitación.** Como yo misma escribí el README y yo misma hice la prueba, esta
verificación es menos independiente que una prueba cruzada: puede pasar por alto
pasos que yo doy por sabidos.

## Créditos
Datos: PokeAPI y el autor del dataset en Kaggle. Pokémon es marca de sus
respectivos dueños; este proyecto es solo con fines educativos.
