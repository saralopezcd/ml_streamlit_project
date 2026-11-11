import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentación con K-Means", page_icon="📊")


@st.cache_resource
def load_artifact():
    return joblib.load("model.joblib")


artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
stats = artifact.get("feature_stats", {})
descriptions = artifact.get("cluster_descriptions", {})
names = artifact.get("cluster_names", {})


def cluster_label(c: int) -> str:
    return names.get(c, f"Cluster {c}")

profiles = artifact.get("profiles", {})

st.title("Segmentación de casos con K-Means")
st.write(
    "Ingrese las características de un nuevo caso para saber a qué "
    "cluster (grupo) pertenece."
)

values = {}
for feature in features:
    s = stats.get(feature, {})
    help_text = None
    if s:
        help_text = f"Rango observado en el entrenamiento: {s['min']:.2f} a {s['max']:.2f}"
    values[feature] = st.number_input(
        feature,
        value=float(s.get("median", 0.0)),
        help=help_text,
    )

if st.button("Asignar cluster"):
    X_new = pd.DataFrame([values], columns=features)
    cluster = int(model.predict(X_new)[0])

    st.success(f"Cluster asignado: {cluster_label(cluster)} (cluster {cluster})")
    st.info(descriptions.get(cluster, "Sin descripción"))

    outside = [
        f for f in features
        if f in stats and not (stats[f]["min"] <= values[f] <= stats[f]["max"])
    ]
    if outside:
        st.warning(
            "Estos valores están fuera del rango visto en el entrenamiento: "
            + ", ".join(outside)
        )

with st.expander("Ver perfil promedio de cada cluster"):
    if profiles:
        prof = pd.DataFrame(profiles).T
        prof.index = [f"{c} - {cluster_label(c)}" for c in prof.index]
        st.dataframe(prof)
    for c, text in descriptions.items():
        st.write(f"**{cluster_label(c)} (cluster {c}):** {text}")