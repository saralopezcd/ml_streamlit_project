from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/datos.csv")
MODEL_PATH = Path("model.joblib")
PROFILES_PATH = Path("profiles.csv")

NUMERIC_FEATURES: list[str] = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]
EXCLUDE: list[str] = []
K_RANGE = range(2, 7)
FORCE_K: int | None = 5
RANDOM_STATE = 42

# Nombres de los clusters (los mismos del README). Se asignan por número de
# cluster para k=5 y random_state=42. Si cambias k, los datos o las variables,
# revisa estos nombres con la tabla de perfiles.
CLUSTER_NAMES: dict[int, str] = {
    0: "Base / bajo nivel",
    1: "Tanques especiales",
    2: "Veloces ofensivos",
    3: "Élite",
    4: "Atacantes físicos pesados",
}
# Total medio de estadísticas esperado por cluster (para verificar que los
# nombres siguen correspondiendo a los perfiles tras reentrenar).
EXPECTED_TOTALS: dict[int, float] = {0: 301, 1: 480, 2: 473, 3: 631, 4: 514}


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=None, engine="python")


def looks_like_id(name: str, series: pd.Series) -> bool:
    n = name.lower()
    by_name = n == "id" or n.endswith("_id") or n.startswith("id_")
    all_unique = series.nunique(dropna=True) == len(series)
    return by_name or (all_unique and pd.api.types.is_integer_dtype(series))


def select_features(df: pd.DataFrame) -> list[str]:
    if NUMERIC_FEATURES:
        missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas no encontradas en el dataset: {missing}")
        return list(NUMERIC_FEATURES)
    numeric = df.select_dtypes(include="number").columns
    features = [
        c for c in numeric
        if c not in EXCLUDE
        and df[c].nunique(dropna=True) > 1
        and not looks_like_id(c, df[c])
    ]
    if len(features) < 2:
        raise ValueError(
            "Se necesitan al menos 2 columnas numéricas. "
            "Define NUMERIC_FEATURES manualmente en train.py."
        )
    return features


def build_pipeline(k: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)),
    ])


def choose_k(X: pd.DataFrame) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in K_RANGE:
        if k >= len(X):
            continue
        pipe = build_pipeline(k)
        labels = pipe.fit_predict(X)
        processed = pipe[:-1].transform(X)
        scores[k] = silhouette_score(processed, labels)
        print(f"  k={k}: silhouette={scores[k]:.3f}")
    best = max(scores, key=scores.get)
    return best, scores


def describe_clusters(X: pd.DataFrame, labels, features: list[str]):
    """Perfiles (promedios por cluster) y una descripción automática basada
    en qué tan por encima o por debajo del promedio general está cada variable.
    Revisa y mejora las descripciones con tu propio análisis."""
    data = X.copy()
    data["cluster"] = labels
    profiles = data.groupby("cluster")[features].mean()
    g_mean, g_std = X[features].mean(), X[features].std().replace(0, 1)
    descriptions: dict[int, str] = {}
    for c, row in profiles.iterrows():
        n = int((labels == c).sum())
        pct = 100 * n / len(X)
        z = (row - g_mean) / g_std
        high = [f for f in features if z[f] > 0.5]
        low = [f for f in features if z[f] < -0.5]
        parts = [f"{n} casos ({pct:.0f}% del total)."]
        if high:
            parts.append("Valores altos en: " + ", ".join(high) + ".")
        if low:
            parts.append("Valores bajos en: " + ", ".join(low) + ".")
        if not high and not low:
            parts.append("Perfil cercano al promedio general.")
        descriptions[int(c)] = " ".join(parts)
    return profiles, descriptions


def assign_names(profiles: pd.DataFrame, k: int) -> dict[int, str]:
    """Usa CLUSTER_NAMES solo si los perfiles reentrenados coinciden con los
    esperados; si no, usa 'Cluster N' para no mostrar un nombre equivocado."""
    totals = profiles.sum(axis=1)
    ok = k == len(CLUSTER_NAMES) and all(
        abs(totals[c] - EXPECTED_TOTALS[c]) <= 15 for c in CLUSTER_NAMES
    )
    if not ok:
        print("AVISO: los perfiles no coinciden con los esperados; "
              "revisa CLUSTER_NAMES en train.py.")
        return {int(c): f"Cluster {c}" for c in profiles.index}
    return dict(CLUSTER_NAMES)


def train_model(df: pd.DataFrame) -> dict:
    features = select_features(df)
    print("Variables usadas:", features)
    X = df[features].copy()

    if FORCE_K:
        k = FORCE_K
        pipe_tmp = build_pipeline(k)
        labels_tmp = pipe_tmp.fit_predict(X)
        scores = {k: silhouette_score(pipe_tmp[:-1].transform(X), labels_tmp)}
    else:
        print("Buscando el mejor número de clusters:")
        k, scores = choose_k(X)
    print(f"Se usará k={k} (silhouette={scores[k]:.3f})")

    model = build_pipeline(k)
    model.fit(X)
    labels = model.predict(X)

    profiles, descriptions = describe_clusters(X, labels, features)
    names = assign_names(profiles, k)
    print("\nPerfiles promedio por cluster:")
    print(profiles.round(2))
    profiles.round(2).to_csv(PROFILES_PATH)

    feature_stats = {
        f: {
            "min": float(X[f].min()),
            "max": float(X[f].max()),
            "median": float(X[f].median()),
        }
        for f in features
    }
    return {
        "model": model,
        "features": features,
        "numeric_features": features,
        "categorical_features": [],
        "task": "clustering",
        "cluster_descriptions": descriptions,
        "cluster_names": names,
        "profiles": profiles.round(2).to_dict(orient="index"),
        "feature_stats": feature_stats,
        "k": k,
        "silhouette": float(scores[k]),
    }


def main() -> None:
    df = load_data()
    print("Dataset:", df.shape)
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")
    print("\nVersiones usadas (cópialas a requirements.txt para que Docker")
    print("use las mismas):")
    print(f"scikit-learn=={sklearn.__version__}")
    print(f"pandas=={pd.__version__}")
    print(f"joblib=={joblib.__version__}")


if __name__ == "__main__":
    main()