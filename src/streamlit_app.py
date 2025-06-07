import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from evaluate import avg_precision_at_k

# 1) Configuración de la página (debe ir primero)
st.set_page_config(page_title="Recomendador de Películas", layout="wide")
st.title("🎬 Recomendador de Películas")

# 2) Funciones de carga y cálculo

@st.cache_data
def load_data(path_ratings: str):
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv(path_ratings, sep='\t', names=cols)
    mat = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return mat, csr_matrix(mat.values)

@st.cache_data
def load_titles(path_items: str) -> dict[int,str]:
    """
    Carga únicamente las dos primeras columnas de u.item:
    - Columna 0: movie_id
    - Columna 1: title
    Devuelve un dict {movie_id: title}.
    """
    df = pd.read_csv(
        path_items,
        sep='|',
        header=None,            # no hay fila de cabecera
        usecols=[0, 1],         # sólo tomo las dos primeras columnas
        names=['movie_id', 'title'],
        encoding='latin-1'
    )
    return df.set_index('movie_id')['title'].to_dict()




@st.cache_data
def compute_similarity(_sparse_mat: csr_matrix) -> np.ndarray:
    return cosine_similarity(_sparse_mat)

def recommend(user_idx: int,
              user_sim: np.ndarray,
              sparse_mat: csr_matrix,
              top_n: int = 10) -> list[int]:
    # 1) vector de similitud (n_users,)
    sim_scores = user_sim[user_idx]
    # 2) convertir a denso para producto (n_users × n_items)
    dense_mat = sparse_mat.toarray()
    # 3) weighted sum → (n_items,)
    scores = np.dot(sim_scores, dense_mat)
    # 4) normalización
    denom = np.sum(np.abs(sim_scores))
    if denom != 0:
        scores = scores / denom
    # 5) filtrar ítems ya vistos
    seen_mask = dense_mat[user_idx] > 0
    scores[seen_mask] = -1
    # 6) extraer top_n índices
    top_idxs = np.argsort(scores)[::-1][:top_n]
    # 7) convertir a movie_id 1-based
    return (top_idxs + 1).tolist()

# 3) Carga de datos y cálculo previo
# Ajusta la ruta según dónde tengas u.data
DATA_PATH = 'data/u.data'  # o 'data/u.data'
mat, sparse_mat = load_data(DATA_PATH)
user_sim = compute_similarity(sparse_mat)

# 3.1) Cargar mapeo de movie_id → title
TITLES = load_titles('data/u.item') 

# 4) Sidebar de parámetros
st.sidebar.header("Parámetros")
usuario_id = st.sidebar.number_input(
    "Usuario (ID):",
    min_value=int(mat.index.min()),
    max_value=int(mat.index.max()),
    value=int(mat.index.min())
) - 1  # convertir a índice 0-based

metodo = st.sidebar.selectbox(
    "Reducción dimensional:",
    ("PCA", "t-SNE", "UMAP")
)

k = st.sidebar.slider(
    "Número de recomendaciones:",
    min_value=5,
    max_value=20,
    value=10
)

# 4.1) Checkbox para activar evaluación Precision@K
show_eval = st.sidebar.checkbox("Mostrar Precision@K (test 10%)", value=False)


# 5) Botón y bloque principal
if st.sidebar.button("▶ Generar"):
    # 5.1) Generar recomendaciones
    recs = recommend(usuario_id, user_sim, sparse_mat, top_n=k)
       # 5.1) Generar recomendaciones (IDs)
    recs = recommend(usuario_id, user_sim, sparse_mat, top_n=k)

    # 5.1.1) Convertir cada movie_id en su título
    rec_titles = [TITLES.get(mid, f"Movie {mid}") for mid in recs]

    # 5.1.2) Mostrar la tabla de títulos
    st.subheader(f"Top {k} recomendaciones para usuario {usuario_id+1}")
    st.table({"Título": rec_titles})

    # ——— Evaluación Precision@K ———
    if show_eval:
        # 1) Leer todo el dataset original
        df_all = pd.read_csv(
            DATA_PATH,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        # 2) Calcular Precision@K (promedio)
        p_at_k = avg_precision_at_k(df_all, recommend, k=k, test_size=0.1)
        # 3) Mostrar el resultado en la sidebar
        st.sidebar.write(f"Precision@{k} promedio: **{p_at_k:.3f}**")


    # 5.2) Obtener coordenadas según técnica
    if metodo == "PCA":
        coords = PCA(n_components=2).fit_transform(user_sim)
    elif metodo == "t-SNE":
        coords = TSNE(n_components=2, random_state=42).fit_transform(user_sim)
    else:  # UMAP
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(user_sim)

    # 5.3) Preparar DataFrame para gráfica
    df_viz = pd.DataFrame(coords, columns=["x", "y"])
    df_viz["usuario"] = df_viz.index + 1

    # 5.4) Mostrar gráfico interactivo
    st.subheader(f"Mapa de usuarios ({metodo})")
    fig = px.scatter(
        df_viz,
        x="x",
        y="y",
        hover_data=["usuario"],
        title="Proyección de usuarios"
    )
    st.plotly_chart(fig, use_container_width=True)
