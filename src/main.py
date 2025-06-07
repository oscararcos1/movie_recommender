import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path_ratings: str):
    """
    Carga u.data en un DataFrame pivotado (usuarios × películas)
    y devuelve también su versión CSR (sparse matrix).
    """
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv(path_ratings, sep='\t', names=cols)
    mat = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return mat, csr_matrix(mat.values)

def compute_similarity(sparse_mat: csr_matrix):
    """
    Dada la matriz CSR de ratings, devuelve la matriz
    de similitud usuario–usuario usando cosine similarity.
    """
    return cosine_similarity(sparse_mat)

def recommend(user_idx: int,
              user_sim: np.ndarray,
              sparse_mat: csr_matrix,
              top_n: int = 10) -> list[int]:
    # 1) Vector de similitud user–user (longitud n_users)
    sim_scores = user_sim[user_idx]  

    # 2) Convertir ratings a denso para asegurar dimensiones (943×1682)
    dense_mat = sparse_mat.toarray()

    # 3) Weighted sum: sim_scores (943,) · dense_mat (943×1682) → (1682,)
    scores = np.dot(sim_scores, dense_mat)  

    # 4) Normalización
    denom = np.sum(np.abs(sim_scores))
    if denom != 0:
        scores = scores / denom

    # 5) Crear máscara de películas ya vistas
    user_ratings = dense_mat[user_idx]  
    seen_mask = user_ratings > 0
    scores[seen_mask] = -1

    # 6) Tomar top_n índices
    top_indices = np.argsort(scores)[::-1][:top_n]  
    # 7) Convertir a movie_id 1-based
    return (top_indices + 1).tolist()



def main():
    # Ruta al archivo de ratings
    DATA_PATH = 'data/u.data'

    # 1) Cargar datos
    mat, sparse_mat = load_data(DATA_PATH)
    print(f"Matriz usuarios×películas: {mat.shape}")
    print(f"Ratings no nulos: {sparse_mat.nnz}")

    # 2) Calcular similitud
    user_sim = compute_similarity(sparse_mat)
    print(f"Matriz de similitud: {user_sim.shape}")

    # 3) Probar recomendaciones para usuario 1 (índice 0)
    top5 = recommend(user_idx=0, user_sim=user_sim, sparse_mat=sparse_mat, top_n=5)
    print(f"Top-5 recomendaciones para usuario 1: {top5}")

    # 4) Probar otro usuario
    top5_u10 = recommend(user_idx=9, user_sim=user_sim, sparse_mat=sparse_mat, top_n=5)
    print(f"Top-5 para usuario 10: {top5_u10}")

if __name__ == "__main__":
    main()
