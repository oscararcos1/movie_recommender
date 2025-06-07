import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def build_matrix(df_ratings: pd.DataFrame):
    """De DataFrame user–movie–rating → mat densa y sparse."""
    mat = df_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return mat, csr_matrix(mat.values)

def precision_at_k(recs: list[int], actual: set[int], k: int) -> float:
    """Precision@K para un solo usuario."""
    hit_count = len(set(recs[:k]) & actual)
    return hit_count / k

def avg_precision_at_k(df_all: pd.DataFrame, recommend_fn, k: int = 10, test_size=0.1):
    """
    - df_all: DataFrame con columnas user_id, movie_id, rating
    - recommend_fn: función recommend(user_idx, user_sim, sparse_mat, top_n)
    - Devuelve Precision@K promedio sobre todos los usuarios.
    """
    # 1) Split train/test
    train_df, test_df = train_test_split(df_all, test_size=test_size, random_state=42)

    # 2) Construir matrices de train
    mat_train, sm_train = build_matrix(train_df)
    user_sim = cosine_similarity(sm_train)

    precisions = []
    users = sorted(train_df['user_id'].unique())
    for u in users:
        # 3) Recs con train
        recs = recommend_fn(u-1, user_sim, sm_train, top_n=k)
        # 4) Actuales en test para ese usuario
        actual = set(test_df[test_df.user_id == u]['movie_id'])
        if actual:
            precisions.append(precision_at_k(recs, actual, k))
    return float(np.mean(precisions)) if precisions else 0.0
