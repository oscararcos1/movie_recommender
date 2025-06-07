# Movie Recommender with Advanced Visualizations

## 1. Problemática y Justificación

**Problema:** Los usuarios de plataformas de streaming disponen de centenares de títulos y necesitan ayuda para descubrir contenido relevante.  
**Solución propuesta:** Un sistema de recomendación colaborativa que, además, muestre visualmente cómo se agrupan los gustos de usuarios mediante técnicas de reducción de dimensionalidad.  
**Impacto potencial:**  
- Mejora del engagement al facilitar descubrimiento.  
- Interpretabilidad: clave para la toma de decisiones y feedback.

## 2. Estructura del Proyecto

movie_recommender/
├── .venv/
├── data/
├── src/
├── requirements.txt
└── README.md

## 3. Tecnologías Utilizadas

- **Python 3.11** + **venv**  
- **Pandas**, **NumPy**, **SciPy**  
- **scikit-learn** (cosine_similarity, PCA, t-SNE)  
- **UMAP-learn**  
- **Plotly** (Gráficos interactivos)  
- **Streamlit** (Interfaz Web)

## 4. Instrucciones de Ejecución

# Clona este repositorio y entra en la carpeta:
   ```bash
   git clone <https://github.com/oscararcos1/movie_recommender>
   cd movie_recommender

Activa el entorno virtual:
.\.venv\Scripts\activate

Instala dependencias:
pip install -r requirements.txt

Descarga MovieLens 100K y descomprímelo en data/

Lanza la app:
streamlit run src/streamlit_app.py

En la sidebar:

Selecciona Usuario (ID), Técnica (PCA/t-SNE/UMAP), K.

(Opcional) Marca “Mostrar Precision@K” para ver la métrica.

Haz clic en ▶ Generar.

## 5. Resultados y Métricas
Precision@10 promedio (test 10%): 0.153

Patrones visuales: usuarios con gustos afines forman clusters claros en t-SNE y UMAP, apoyando la calidad de las recomendaciones.

# 6. Posibles Mejoras
Incorporar datos de contenido (géneros, sinopsis) para un sistema híbrido.

Probar autoencoders o matrix factorization (p. ej. SVD).

Ajustar parámetros de t-SNE/UMAP (perplexity, n_neighbors).

Desplegar en Streamlit Cloud o Heroku.