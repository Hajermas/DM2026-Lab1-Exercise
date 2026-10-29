from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@tool
def reduce_dimensions_tool(
    method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
) -> Dict[str, Any]:
    """Reduces feature matrix dimensionality using PCA, t-SNE, or UMAP, and creates a 2D plot.

    Args:
        method: Reduction method to use ('pca', 'tsne', or 'umap').
        n_components: Number of target dimensions (default: 2).
        random_state: Random state seed for reproducibility.
        perplexity: Perplexity parameter for t-SNE (default: 30.0).
        n_neighbors: Number of neighbors for UMAP (default: 15).

    Returns:
        Dict containing result_id, method, reduced coordinates, and explained variance (if PCA).
    """
    if session.feature_matrix is None:
        return {"error": "No feature matrix available in session. Run build_dtm_tool first."}

    X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

    explained_variance_ratio = None

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        coords = reducer.fit_transform(X)
        explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
        )
        coords = reducer.fit_transform(X)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            return {"error": "umap-learn package is not installed."}
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        coords = reducer.fit_transform(X)
    else:
        return {"error": f"Unsupported reduction method: {method}"}

    coords_list = coords.tolist()
    session.artifacts[f"reduced_coords_{method}"] = coords_list

    # Generate 2D Scatter Plot requirement
    fig, ax = plt.subplots(figsize=(8, 6))
    if session.dataframe is not None and "category_name" in session.dataframe.columns:
        categories = session.dataframe["category_name"].tolist()
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=categories,
            style=categories,
            s=100,
            ax=ax,
        )
    else:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], s=100, ax=ax)

    ax.set_title(f"2D Dimension Reduction ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()

    session.pending_figure = fig
    plt.close(fig)

    res_id = session.next_result_id("reduction")
    summary = {
        "result_id": res_id,
        "method": method,
        "n_components": n_components,
        "n_samples": len(coords_list),
        "coordinates": coords_list,
        "explained_variance_ratio": explained_variance_ratio,
    }

    session.store_result("reduce_dimensions_tool", {"method": method}, summary)
    return summary


@tool
def binarize_labels_tool() -> Dict[str, Any]:
    """Binarizes target category labels into integer one-hot dummy columns.

    Returns:
        Dict containing result_id, categories, shape, and head of binarized labels.
    """
    if session.dataframe is None or "category_name" not in session.dataframe.columns:
        return {"error": "session.dataframe missing or lacks 'category_name' column."}

    binarized_df = pd.get_dummies(
        session.dataframe["category_name"], prefix="category", dtype=int
    )

    session.artifacts["binarized_labels"] = binarized_df

    res_id = session.next_result_id("binarize")
    summary = {
        "result_id": res_id,
        "columns": binarized_df.columns.tolist(),
        "shape": list(binarized_df.shape),
        "head": binarized_df.head().to_dict(orient="records"),
    }

    session.store_result("binarize_labels_tool", {}, summary)
    return summary


def make_tools(session_obj):
    global session
    session = session_obj
    return [reduce_dimensions_tool, binarize_labels_tool]
