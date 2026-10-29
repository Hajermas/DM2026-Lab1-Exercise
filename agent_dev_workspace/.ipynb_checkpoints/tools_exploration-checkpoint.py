from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


@tool
def cosine_similarity_tool(
    doc1_index: int,
    doc2_index: int,
) -> Dict[str, Any]:
    """Computes cosine similarity between two documents by their row indices in session.feature_matrix.

    Args:
        doc1_index: Row index of the first document (0-based).
        doc2_index: Row index of the second document (0-based).

    Returns:
        Dict containing result_id, doc1_index, doc2_index, and the cosine similarity score.
    """
    if session.feature_matrix is None:
        return {"error": "No feature matrix available in session. Run build_dtm_tool first."}

    n_docs = session.feature_matrix.shape[0]
    if not (0 <= doc1_index < n_docs):
        return {"error": f"doc1_index {doc1_index} out of bounds for matrix with {n_docs} documents."}
    if not (0 <= doc2_index < n_docs):
        return {"error": f"doc2_index {doc2_index} out of bounds for matrix with {n_docs} documents."}

    vec1 = session.feature_matrix[doc1_index : doc1_index + 1]
    vec2 = session.feature_matrix[doc2_index : doc2_index + 1]

    # Handle zero vectors explicitly (empty document yields 0.0 similarity)
    norm1 = np.linalg.norm(vec1.toarray() if hasattr(vec1, "toarray") else vec1)
    norm2 = np.linalg.norm(vec2.toarray() if hasattr(vec2, "toarray") else vec2)

    if norm1 == 0 or norm2 == 0:
        sim_val = 0.0
    else:
        sim_val = float(cosine_similarity(vec1, vec2)[0, 0])

    res_id = session.next_result_id("cosine")
    summary = {
        "result_id": res_id,
        "doc1_index": doc1_index,
        "doc2_index": doc2_index,
        "similarity": sim_val,
    }

    session.store_result(
        "cosine_similarity_tool",
        {"doc1_index": doc1_index, "doc2_index": doc2_index},
        summary,
    )
    return summary


@tool
def feature_correlation_matrix_tool(
    top_n: int = 20,
) -> Dict[str, Any]:
    """Computes feature-vs-feature Pearson correlation matrix over top variance terms and sets a heatmap figure.

    Args:
        top_n: Number of top terms by variance to include (default: 20).

    Returns:
        Dict containing result_id, terms list, and correlation matrix as a nested list.
    """
    if session.feature_matrix is None or session.feature_names is None:
        return {"error": "No feature matrix available in session. Run build_dtm_tool first."}

    X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
    terms_all = np.array(session.feature_names)

    if X.shape[1] == 0:
        return {"error": "Feature matrix has 0 terms."}

    # Compute variance for each term across all documents
    variances = np.var(X, axis=0)
    top_k = min(top_n, X.shape[1])
    
    # Sort terms descending by variance
    top_indices = np.argsort(variances)[::-1][:top_k]
    selected_terms = terms_all[top_indices].tolist()
    selected_X = X[:, top_indices]

    # Compute Pearson correlation matrix across term columns
    # np.corrcoef expects variables as rows, so transpose selected_X
    corr_matrix = np.corrcoef(selected_X, rowvar=False)
    
    # Handle NaN in correlation matrix if variance is 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Plot annotated heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        xticklabels=selected_terms,
        yticklabels=selected_terms,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
    )
    ax.set_title(f"Feature Correlation Matrix (Top {top_k} Terms by Variance)")
    plt.tight_layout()

    session.pending_figure = fig

    res_id = session.next_result_id("feat_corr")
    summary = {
        "result_id": res_id,
        "n_terms": top_k,
        "terms": selected_terms,
        "correlation_matrix": corr_matrix.tolist(),
    }

    session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
    return summary


def make_tools(session_obj):
    global session
    session = session_obj
    return [cosine_similarity_tool, feature_correlation_matrix_tool]
