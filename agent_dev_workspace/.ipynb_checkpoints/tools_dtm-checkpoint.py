from typing import List, Optional, Union, Dict, Any
from langchain_core.tools import tool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_df: float = 1.0,
        min_df: Union[int, float] = 1,
        max_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """Build a document-term matrix (DTM) from session.dataframe using CountVectorizer.

        Args:
            ngram_range: Lower and upper boundary of the range of n-values for different n-grams (default [1, 1]).
            max_df: Ignore terms that have a document frequency strictly higher than the given threshold (default 1.0).
            min_df: Ignore terms that have a document frequency strictly lower than the given threshold (default 1).
            max_features: If not None, build a vocabulary that only considers the top max_features ordered by term frequency across the corpus.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        # Convert List[int] to Tuple[int, int] for CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features
        )

        texts = df["text"].fillna("").astype(str).tolist()
        X_sparse = vectorizer.fit_transform(texts)

        session.feature_matrix = X_sparse
        session.feature_names = list(vectorizer.get_feature_names_out())
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in df.columns:
            session.set_labels(df["category_name"].values)

        n_docs, n_terms = X_sparse.shape
        non_zero = int(X_sparse.nnz)
        total_elements = int(n_docs * n_terms)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {
            "ngram_range": ngram_range,
            "max_df": max_df,
            "min_df": min_df,
            "max_features": max_features
        }, summary)

        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregate total term frequencies across all documents in session.feature_matrix."""
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        X_sparse = session.feature_matrix
        feature_names = session.feature_names

        counts = np.asarray(X_sparse.sum(axis=0)).ravel().tolist()
        term_frequencies = {name: int(count) for name, count in zip(feature_names, counts)}

        sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_terms = dict(sorted_terms[:10])

        full_report = pd.DataFrame(sorted_terms, columns=["term", "frequency"])

        result_id = session.next_result_id("tf")
        summary = {
            "result_id": result_id,
            "n_terms": len(feature_names),
            "total_word_count": int(sum(counts)),
            "top_terms": top_terms,
            "term_frequencies": term_frequencies
        }

        session.store_result("term_frequency_tool", {}, summary, full_report=full_report)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Render a heatmap of a positional slice of the raw document-term matrix.

        Args:
            n_terms: Number of terms (columns) to slice from the DTM (default 20).
            n_documents: Number of documents (rows) to slice from the DTM (default 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        terms_slice = session.feature_names[0:n_terms]
        submatrix_sparse = session.feature_matrix[0:n_documents, 0:n_terms]
        dense_matrix = submatrix_sparse.toarray()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            dense_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms_slice,
            yticklabels=[f"Doc {i}" for i in range(dense_matrix.shape[0])],
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (Top {dense_matrix.shape[0]} Docs x {dense_matrix.shape[1]} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents_sliced": dense_matrix.shape[0],
            "n_terms_sliced": dense_matrix.shape[1],
            "terms": terms_slice,
            "matrix": dense_matrix.tolist()
        }

        session.store_result("dtm_heatmap_tool", {
            "n_terms": n_terms,
            "n_documents": n_documents
        }, summary)

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
