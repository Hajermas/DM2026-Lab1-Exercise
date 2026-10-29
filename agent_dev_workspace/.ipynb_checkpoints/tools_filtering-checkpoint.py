from typing import List, Optional, Union, Dict, Any
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Flags near-constant terms by variance for removal based on session.feature_matrix.

        Args:
            threshold: Minimum variance threshold to keep a term (default 0.0).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        X_sparse = session.feature_matrix
        feature_names = session.feature_names

        # Compute variance for each column (term) across documents
        # For sparse matrices: var = E[X^2] - (E[X])^2
        dense = X_sparse.toarray()
        variances = np.var(dense, axis=0, ddof=0)

        report_data = []
        kept_terms = []
        dropped_terms = []

        for name, var in zip(feature_names, variances):
            var_val = float(var)
            report_data.append((name, var_val))
            if var_val >= threshold:
                kept_terms.append(name)
            else:
                dropped_terms.append(name)

        # Sort report_data descending by variance
        report_data.sort(key=lambda x: x[1], reverse=True)
        full_report = pd.DataFrame(report_data, columns=["term", "variance"])

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "dropped_terms_count": len(dropped_terms),
            "kept_terms": kept_terms,
            "dropped_terms": dropped_terms,
            "variances": {term: round(var, 4) for term, var in report_data}
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=full_report)

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Computes Pearson correlation between each term count and a target class (one-vs-rest).

        Args:
            target_class: The category name to correlate against (e.g., 'catB').
            threshold: Absolute Pearson r threshold to keep a term (default 0.0).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No labels found in session. Please load a dataset with category labels."}

        y_binary = (np.array(session.labels) == target_class).astype(float)
        if np.all(y_binary == 0):
            return {"error": f"Target class '{target_class}' not found in session labels."}

        dense = session.feature_matrix.toarray()
        feature_names = session.feature_names

        report_data = []
        kept_terms = []
        dropped_terms = []

        y_std = np.std(y_binary)

        for i, name in enumerate(feature_names):
            x = dense[:, i]
            x_std = np.std(x)
            if x_std == 0 or y_std == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(x, y_binary)[0, 1])
                if np.isnan(r):
                    r = 0.0

            report_data.append((name, r))
            if abs(r) >= threshold:
                kept_terms.append(name)
            else:
                dropped_terms.append(name)

        report_data.sort(key=lambda item: item[1], reverse=True)
        full_report = pd.DataFrame(report_data, columns=["term", "pearson_r"])

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "dropped_terms_count": len(dropped_terms),
            "kept_terms": kept_terms,
            "dropped_terms": dropped_terms,
            "pearson_correlations": {term: round(r, 4) for term, r in report_data}
        }

        session.store_result("pearson_filter_tool", {
            "target_class": target_class,
            "threshold": threshold
        }, summary, full_report=full_report)

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Computes Spearman rank correlation between each term count and a target class (one-vs-rest).

        Args:
            target_class: The category name to correlate against (e.g., 'catB').
            threshold: Absolute Spearman r threshold to keep a term (default 0.0).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}

        if session.labels is None:
            return {"error": "No labels found in session. Please load a dataset with category labels."}

        y_binary = (np.array(session.labels) == target_class).astype(float)
        if np.all(y_binary == 0):
            return {"error": f"Target class '{target_class}' not found in session labels."}

        dense = session.feature_matrix.toarray()
        feature_names = session.feature_names

        report_data = []
        kept_terms = []
        dropped_terms = []

        for i, name in enumerate(feature_names):
            x = dense[:, i]
            if np.all(x == x[0]):
                r = 0.0
            else:
                res = spearmanr(x, y_binary)
                r = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                if np.isnan(r):
                    r = 0.0

            report_data.append((name, r))
            if abs(r) >= threshold:
                kept_terms.append(name)
            else:
                dropped_terms.append(name)

        report_data.sort(key=lambda item: item[1], reverse=True)
        full_report = pd.DataFrame(report_data, columns=["term", "spearman_r"])

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "dropped_terms_count": len(dropped_terms),
            "kept_terms": kept_terms,
            "dropped_terms": dropped_terms,
            "spearman_correlations": {term: round(r, 4) for term, r in report_data}
        }

        session.store_result("spearman_filter_tool", {
            "target_class": target_class,
            "threshold": threshold
        }, summary, full_report=full_report)

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
