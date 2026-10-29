import os
import tempfile
from typing import Dict, Any, Optional, Literal, List
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
from PAMI.frequentPattern.topk.FAE import FAE
from PAMI.frequentPattern.maximal.MaxFPGrowth import MaxFPGrowth


@tool
def mine_patterns_tool(
    category_name: str,
    filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
    algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
    min_sup: Optional[int] = 1,
    k: Optional[int] = 10,
) -> Dict[str, Any]:
    """Mines frequent itemsets/patterns for a specific document category using PAMI algorithms.

    Args:
        category_name: Target category name to filter and mine.
        filtering_method: Method to filter vocabulary before mining ('variance', 'tfidf', or 'term_frequency').
        algorithm: Mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
        min_sup: Minimum support count required for 'fpgrowth' and 'maxfpgrowth'.
        k: Number of top patterns to return for 'topk' (FAE algorithm).

    Returns:
        Dict containing result_id, category_name, filtering_method, algorithm, and mined patterns.
    """
    if session.dataframe is None or "category_name" not in session.dataframe.columns:
        return {"error": "session.dataframe missing or lacks 'category_name' column."}

    # Step 1: Filter DataFrame by category and create category-local CountVectorizer with English stopwords
    cat_df = session.dataframe[session.dataframe["category_name"] == category_name].copy()
    if cat_df.empty:
        return {"error": f"No documents found for category '{category_name}'."}

    texts = cat_df["text"].fillna("").tolist()
    
    vectorizer = CountVectorizer(stop_words="english")
    try:
        dtm_counts = vectorizer.fit_transform(texts)
    except ValueError:
        # Happens if text is empty or only stopwords
        return {
            "result_id": session.next_result_id("patterns"),
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": [],
            "note": "no terms survived stopword removal",
        }

    terms = np.array(vectorizer.get_feature_names_out())
    if len(terms) == 0:
        return {
            "result_id": session.next_result_id("patterns"),
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": [],
            "note": "no terms survived stopword removal",
        }

    dense_dtm = dtm_counts.toarray()

    # Step 2: Apply per-category vocabulary filtering
    if filtering_method == "variance":
        variances = np.var(dense_dtm, axis=0)
        p5, p95 = np.percentile(variances, 5), np.percentile(variances, 95)
        mask = (variances > p5) & (variances < p95)
    elif filtering_method == "term_frequency":
        tf_sums = np.sum(dense_dtm, axis=0)
        p5, p95 = np.percentile(tf_sums, 5), np.percentile(tf_sums, 95)
        mask = (tf_sums > p5) & (tf_sums < p95)
    elif filtering_method == "tfidf":
        tfidf_trans = TfidfTransformer()
        tfidf_mat = tfidf_trans.fit_transform(dense_dtm).toarray()
        mean_tfidf = np.mean(tfidf_mat, axis=0)
        p20 = np.percentile(mean_tfidf, 20)
        mask = mean_tfidf >= p20
    else:
        return {"error": f"Unsupported filtering method: {filtering_method}"}

    kept_terms = terms[mask]
    kept_dtm = dense_dtm[:, mask]

    # Step 3: Handle zero terms kept
    if len(kept_terms) == 0:
        res_id = session.next_result_id("patterns")
        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": [],
            "note": "no terms survived filtering",
        }
        session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
        session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
        return summary

    # Step 4: Convert surviving terms into transactional database DataFrame
    df_binary = pd.DataFrame(kept_dtm, columns=kept_terms)
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
        temp_tx_file = tmp.name

    try:
        converter = DF2DB(df_binary)
        converter.convert2TransactionalDatabase(temp_tx_file, ">=", 1)

        # Step 5: Execute PAMI mining algorithm
        if algorithm == "fpgrowth":
            sup_val = min_sup if min_sup is not None else 1
            miner = FPGrowth(temp_tx_file, minSup=sup_val)
        elif algorithm == "topk":
            k_val = k if k is not None else 10
            miner = FAE(temp_tx_file, k=k_val)
        elif algorithm == "maxfpgrowth":
            sup_val = min_sup if min_sup is not None else 1
            miner = MaxFPGrowth(temp_tx_file, minSup=sup_val)
        else:
            return {"error": f"Unsupported pattern mining algorithm: {algorithm}"}

        miner.mine()
        patterns_df = miner.getPatternsAsDataFrame()

        patterns = []
        if patterns_df is not None and not patterns_df.empty:
            for _, row in patterns_df.iterrows():
                p_str = str(row.iloc[0])
                sup = int(row.iloc[1])
                patterns.append({"pattern": p_str, "support": sup})

    finally:
        if os.path.exists(temp_tx_file):
            os.remove(temp_tx_file)

    res_id = session.next_result_id("patterns")
    summary = {
        "result_id": res_id,
        "category_name": category_name,
        "filtering_method": filtering_method,
        "algorithm": algorithm,
        "pattern_count": len(patterns),
        "patterns": patterns,
    }

    session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = patterns
    session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
    return summary


def make_tools(session_obj):
    global session
    session = session_obj
    return [mine_patterns_tool]
