import os
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text in session.dataframe into unigram tokens using NLTK.
        
        Returns:
            Dict[str, Any]: Summary containing total tokenized documents and average token count.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session."}
        
        # Ensure NLTK punkt tokenizer data is available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            
        def _tokenize(text):
            if not isinstance(text, str) or not text.strip():
                return []
            return nltk.word_tokenize(text.lower())
        
        tokens_list = session.dataframe["text"].apply(_tokenize).tolist()
        session.dataframe["unigrams"] = tokens_list
        session.artifacts["unigrams"] = tokens_list
        
        total_tokens = sum(len(t) for t in tokens_list)
        avg_tokens = total_tokens / len(tokens_list) if tokens_list else 0.0
        
        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": len(tokens_list),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 4)
        }
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and folders in a specified relative subdirectory.
        
        Args:
            subdirectory: Path relative to workspace/repo root.
            
        Returns:
            Dict[str, Any]: Lists of directory names and file names found.
        """
        target_dir = subdirectory
        if not os.path.exists(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist."}
            
        directories = []
        files = []
        for entry in os.scandir(target_dir):
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)
                
        directories.sort()
        files.sort()
        
        summary = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> Dict[str, Any]:
        """ Inspects top n_rows of the working dataset in session.dataframe.
        
        Args:
            n_rows: Number of sample rows to inspect.
            
        Returns:
            Dict[str, Any]: Columns list, row count, column count, and sample records.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session."}
            
        head_df = session.dataframe.head(n_rows)
        summary = {
            "result_id": session.next_result_id("inspect_data"),
            "n_rows": len(session.dataframe),
            "n_cols": len(session.dataframe.columns),
            "columns": list(session.dataframe.columns),
            "sample": head_df.to_dict(orient="records")
        }
        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text values in session.dataframe.
        
        Returns:
            Dict[str, Any]: Number of missing/empty text rows found.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session."}
            
        def is_empty(val):
            if pd.isna(val):
                return True
            if isinstance(val, str) and not val.strip():
                return True
            return False
            
        missing_mask = session.dataframe["text"].apply(is_empty)
        missing_count = int(missing_mask.sum())
        
        summary = {
            "result_id": session.next_result_id("check_missing"),
            "total_rows": len(session.dataframe),
            "missing_count": missing_count,
            "missing_indices": session.dataframe.index[missing_mask].tolist()
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for exact text duplicates in session.dataframe, and optionally drops them using keep=False.
        
        Args:
            drop: If True, drops ALL copies of duplicated rows (keep=False) in place.
            
        Returns:
            Dict[str, Any]: Duplicate count and remaining row count.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session."}
            
        # duplicates mask: duplicated text entries
        dup_mask = session.dataframe.duplicated(subset=["text"], keep=False)
        duplicate_count = int(session.dataframe.duplicated(subset=["text"], keep="first").sum())
        
        if drop:
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)
            
        summary = {
            "result_id": session.next_result_id("check_duplicates"),
            "n_duplicates": duplicate_count,
            "remaining_rows": len(session.dataframe),
            "dropped": drop
        }
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 4, random_state: int = 42) -> Dict[str, Any]:
        """Subsamples n documents from session.dataframe using a random seed.
        
        Args:
            n: Number of rows to sample.
            random_state: Seed for reproducibility.
            
        Returns:
            Dict[str, Any]: Sampled row indices and summary information.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session."}
            
        sampled = session.dataframe.sample(n=n, random_state=random_state)
        sampled_indices = sampled.index.tolist()
        
        summary = {
            "result_id": session.next_result_id("sample_data"),
            "n_sampled": len(sampled),
            "sampled_indices": sampled_indices,
            "random_state": random_state
        }
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes text length statistics overall and per category, and builds a boxplot set on session.pending_figure.
        
        Returns:
            Dict[str, Any]: Overall and per-category text length summary statistics.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session."}
            
        df = session.dataframe.copy()
        df["text_length"] = df["text"].apply(lambda s: len(s) if isinstance(s, str) else 0)
        
        overall_desc = df["text_length"].describe().to_dict()
        
        per_category = {}
        if "category_name" in df.columns:
            for cat_name, group in df.groupby("category_name"):
                per_category[str(cat_name)] = group["text_length"].describe().to_dict()
                
        # Build boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df.columns:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")
            
        ax.set_ylabel("Text Length (characters)")
        session.pending_figure = fig
        
        summary = {
            "result_id": session.next_result_id("describe_data"),
            "overall": overall_desc,
            "per_category": per_category
        }
        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
