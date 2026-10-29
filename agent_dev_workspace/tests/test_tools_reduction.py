import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def setup_fixture_session():
    """Helper to initialize session, load sample fixture, and build DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_pca():
    """Test known answer #7: PCA reduction on sample_fixture.csv."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    expected_explained_var = [0.7532, 0.1965]

    coords = np.array(res["coordinates"])
    explained_var = res["explained_variance_ratio"]

    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(explained_var, expected_explained_var, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    """Test t-SNE reduction on sample_fixture.csv with perplexity=2, random_state=42."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})

    expected_coords = np.array([
        [986.3912, 895.4789],
        [532.2820, 1125.0872],
        [49.2527, 1230.0204],
        [-1088.9310, 2.2856],
        [-517.0869, -274.7408],
        [-665.1573, -1313.2200],
        [-665.1573, -1313.2200],
        [1300.3058, -742.4253]
    ])

    coords = np.array(res["coordinates"])

    assert np.allclose(coords, expected_coords, atol=1e-2)
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    """Test known answer #9: UMAP reduction on sample_fixture.csv with n_neighbors=3, random_state=42."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})

    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    coords = np.array(res["coordinates"])

    assert np.allclose(coords, expected_coords, atol=1e-2)
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    """Test 5 logical invariants for binarize_labels_tool on sample_fixture.csv."""
    session = setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    binarized_df = session.artifacts.get("binarized_labels")
    assert binarized_df is not None

    # Invariant 1: Columns list matches expected dummy variable names
    expected_cols = ["category_catA", "category_catB"]
    assert res["columns"] == expected_cols
    assert binarized_df.columns.tolist() == expected_cols

    # Invariant 2: Shape matches (n_samples, n_categories) = (8, 2)
    assert res["shape"] == [8, 2]
    assert binarized_df.shape == (8, 2)

    # Invariant 3: Rows are one-hot encoded (row sum equals 1 for each row)
    assert (binarized_df.sum(axis=1) == 1).all()

    # Invariant 4: Column data types are integer
    assert all(dtype == np.int64 or dtype == int or dtype == "int32" or dtype == "int64" for dtype in binarized_df.dtypes)

    # Invariant 5: Category counts match dataset category counts (4 catA, 4 catB)
    assert binarized_df["category_catA"].sum() == 4
    assert binarized_df["category_catB"].sum() == 4
