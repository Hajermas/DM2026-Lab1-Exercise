import os
import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def _load_fixture_dataset(session):
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })


def test_tokenize_tool():
    session = SessionState()
    _load_fixture_dataset(session)
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    
    res = tokenize_tool.invoke({})
    assert res["n_documents"] == 8
    
    # Check known answer 10
    expected_tokens_doc0 = ["always", "alpha", "alpha", "alpha", "beta"]
    expected_tokens_doc7 = []
    assert session.dataframe["unigrams"].iloc[0] == expected_tokens_doc0
    assert session.dataframe["unigrams"].iloc[7] == expected_tokens_doc7


def test_list_files_tool():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    # Known answer 11
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool():
    session = SessionState()
    _load_fixture_dataset(session)
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    res = inspect_data_tool.invoke({"n_rows": 3})
    assert res["n_rows"] == 8
    assert res["n_cols"] == len(session.dataframe.columns)
    assert len(res["sample"]) == 3


def test_check_missing_tool():
    session = SessionState()
    _load_fixture_dataset(session)
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")
    
    # Known answer 2
    res = check_missing_tool.invoke({})
    assert res["missing_count"] == 1
    assert 7 in res["missing_indices"]


def test_check_duplicates_tool():
    session = SessionState()
    _load_fixture_dataset(session)
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    # Known answer 3
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["n_duplicates"] == 1
    assert res_check["remaining_rows"] == 8
    
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["remaining_rows"] == 6


def test_sample_data_tool():
    session = SessionState()
    _load_fixture_dataset(session)
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")
    
    # Known answer 13
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["n_sampled"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_describe_data_tool():
    session = SessionState()
    _load_fixture_dataset(session)
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")
    
    # Known answer 18
    res = describe_data_tool.invoke({})
    overall = res["overall"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert np.isclose(overall["min"], 0, atol=1e-3)
    assert np.isclose(overall["25%"], 17.75, atol=1e-3)
    assert np.isclose(overall["50%"], 20.5, atol=1e-3)
    assert np.isclose(overall["75%"], 25.25, atol=1e-3)
    assert np.isclose(overall["max"], 30, atol=1e-3)
    
    catA = res["per_category"]["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    
    catB = res["per_category"]["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    
    assert session.pending_figure is not None
