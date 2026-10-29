import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def setup_fixture_session():
    """Helper to initialize session and load sample fixture."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session


def test_mine_patterns_zero_terms():
    """Test that variance and term_frequency filtering methods return zero terms cleanly."""
    session = setup_fixture_session()
    pattern_tools = make_patterns_tools(session)
    mine_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Variance filtering keeps 0 terms on this 2-term vocabulary
    res = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res["patterns"] == []
    assert res.get("note") == "no terms survived filtering"


def test_mine_patterns_fpgrowth():
    """Test known answer #17: fpgrowth algorithm with tfidf filtering."""
    session = setup_fixture_session()
    pattern_tools = make_patterns_tools(session)
    mine_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # catA at min_sup=1 -> pattern alpha with support 3
    res_a = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_a["patterns"]) == 1
    assert "alpha" in res_a["patterns"][0]["pattern"]
    assert res_a["patterns"][0]["support"] == 3

    # catB at min_sup=1 -> pattern gamma with support 4
    res_b = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_b["patterns"]) == 1
    assert "gamma" in res_b["patterns"][0]["pattern"]
    assert res_b["patterns"][0]["support"] == 4

    # catA at min_sup=4 -> 0 patterns
    res_a_high = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert len(res_a_high["patterns"]) == 0


def test_mine_patterns_topk():
    """Test known answer #20: topk (FAE) algorithm with tfidf filtering."""
    session = setup_fixture_session()
    pattern_tools = make_patterns_tools(session)
    mine_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # catA at k=1 -> pattern alpha with support 3
    res_a = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_a["patterns"]) == 1
    assert "alpha" in res_a["patterns"][0]["pattern"]
    assert res_a["patterns"][0]["support"] == 3

    # catB at k=1 -> pattern gamma with support 4
    res_b = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_b["patterns"]) == 1
    assert "gamma" in res_b["patterns"][0]["pattern"]
    assert res_b["patterns"][0]["support"] == 4


def test_mine_patterns_maxfpgrowth():
    """Test known answer #20: maxfpgrowth algorithm with tfidf filtering returns zero patterns."""
    session = setup_fixture_session()
    pattern_tools = make_patterns_tools(session)
    mine_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # maxfpgrowth returns 0 patterns on single-term database
    res_a = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert len(res_a["patterns"]) == 0

    res_b = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert len(res_b["patterns"]) == 0
