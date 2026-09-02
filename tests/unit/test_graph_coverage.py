from __future__ import annotations
import collections
import pytest
from unittest.mock import AsyncMock, patch
from src.agent.graph import route_after_read, route_after_test, build_graph

def test_route_after_read():
    assert route_after_read({"status": "failed"}) == "__end__"
    assert route_after_read({"status": "running"}) == "plan"

def test_route_after_test():
    TestResult = collections.namedtuple("TestResult", ["passed", "coverage_pct"])
    
    assert route_after_test({"test_result": TestResult(True, 100.0)}) == "open_pr"
    assert route_after_test({"test_result": TestResult(False, 0.0), "retry_count": 0}) == "correct"
    assert route_after_test({"test_result": TestResult(False, 0.0), "retry_count": 50}) == "fail"

@pytest.mark.asyncio
async def test_wrapper_invocation():
    from src.agent.graph import build_graph
    from src.agent import nodes

    with patch("src.agent.nodes.read_issue_node", new_callable=AsyncMock) as mock_read, \
         patch("src.agent.nodes.plan_node", new_callable=AsyncMock) as mock_plan, \
         patch("src.agent.nodes.code_node", new_callable=AsyncMock) as mock_code, \
         patch("src.agent.nodes.test_node", new_callable=AsyncMock) as mock_test, \
         patch("src.agent.nodes.correct_node", new_callable=AsyncMock) as mock_correct, \
         patch("src.agent.nodes.open_pr_node", new_callable=AsyncMock) as mock_open, \
         patch("src.agent.nodes.fail_node", new_callable=AsyncMock) as mock_fail:

        graph = build_graph()
        
        await graph.nodes["read_issue"].runnable.ainvoke({"run_id": "123"}, {})
        mock_read.assert_awaited_once()

        await graph.nodes["plan"].runnable.ainvoke({"run_id": "123"}, {})
        mock_plan.assert_awaited_once()

        await graph.nodes["code"].runnable.ainvoke({"run_id": "123"}, {})
        mock_code.assert_awaited_once()

        await graph.nodes["test"].runnable.ainvoke({"run_id": "123"}, {})
        mock_test.assert_awaited_once()

        await graph.nodes["correct"].runnable.ainvoke({"run_id": "123"}, {})
        mock_correct.assert_awaited_once()

        await graph.nodes["open_pr"].runnable.ainvoke({"run_id": "123"}, {})
        mock_open.assert_awaited_once()

        await graph.nodes["fail"].runnable.ainvoke({"run_id": "123"}, {})
        mock_fail.assert_awaited_once()

    from src.agent.graph import get_compiled_graph
    compiled = await get_compiled_graph()
    assert compiled is not None

    from langgraph.checkpoint.memory import MemorySaver
    compiled_with_cp = await get_compiled_graph(checkpointer=MemorySaver())
    assert compiled_with_cp is not None
