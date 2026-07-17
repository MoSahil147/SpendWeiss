from unittest.mock import patch

from phase6_multiagent.subscription_hunter import find_recurring_charges
from phase6_multiagent.supervisor import (
    _normalise_classification,
    _stream_with_trace,
    _summarize_update,
    dispatch,
)


def test_find_recurring_charges_finds_netflix():
    recurring = find_recurring_charges()
    netflix = next(entry for entry in recurring if entry["merchant"] == "Netflix")
    assert netflix["occurrences"] == 5
    assert netflix["category"] == "other"


def test_find_recurring_charges_finds_cultfit():
    recurring = find_recurring_charges()
    cultfit = next(entry for entry in recurring if entry["merchant"] == "Cult.fit")
    assert cultfit["occurrences"] == 3


def test_find_recurring_charges_excludes_single_occurrence_merchants():
    # DMart appears exactly once in the mock data (2026-05-19), so it must
    # not show up in the recurring list at all, unlike Netflix or Cult.fit.
    recurring = find_recurring_charges()
    merchants = {entry["merchant"] for entry in recurring}
    assert "DMart" not in merchants


def test_normalise_classification_passes_through_valid_values():
    assert _normalise_classification("card_optimizer") == "card_optimizer"
    assert _normalise_classification("subscription_hunter") == "subscription_hunter"
    assert _normalise_classification("both") == "both"


def test_normalise_classification_falls_back_to_card_optimizer():
    assert _normalise_classification("") == "card_optimizer"
    assert _normalise_classification("garbage") == "card_optimizer"


def test_summarize_update_with_dict_message():
    update = {"messages": [{"role": "assistant", "content": "APPROVED"}]}
    assert _summarize_update("critic", update) == "critic: APPROVED"


def test_summarize_update_with_object_message():
    class FakeMessage:
        content = "Use HDFC Millennia for this purchase."

    update = {"messages": [FakeMessage()]}
    assert _summarize_update("respond", update) == "respond: Use HDFC Millennia for this purchase."


def test_summarize_update_truncates_long_content():
    update = {"messages": [{"role": "assistant", "content": "x" * 300}]}
    result = _summarize_update("respond", update)
    assert result == "respond: " + "x" * 200


def test_summarize_update_with_no_messages_key():
    assert _summarize_update("call_tool", {}) == "call_tool ran"


class _FakeStreamGraph:
    # Stands in for a CompiledStateGraph: yields the same (mode, chunk)
    # shape LangGraph's own .stream(stream_mode=["updates", "values"])
    # yields, confirmed directly against a real two-node graph before
    # writing this plan.
    def stream(self, input_data, stream_mode):
        assert stream_mode == ["updates", "values"]
        yield "values", {"messages": input_data["messages"]}
        yield "updates", {"reason": {"messages": [{"role": "assistant", "content": "thinking"}]}}
        yield "values", {"messages": input_data["messages"] + [{"role": "assistant", "content": "thinking"}]}
        yield "updates", {"respond": {"messages": [{"role": "assistant", "content": "Use card X"}]}}
        yield "values", {"messages": input_data["messages"] + [
            {"role": "assistant", "content": "thinking"},
            {"role": "assistant", "content": "Use card X"},
        ]}


def test_stream_with_trace_returns_final_state_and_ordered_trace():
    final_state, trace = _stream_with_trace(_FakeStreamGraph(), {"messages": []}, "card_optimizer")

    assert final_state["messages"][-1]["content"] == "Use card X"
    assert [entry["node"] for entry in trace] == ["reason", "respond"]
    assert all(entry["graph"] == "card_optimizer" for entry in trace)
    assert trace[1]["summary"] == "respond: Use card X"


def test_dispatch_card_optimizer_returns_messages_and_trace():
    fake_final_state = {"messages": [{"role": "assistant", "content": "Use card X"}]}
    fake_trace = [{"node": "respond", "graph": "card_optimizer", "summary": "respond: Use card X"}]

    with patch("phase6_multiagent.supervisor._stream_with_trace", return_value=(fake_final_state, fake_trace)) as mock_stream:
        messages, trace = dispatch("card_optimizer", [{"role": "user", "content": "hi"}])

    assert messages == fake_final_state["messages"]
    assert trace == fake_trace
    mock_stream.assert_called_once()
    assert mock_stream.call_args.args[2] == "card_optimizer"


def test_dispatch_both_concatenates_traces_from_both_specialists():
    card_state = {"messages": [{"role": "assistant", "content": "Use card X"}]}
    card_trace = [{"node": "respond", "graph": "card_optimizer", "summary": "respond: Use card X"}]
    sub_state = {"messages": [{"role": "assistant", "content": "Use card X"}, {"role": "assistant", "content": "Cancel Netflix"}]}
    sub_trace = [{"node": "model", "graph": "subscription_hunter", "summary": "model: Cancel Netflix"}]

    with patch("phase6_multiagent.supervisor.get_subscription_hunter_agent"), patch(
        "phase6_multiagent.supervisor._stream_with_trace", side_effect=[(card_state, card_trace), (sub_state, sub_trace)]
    ):
        messages, trace = dispatch("both", [{"role": "user", "content": "hi"}])

    assert messages == sub_state["messages"]
    assert trace == card_trace + sub_trace
