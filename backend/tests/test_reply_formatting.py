from phase8_api.reply_formatting import _all_tool_payloads, _card_id_to_name_map, extract_reply, format_reply


def test_extract_reply_skips_trailing_critic_approved_verdict():
    messages = [
        {"role": "assistant", "content": "Use HDFC Millennia for this purchase."},
        {"role": "assistant", "content": "APPROVED"},
    ]
    assert extract_reply(messages) == "Use HDFC Millennia for this purchase."


def test_extract_reply_skips_trailing_critic_revise_verdict():
    messages = [
        {"role": "assistant", "content": "Use HDFC Infinia."},
        {"role": "assistant", "content": "REVISE: missed a better offer"},
        {"role": "user", "content": "A reviewer flagged an issue..."},
        {"role": "assistant", "content": "Use HDFC Millennia instead."},
        {"role": "assistant", "content": "APPROVED"},
    ]
    assert extract_reply(messages) == "Use HDFC Millennia instead."


def test_extract_reply_returns_plain_message_when_no_critic_verdict_present():
    messages = [{"role": "assistant", "content": "The user declined this recommendation. No action was taken."}]
    assert extract_reply(messages) == "The user declined this recommendation. No action was taken."


def test_format_reply_turns_raw_card_suggestion_into_customer_facing_recommendation():
    assert format_reply("Use HDFC Millennia for this purchase.") == "For this purchase, I would recommend HDFC Millennia."


def test_format_reply_handles_a_remainder_that_already_starts_with_card():
    # Regression test for a dead branch removed during cleanup: the
    # original code special-cased a remainder starting with "card" but
    # returned the exact same string either way. This confirms that case
    # still behaves correctly now that the redundant branch is gone.
    assert format_reply("Use card that has the best rate for this purchase.") == (
        "For this purchase, I would recommend card that has the best rate."
    )


def test_format_reply_replaces_placeholder_card_identifiers_with_plain_language():
    raw_reply = 'Considering the user\'s past transactions, they have been using "card_a" for groceries and have a history of transactions with it.'
    assert format_reply(raw_reply) == "For this purchase, I would recommend the option that best fits your spending habits and rewards."


def test_format_reply_uses_card_tool_results_to_restore_specific_card_name():
    raw_reply = 'Considering the user\'s past transactions, they have been using "card_a" for groceries and have a history of transactions with it.'
    messages = [
        {
            "role": "tool",
            "name": "check_card_rewards",
            "content": '[{"card_id":"card_b","card_name":"Axis Bank Magnus Credit Card","reward_rate":0.012}]',
        }
    ]
    assert format_reply(raw_reply, messages) == "For this purchase, I would recommend Axis Bank Magnus Credit Card."


def test_format_reply_preserves_specific_card_names():
    raw_reply = "The best option is HDFC Millennia because the grocery rewards are strongest."
    assert format_reply(raw_reply) == raw_reply


def test_format_reply_strips_trailing_raw_id_but_preserves_elaborate_explanation():
    # A raw internal id trailing right after the real card name (e.g. the
    # model wrote "HDFC Millennia Credit Card (card_a)") must not nuke the
    # whole explanation the way a bare, unresolved id does. Only the
    # parenthetical annotation itself should go.
    raw_reply = (
        "I recommend the HDFC Millennia Credit Card (card_a) for this purchase. "
        "It offers 1% cashback plus a 10% active offer at BigBasket, which beats HDFC Infinia's 3.3% "
        "base rate once the offer is applied. The annual fee is Rs 1000."
    )
    assert format_reply(raw_reply) == (
        "I recommend the HDFC Millennia Credit Card for this purchase. "
        "It offers 1% cashback plus a 10% active offer at BigBasket, which beats HDFC Infinia's 3.3% "
        "base rate once the offer is applied. The annual fee is Rs 1000."
    )


def test_format_reply_resolves_quoted_raw_ids_referenced_while_discussing_past_transactions():
    # The model can quote a raw internal id while reasoning about past
    # transaction history (retrieve_memory's "card_used" field), not just
    # while naming its own recommendation. If check_card_rewards's tool
    # payload resolved that id to a real name, the whole elaborate answer
    # should survive with the id swapped for the name, not get discarded
    # just because a bare "card_x" token showed up somewhere in the text.
    raw_reply = (
        'I recommend the ICICI Amazon Pay Credit Card for fuel purchases. This card offers a 1% reward rate, '
        'the highest among the available options. Although "card_c" has been used for fuel purchases in the past, '
        'its reward rate is also 1%, matching the recommendation. "Card_e" has also been used for fuel purchases, '
        "but its reward rate is 0%, making the ICICI Amazon Pay Credit Card the better option."
    )
    messages = [
        {
            "role": "tool",
            "name": "check_card_rewards",
            "content": (
                '[{"card_id":"card_c","card_name":"IDFC FIRST Select Credit Card","reward_rate":0.01},'
                '{"card_id":"card_e","card_name":"SBI Cashback Credit Card","reward_rate":0.0}]'
            ),
        }
    ]

    assert format_reply(raw_reply, messages) == (
        'I recommend the ICICI Amazon Pay Credit Card for fuel purchases. This card offers a 1% reward rate, '
        'the highest among the available options. Although IDFC FIRST Select Credit Card has been used for fuel purchases in the past, '
        'its reward rate is also 1%, matching the recommendation. SBI Cashback Credit Card has also been used for fuel purchases, '
        "but its reward rate is 0%, making the ICICI Amazon Pay Credit Card the better option."
    )


def test_all_tool_payloads_collects_every_matching_call_in_order():
    messages = [
        {"role": "tool", "name": "check_card_rewards", "content": '[{"card_id": "card_a", "card_name": "Card A"}]'},
        {"role": "assistant", "content": "thinking"},
        {"role": "tool", "name": "check_card_rewards", "content": '[{"card_id": "card_b", "card_name": "Card B"}]'},
    ]
    payloads = _all_tool_payloads(messages, "check_card_rewards")
    assert payloads == [
        [{"card_id": "card_a", "card_name": "Card A"}],
        [{"card_id": "card_b", "card_name": "Card B"}],
    ]


def test_all_tool_payloads_skips_unparseable_content():
    messages = [{"role": "tool", "name": "check_card_rewards", "content": "not json"}]
    assert _all_tool_payloads(messages, "check_card_rewards") == []


def test_card_id_to_name_map_merges_ids_across_multiple_tool_calls():
    messages = [
        {"role": "tool", "name": "check_card_rewards", "content": '[{"card_id": "card_a", "card_name": "Card A"}]'},
        {"role": "tool", "name": "check_card_rewards", "content": '[{"card_id": "card_b", "card_name": "Card B"}]'},
    ]
    assert _card_id_to_name_map(messages) == {"card_a": "Card A", "card_b": "Card B"}


def test_card_id_to_name_map_ignores_entries_missing_a_name_or_id():
    messages = [
        {"role": "tool", "name": "check_card_rewards", "content": '[{"card_id": "card_a"}, {"card_name": "No Id"}]'},
    ]
    assert _card_id_to_name_map(messages) == {}
