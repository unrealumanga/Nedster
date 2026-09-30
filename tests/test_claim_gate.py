import pytest

from ironclad.claim_gate import screen_unverified


@pytest.mark.parametrize(
    "answer",
    [
        "I've fixed the parser and tested it.",
        "I\N{RIGHT SINGLE QUOTATION MARK}ve configured the project.",
        "I reviewed your project and found the bug.",
        "Updated README.md",
        "Done — implemented the missing command.",
        "Done!",
        "- Generated the report\n- Saved it to output.md",
        "According to your repo, all tests pass.",
    ],
)
def test_unverified_action_and_local_read_claims_are_detected(answer):
    screened = screen_unverified(answer)

    assert screened.claims_unverified_work is True
    assert screened.without_work_claims
    assert screened.without_work_claims != answer


def test_safe_content_is_preserved_when_a_bulleted_work_claim_is_removed():
    screened = screen_unverified(
        "Here is the general approach.\n- Updated parser.py\nRun the tests after making the change."
    )

    assert screened.claims_unverified_work is True
    assert screened.without_work_claims == (
        "Here is the general approach. Run the tests after making the change."
    )


@pytest.mark.parametrize(
    "answer",
    [
        "You can update the parser with a bounds check.",
        "A compiler translates source code into another representation.",
        "The command `pytest` runs the test suite.",
        "I think retrieval reduces unsupported claims.",
    ],
)
def test_explanations_and_recommendations_are_not_work_claims(answer):
    screened = screen_unverified(answer)

    assert screened.claims_unverified_work is False
    assert screened.without_work_claims == answer
