# Deterministic, model-free checks that decide whether approval_gate()
# should pause the graph. Kept as pure functions, separate from the graph
# node itself, so the trigger logic is unit testable without a live model
# call, the same split Phase 5 used for _should_revise().
import re

LARGE_PURCHASE_THRESHOLD = 5000

# Matches "₹5000", "₹5,000", "Rs 5000", "Rs. 5,000", "Rs.5000", "5000 rupees".
# Commas inside the number are handled by stripping them before int().
_RUPEE_PATTERN = re.compile(
    r"(?:₹\s?|\brs\.?\s?)([\d,]+)|([\d,]+)\s?rupees",
    re.IGNORECASE,
)


def _extract_rupee_amount(query: str) -> int | None:
    amounts = []
    for match in _RUPEE_PATTERN.finditer(query):
        raw = match.group(1) or match.group(2)
        amounts.append(int(raw.replace(",", "")))
    return max(amounts) if amounts else None


def _mentions_cancellation(text: str) -> bool:
    return "cancel" in text.lower()
