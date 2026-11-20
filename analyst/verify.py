import re

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_FACT_ID = re.compile(r"\bF\d+\b")

def extract_numbers(text: str) -> list[float]:
    text = _FACT_ID.sub(" ", text)
    return [float(x.replace(",", ".")) for x in _NUM.findall(text)]

def unsupported_numbers(llm_text: str, source_text: str) -> list[float]:
    source = extract_numbers(source_text)
    candidates = source + [s * 100 for s in source]
    bad = []
    for n in extract_numbers(llm_text):
        if n in (0, 1) or 1990 <= n <= 2100:  # тривіальні числа та роки
            continue
        if not any(abs(n - c) <= max(0.03 * abs(c), 0.051) for c in candidates):
            bad.append(n)
    return bad