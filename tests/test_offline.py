import contextlib
import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyst import anomaly, eda, hypotheses, nl2pandas
from analyst.verify import unsupported_numbers
from generate_synthetic import make_synthetic

DF = make_synthetic()


@contextlib.contextmanager
def patched(module, name, fake):
    old = getattr(module, name)
    setattr(module, name, fake)
    try:
        yield
    finally:
        setattr(module, name, old)


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    return False

def test_verify_numbers():
    src = "conversion 6.12% n=500"
    assert unsupported_numbers("конверсія 6.1% у 500 сесіях", src) == []
    assert unsupported_numbers("конверсія 42% у 500 сесіях", src) == [42.0]

def test_profile_contains_all_columns_and_known_issues():
    f = eda.profile_dataframe(DF)
    assert len(f["columns"]) == DF.shape[1], "LLM має отримати інформацію про КОЖНУ колонку"
    assert f["duplicate_rows"] == 200
    country = next(c for c in f["columns"] if c["name"] == "country")
    assert 2.3 < country["missing_pct"] < 2.7
    revenue = next(c for c in f["columns"] if c["name"] == "revenue")
    assert revenue["max"] == 99999.0 and revenue["outliers_iqr"] >= 40
    assert next(c for c in f["columns"] if c["name"] == "session_id")["kind"] == "numeric"

def test_load_csv_parses_dates():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        pd.DataFrame({"order_date": ["2024-01-01", "2024-01-02"], "name": ["a", "b"]}).to_csv(p, index=False)
        df = eda.load_csv(p)
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert not pd.api.types.is_datetime64_any_dtype(df["name"])


def test_auto_eda_edge_cases():
    assert raises(ValueError, eda.auto_eda, pd.DataFrame())
    f = eda.profile_dataframe(pd.DataFrame({"a": [np.nan], "b": ["x"]}))
    assert len(f["columns"]) == 2

def test_check_code_blocks_dangerous():
    for bad in ["import os", "result = open('f').read()", "result = df.to_csv('x.csv')",
                "result = pd.read_csv('x.csv')", "result = ().__class__.__bases__",
                "result = getattr(df, 'to_' + 'csv')", "result = eval('1+1')"]:
        assert raises(ValueError, nl2pandas.check_code, bad), bad

def test_run_code_ok_and_requires_result():
    r = nl2pandas.run_code("result = df.groupby('country')['revenue'].sum().head(2)", DF)
    assert len(r) == 2
    assert raises(ValueError, nl2pandas.run_code, "x = 1", DF)

def test_describe_df_shows_values_for_strings():
    s = nl2pandas.describe_df(DF)
    assert "'UA'" in s and "'android'" in s, "модель має бачити реальні значення колонок"

def test_ask_data_paths():
    A = nl2pandas.CodeAnswer

    with patched(nl2pandas, "ask_json", lambda *a, **k: A(can_answer=False, reason_if_not="немає колонки sentiment", code="", explanation="")):
        r = nl2pandas.ask_data(DF, "який настрій у клієнтів?")
    assert not r["ok"] and "sentiment" in r["message"]
    answers = iter([A(can_answer=True, reason_if_not="", code="result = df['nope'].sum()", explanation="x"),
                    A(can_answer=True, reason_if_not="", code="result = df['revenue'].sum()", explanation="сума")])
    with patched(nl2pandas, "ask_json", lambda *a, **k: next(answers)):
        r = nl2pandas.ask_data(DF, "сума виручки")
    assert r["ok"] and r["attempts"] == 2
    evil = A(can_answer=True, reason_if_not="", code="import os\nresult = 1", explanation="x")
    with patched(nl2pandas, "ask_json", lambda *a, **k: evil):
        r = nl2pandas.ask_data(DF, "видали все", max_repairs=1)
    assert not r["ok"] and "Імпорти" in r["message"]
    assert not nl2pandas.ask_data(pd.DataFrame(), "q")["ok"] and not nl2pandas.ask_data(DF, "  ")["ok"]

KW = dict(date_col="date", metric="converted", agg="mean",
          before=("2026-10-02", "2026-10-15"), after=("2026-10-16", "2026-10-29"),
          dims=["channel", "platform", "country"])

def test_anomaly_finds_planted_segment():
    facts = anomaly.compute_drivers(DF, **KW)
    pair = [v for v in facts.values() if v.startswith("Розріз channel × platform")]
    assert "paid_social × android" in pair[0], f"топ-перетин має бути paid_social × android, а це: {pair[0]}"
    seg_mean = pair[0].split("середнє в сегменті ")[1].split(";")[0]
    before, after = (float(x) for x in seg_mean.split(" → "))
    assert after < 0.5 * before, seg_mean

def test_contributions_sum_to_total_delta():
    for metric, agg in (("converted", "mean"), ("revenue", "sum")):
        d = DF.copy()
        if metric == "converted":
            d.loc[d.sample(frac=0.1, random_state=0).index, "converted"] = np.nan
        d = d.dropna(subset=[metric])
        b = d[(d.date >= "2026-10-02") & (d.date < "2026-10-16")]
        a = d[(d.date >= "2026-10-16") & (d.date < "2026-10-30")]
        delta = (a[metric].sum() - b[metric].sum()) if agg == "sum" else (a[metric].mean() - b[metric].mean())
        t = anomaly._segment_table(b, a, "channel", metric, agg)
        assert abs(t["contribution"].sum() - delta) < 1e-9, (metric, t["contribution"].sum(), delta)


def test_anomaly_errors():
    assert raises(ValueError, anomaly.compute_drivers, DF, **{**KW, "after": ("2030-01-01", "2030-01-07")})
    assert raises(ValueError, anomaly.compute_drivers, DF, **{**KW, "dims": ["nonexistent"]})
    assert raises(ValueError, anomaly.compute_drivers, DF, **{**KW, "agg": "median"})
    assert raises(ValueError, anomaly.compute_drivers, DF, **{**KW, "before": ("2026-10-22", "2026-10-16")})


def test_explain_change_pipeline_with_fake_llm():
    def fake(prompt, schema, system=None, **k):
        assert "F0:" in prompt and "paid_social × android" in prompt
        return anomaly.Explanation(headline="h", top_drivers="d", caveats="c", causes=[
            anomaly.Cause(hypothesis="x", confidence="medium", supporting_fact_ids=["F0"], how_to_check="y")])
    with patched(anomaly, "ask_json", fake):
        facts, expl = anomaly.explain_change(DF, **KW)
    assert expl.headline == "h" and "F0" in facts

def test_build_facts_and_hypothesis_filtering():
    facts = hypotheses.build_facts(DF, "converted", "date", exclude=["revenue", "session_id"])
    assert facts and list(facts)[0] == "F1"
    assert any("channel=email" in v for v in facts.values())
    H = hypotheses.Hypothesis
    good = H(statement="ok", rationale="r", supporting_fact_ids=["F1"], how_to_validate="v", priority=2)
    bad = H(statement="виграна", rationale="r", supporting_fact_ids=["F999"], how_to_validate="v", priority=1)
    with patched(hypotheses, "ask_json", lambda *a, **k: hypotheses.HypothesesResult(hypotheses=[bad, good])):
        _, hyps = hypotheses.suggest_hypotheses(DF, "converted", "date", exclude=["revenue", "session_id"])
    assert [h.statement for h in hyps] == ["ok"], "гіпотези з вигаданими ID мають відкидатися"

def test_olist_if_available():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "olist_clean.csv")
    if not os.path.exists(path):
        print("  (пропущено: немає data/olist_clean.csv)")
        return
    df = eda.load_csv(path)
    assert pd.api.types.is_datetime64_any_dtype(df["purchase_date"])
    f = eda.profile_dataframe(df)
    assert len(f["columns"]) == df.shape[1]
    facts = anomaly.compute_drivers(df, "purchase_date", "is_late", "mean", ("2017-10-01", "2017-10-31"),
                                    ("2017-11-01", "2017-11-30"), ["seller_state", "customer_state", "category_grouped"])
    assert "seller_state = SP" in " ".join(v for v in facts.values() if v.startswith("Розріз seller_state ="))
    hf = hypotheses.build_facts(df, "is_late", "purchase_date", exclude=["delay_days", "delivery_days"])
    assert any("customer_state=" in v for v in hf.values()), "customer_state (27 значень) має потрапляти у факти"

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"{name}")
        except Exception as e:
            failed += 1
            print(f"{name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} тестів пройдено")
    sys.exit(1 if failed else 0)
