import json
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from .llm import ask_json
from .utils import looks_like_id
from .verify import unsupported_numbers

DATE_HINTS = ("date", "time", "timestamp")
SEV_ORDER = {"low": 0, "medium": 1, "high": 2}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if any(k in col.lower() for k in DATE_HINTS):
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().sum() >= 0.9 * s.notna().sum():  # парсимо, лише якщо це справді дати
                df[col] = parsed
    return df

def profile_dataframe(df: pd.DataFrame) -> dict:
    facts = {
        "rows": int(len(df)),
        "columns_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": [],
    }
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        info = {
            "name": col,
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": n_unique,
        }
        if pd.api.types.is_bool_dtype(s):
            info["kind"] = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            info["kind"] = "numeric"
            clean = s.dropna()
            is_coord = any(k in col.lower() for k in ("lat", "lng", "lon"))
            is_counter = len(clean) > 0 and (clean % 1 == 0).all() and n_unique <= 30
            if len(clean):
                info.update(min=round(float(clean.min()), 4), median=round(float(clean.median()), 4),
                            mean=round(float(clean.mean()), 4), max=round(float(clean.max()), 4),
                            std=round(float(clean.std()), 4) if len(clean) > 1 else 0.0)
                if is_coord or is_counter:
                    info["note"] = "координата/лічильник: IQR-викиди не застосовні"
                else:
                    q1, q3 = clean.quantile([0.25, 0.75]).tolist()
                    iqr = q3 - q1
                    n_out = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
                    info.update(outliers_iqr=n_out, outliers_pct=round(n_out / len(clean) * 100, 2))
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["kind"] = "datetime"
            info.update(min=str(s.min()), max=str(s.max()))
        else:
            info["kind"] = "categorical"
            info["looks_like_id"] = looks_like_id(col, s)
            top = s.value_counts(normalize=True).head(5)
            info["top_values_pct"] = {str(k): round(float(v * 100), 2) for k, v in top.items()}
        facts["columns"].append(info) # (!) без цього LLM не отримала б жодної колонки

    groups = {}
    for c in facts["columns"]:
        if c["missing_pct"] > 0:
            groups.setdefault(c["missing_pct"], []).append(c["name"])
    facts["same_missing_groups"] = [{"missing_pct": k, "columns": v} for k, v in groups.items() if len(v) > 1]
    return facts

def quality_score(facts: dict) -> tuple[int, dict]:
    n = max(facts["rows"], 1)
    cols = [c for c in facts["columns"] if not c.get("looks_like_id")]
    dup_pct = facts["duplicate_rows"] / n * 100
    dup_pen = min(3.0, dup_pct)
    share_miss = sum(c["missing_pct"] > 5 for c in cols) / max(len(cols), 1)
    miss_pen = min(3.0, share_miss * 6)
    num = [c for c in cols if c.get("kind") == "numeric" and "outliers_pct" in c]
    share_out = sum(c["outliers_pct"] > 5 for c in num) / max(len(num), 1)
    out_pen = min(2.0, share_out * 4)
    score = max(1, min(10, round(10 - dup_pen - miss_pen - out_pen)))
    return score, {"дублікати": f"-{dup_pen:.1f} ({dup_pct:.2f}% рядків)",
                   "пропуски": f"-{miss_pen:.1f} ({share_miss:.0%} колонок мають >5% пропусків)",
                   "викиди": f"-{out_pen:.1f} ({share_out:.0%} числових колонок мають >5% викидів за IQR)"}

class Issue(BaseModel):
    severity: str = Field(description="high, medium або low")
    column: str
    problem: str
    evidence: str = Field(description="Конкретне число з наданих даних")
    recommendation: str


class EDAReport(BaseModel):
    quality_score: int = Field(description="Завжди став 0: оцінку розраховує код")
    summary: str
    key_findings: list[str]
    issues: list[Issue]


EDA_SYSTEM = """Ти — старший аналітик даних. Тобі надають JSON з УЖЕ ПОРАХОВАНИМИ статистиками датасету.
Правила:
1. Використовуй ТІЛЬКИ числа з наданого JSON. Нічого не рахуй самостійно і не вигадуй.
2. Кожна проблема має містити назву колонки та конкретне число в полі evidence.
3. Якщо даних недостатньо для висновку — прямо скажи про це.
4. Не вигадуй бізнес-контекст, якого немає в даних; якщо припускаєш — познач як припущення.
5. Колонки з looks_like_id=true — це ідентифікатори: не аналізуй їхні розподіли.
6. quality_score став 0 (його розрахує код). Не оцінюй «загальну якість» словами типу «добра/погана» без чисел.
7. Обов'язково розглянь: (а) колонки з outliers_pct > 5 — вкажи max і median, чи це схоже на помилку даних
   чи на природний «важкий хвіст»; (б) same_missing_groups — опиши ОДНОЮ проблемою, бо причина спільна;
   (в) пропуски >5%, дублікати, константні колонки (n_unique ≤ 1).
8. У evidence пиши число з одиницями, як у JSON (напр. «1.68% пропусків», «max=6735 при median=75»).
9. severity: high — пропуски >20%, дублікати >1% або max > 10× median; medium — 5–20% / викиди >5%; low — решта.
10. Відповідай українською, стисло: 3–6 key_findings."""

def rule_severity(info: dict) -> str:
    sev = 0
    miss = info["missing_pct"]
    sev = max(sev, 2 if miss > 20 else 1 if miss > 5 else 0)
    if info.get("outliers_pct", 0) > 5:
        sev = max(sev, 1)
    if info.get("median") and info.get("max") is not None and info["median"] > 0 \
            and info["max"] > 10 * info["median"]:
        sev = max(sev, 2)
    return ["low", "medium", "high"][sev]

def enforce_severity(report, facts):
    by_col = {c["name"]: c for c in facts["columns"]}
    for issue in report.issues:
        if issue.column in by_col:
            issue.severity = rule_severity(by_col[issue.column])

def auto_eda(df: pd.DataFrame) -> tuple[dict, EDAReport]:
    if df.empty:
        raise ValueError("Датафрейм порожній — нічого аналізувати.")
    facts = profile_dataframe(df)
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    report = ask_json(f"СТАТИСТИКИ ДАТАСЕТУ:\n{facts_json}", EDAReport, system=EDA_SYSTEM)
    bad = unsupported_numbers(report.model_dump_json(), facts_json)
    if bad:
        print(f"Числа у відповіді LLM, яких немає у фактах: {bad}")
    enforce_severity(report, facts)
    report.quality_score, facts["quality_breakdown"] = quality_score(facts)
    return facts, report

def plot_distributions(df: pd.DataFrame, max_cols: int = 6, clip_q: float = 0.99):
    import matplotlib.pyplot as plt
    num = [c for c in df.select_dtypes("number").columns if df[c].nunique() > 2][:max_cols]
    if not num:
        print("Немає числових колонок для гістограм.")
        return
    fig, axes = plt.subplots(1, len(num), figsize=(4 * len(num), 3))
    for ax, c in zip(np.atleast_1d(axes), num):
        s = df[c].dropna()
        shown = s[s <= s.quantile(clip_q)]
        shown.hist(ax=ax, bins=40)
        ax.set_title(f"{c}\n(до {clip_q:.0%} перцентиля; приховано {len(s) - len(shown):,})", fontsize=9)
    plt.tight_layout()
    plt.show()
