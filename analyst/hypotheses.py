import pandas as pd
from pydantic import BaseModel
from .llm import ask_json
from .verify import unsupported_numbers

def _quartile_groups(s: pd.Series):
    codes = pd.qcut(s, 4, labels=False, duplicates="drop")
    b = s.groupby(codes).agg(["min", "max"])
    mapping = {float(k): f"{r['min']:.4g}…{r['max']:.4g}" for k, r in b.iterrows()}
    return codes.map(mapping)


def build_facts(df, metric, date_col=None, exclude=None, max_unique=30, min_count=100, top_n=25, per_col=2):
    exclude = set(exclude or [])
    work = df.dropna(subset=[metric])
    overall = work[metric].mean()
    rows = []
    for col in work.columns:
        if col in (metric, date_col) or col in exclude:
            continue
        s = work[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        n_unique = s.nunique(dropna=True)
        if n_unique < 2:
            continue
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if is_num and n_unique > max_unique:
            try:
                groups = _quartile_groups(s)
            except ValueError:
                continue
            label = f"{col} (квартиль)"
        elif n_unique > max_unique:
            continue
        else:
            groups, label = s, col
        g = work.groupby(groups, observed=True, dropna=False)[metric].agg(["mean", "count", "std"])
        col_rows = []
        for seg, r in g.iterrows():
            if r["count"] < max(min_count, 0.005 * len(work)):
                continue
            seg_label = "(пропуск)" if pd.isna(seg) else seg
            lift = r["mean"] / overall - 1 if overall else 0
            se = (r["std"] / r["count"] ** 0.5) if r["std"] and r["count"] > 1 else float("nan")
            z = (r["mean"] - overall) / se if se == se and se > 0 else float("nan")
            z_txt = f", z={z:+.1f}" if z == z else ""
            col_rows.append({
                "score": abs(lift) * (r["count"] ** 0.5),
                "text": (f"{label}={seg_label}: середнє {metric} = {r['mean']:.4f} (n={int(r['count'])}), "
                         f"{lift:+.0%} відносно загального {overall:.4f}{z_txt}"),
            })
        # не більше per_col найсильніших сегментів на колонку: щоб гіпотези не зводились до однієї колонки
        rows += sorted(col_rows, key=lambda x: x["score"], reverse=True)[:per_col]
    rows.sort(key=lambda x: x["score"], reverse=True)
    facts = [r["text"] for r in rows[:top_n]]

    if date_col:
        d = work.assign(_d=pd.to_datetime(work[date_col]))
        weekly = d.set_index("_d")[metric].resample("W").mean().dropna()
        if len(weekly) >= 3:
            facts.append(f"Тижневий тренд {metric}: перший тиждень {weekly.iloc[0]:.4f}, "
                         f"передостанній {weekly.iloc[-2]:.4f}, останній {weekly.iloc[-1]:.4f}")
        dow = d.groupby(d["_d"].dt.day_name())[metric].agg(["mean", "count", "std"])
        hi, lo = dow["mean"].idxmax(), dow["mean"].idxmin()
        se = (dow.loc[hi, "std"] ** 2 / dow.loc[hi, "count"] + dow.loc[lo, "std"] ** 2 / dow.loc[lo, "count"]) ** 0.5
        diff = dow.loc[hi, "mean"] - dow.loc[lo, "mean"]
        z = diff / se if se and se == se else float("nan")
        note = " (в межах шуму)" if z == z and abs(z) < 2 else ""
        facts.append(f"За днем тижня: найвищий {hi} ({dow.loc[hi, 'mean']:.4f}, n={int(dow.loc[hi, 'count'])}), "
                     f"найнижчий {lo} ({dow.loc[lo, 'mean']:.4f}, n={int(dow.loc[lo, 'count'])}); "
                     f"різниця {diff:+.4f}" + (f", z={z:.1f}{note}" if z == z else ""))
    return {f"F{i + 1}": t for i, t in enumerate(facts)}


class Hypothesis(BaseModel):
    statement: str
    rationale: str
    supporting_fact_ids: list[str]
    how_to_validate: str
    priority: int  # 1 = найвищий


class HypothesesResult(BaseModel):
    hypotheses: list[Hypothesis]


HYP_SYSTEM = """Ти — продуктовий аналітик. Тобі надають пронумеровані ФАКТИ (F1, F2...) та список ДОСТУПНИХ КОЛОНОК.
Запропонуй 5–10 гіпотез, які варто перевірити.
Правила:
1. Кожна гіпотеза ОБОВ'ЯЗКОВО спирається на 1–3 факти: вкажи їх ID у supporting_fact_ids.
2. Не вигадуй факти й числа, яких немає у списку. Не пиши загальних фраз типу «варто покращити UX».
3. Розрізняй кореляцію та причинність: формулюй як гіпотезу, а не як встановлений факт.
4. how_to_validate — конкретний крок, який можна виконати ЛИШЕ з ДОСТУПНИХ КОЛОНОК (не згадуй даних,
   яких немає: перевізники, погода, свята, склади тощо).
5. Не використовуй зовнішніх знань (географія штатів/країн, «віддалені регіони», «сезонність»), якщо
   цього немає у фактах. Слово «сезонність» заборонене: у даних лише кілька місяців, це може бути інцидент.
6. Гіпотези мають стосуватися РІЗНИХ колонок/розрізів і не дублювати одна одну. Якщо один фактор
   (напр. відстань) пояснює кілька фактів — зроби одну гіпотезу, а інші зміни про інші колонки.
7. Якщо факт має z у межах ±2 або позначку «(в межах шуму)» — не будуй на ньому гіпотезу або познач
   прямо, що ефект малий і може бути випадковим.
8. Відповідай українською."""


def suggest_hypotheses(df, metric, date_col=None, exclude=None):
    facts = build_facts(df, metric, date_col, exclude=exclude)
    if not facts:
        raise ValueError("Не знайдено сегментів для аналізу — перевірте metric та дані.")
    facts_text = "\n".join(f"{k}: {v}" for k, v in facts.items())
    cols = [c for c in df.columns if c != metric and c not in set(exclude or [])]
    result = ask_json(f"МЕТРИКА: {metric}\n\nДОСТУПНІ КОЛОНКИ: {', '.join(cols)}\n\nФАКТИ:\n{facts_text}",
                      HypothesesResult, system=HYP_SYSTEM)

    valid = []
    for h in result.hypotheses:
        if h.supporting_fact_ids and all(i in facts for i in h.supporting_fact_ids):
            valid.append(h)
        else:
            print(f"Відкинуто гіпотезу з неіснуючими ID: {h.statement[:60]}...")
    bad = unsupported_numbers(" ".join(h.statement + h.rationale for h in valid), facts_text)
    if bad:
        print(f"Числа без підтвердження у фактах: {bad}")
    return facts, sorted(valid, key=lambda h: h.priority)