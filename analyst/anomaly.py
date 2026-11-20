import itertools
import math
from .utils import hide_raw_contribution
import pandas as pd
from pydantic import BaseModel
from .llm import ask_json
from .verify import unsupported_numbers

MIN_N = 30       
Z_NOISE = 2.0
Z_HIGH = 5.0

def _segment_table(b, a, dim, metric, agg):
    if agg == "sum":
        sb, sa = b.groupby(dim)[metric].sum(), a.groupby(dim)[metric].sum()
        t = pd.concat([sb.rename("before"), sa.rename("after")], axis=1).fillna(0)
        t["contribution"] = t["after"] - t["before"]
    else:
        gb = b.groupby(dim)[metric].agg(["sum", "count", "std"])
        ga = a.groupby(dim)[metric].agg(["sum", "count", "std"])
        t = pd.concat([gb.add_prefix("b_"), ga.add_prefix("a_")], axis=1).fillna(0)
        t["before"] = t["b_sum"] / t["b_count"].where(t["b_count"] > 0)
        t["after"] = t["a_sum"] / t["a_count"].where(t["a_count"] > 0)
        t["share_before"] = t["b_count"] / len(b)
        t["share_after"] = t["a_count"] / len(a)
        t["contribution"] = t["a_sum"] / len(a) - t["b_sum"] / len(b)
        se = ((t["b_std"] ** 2 / t["b_count"].where(t["b_count"] > 0))
              + (t["a_std"] ** 2 / t["a_count"].where(t["a_count"] > 0))) ** 0.5
        t["z"] = (t["after"] - t["before"]) / se.where(se > 0)
    return t


def _is_binary(s: pd.Series) -> bool:
    return set(pd.unique(s.dropna())) <= {0, 1, 0.0, 1.0}


def _welch_z(x_b: pd.Series, x_a: pd.Series) -> float:
    if len(x_b) < 2 or len(x_a) < 2:
        return float("nan")
    se = math.sqrt(x_b.var() / len(x_b) + x_a.var() / len(x_a))
    return (x_a.mean() - x_b.mean()) / se if se > 0 else float("nan")


def _compute(df, date_col, metric, agg, before, after, dims, top_k=3, id_col=None):
    if agg not in ("sum", "mean"):
        raise ValueError("agg має бути 'sum' або 'mean'")
    missing = [c for c in [date_col, metric, *dims, *([id_col] if id_col else [])] if c not in df.columns]
    if missing:
        raise ValueError(f"Колонок немає в даних: {missing}")
    for name, p in (("before", before), ("after", after)):
        if pd.Timestamp(p[0]) > pd.Timestamp(p[1]):
            raise ValueError(f"Період {name}: початок пізніше за кінець")

    cols_all = [date_col] + ([id_col] if id_col else [])
    d_all = df[cols_all].copy()
    d_all[date_col] = pd.to_datetime(d_all[date_col])
    d = df.dropna(subset=[metric]).copy()
    d[date_col] = pd.to_datetime(d[date_col])
    for dim in dims:
        d[dim] = d[dim].astype("string").fillna("(missing)").astype(str)
    for x, y in itertools.combinations(dims, 2):
        d[f"{x} × {y}"] = d[x] + " × " + d[y]
    all_dims = list(dims) + [f"{x} × {y}" for x, y in itertools.combinations(dims, 2)]

    def period(p, src=None):
        src = d if src is None else src
        return src[(src[date_col] >= pd.Timestamp(p[0])) &
                   (src[date_col] < pd.Timestamp(p[1]) + pd.Timedelta(days=1))]

    b, a = period(before), period(after)
    b_all, a_all = period(before, d_all), period(after, d_all)
    if b.empty or a.empty:
        raise ValueError("Один із періодів не містить даних — перевірте дати.")

    f = (lambda x: x[metric].sum()) if agg == "sum" else (lambda x: x[metric].mean())
    total_b, total_a = f(b), f(a)
    delta = total_a - total_b
    binary = agg == "mean" and _is_binary(d[metric])

    facts, meta, i = {}, {}, 1
    pct = f"{delta / total_b:+.1%}" if total_b else "n/a"
    f0 = (f"{metric} ({agg}): {total_b:.4f} у періоді {before[0]}…{before[1]} → "
          f"{total_a:.4f} у періоді {after[0]}…{after[1]}; зміна {delta:+.4f} ({pct}).")
    if binary:
        f0 += f" У відсотках: {total_b:.2%} → {total_a:.2%}, {delta * 100:+.2f} п.п."
    if id_col:
        f0 += (f" Унікальних {id_col} (усі в періоді): {b_all[id_col].nunique()} → {a_all[id_col].nunique()}; "
               f"з відомим {metric}: {b[id_col].nunique()} → {a[id_col].nunique()}; "
               f"позицій (рядків) з відомим {metric}: {len(b)} → {len(a)}.")
    else:
        f0 += f" Рядків з відомим {metric}: {len(b)} → {len(a)}."
    if agg == "mean":
        z_total = _welch_z(b[metric], a[metric])
        if not math.isnan(z_total):
            verdict = "в межах випадкового шуму" if abs(z_total) < Z_NOISE else "статистично значуща"
            f0 += f" Загальна зміна: z={z_total:+.1f} ({verdict})."
        meta["F0"] = {"z": z_total, "excess_share": float("nan")}
    facts["F0"] = f0

    candidates = []   # (|excess|, dim, seg, row) для explain-away
    for dim in all_dims:
        t = _segment_table(b, a, dim, metric, agg)
        if agg == "mean":
            t["excess"] = t["contribution"] - t["share_after"] * delta
            ok = (t["b_count"] >= MIN_N) & (t["a_count"] >= MIN_N)
        else:
            expected = t["before"] * (total_a / total_b - 1) if total_b else 0
            t["excess"] = t["contribution"] - expected
            ok = pd.Series(True, index=t.index)

        by_contrib = t["contribution"].abs().sort_values(ascending=False).head(top_k).index.tolist()
        by_excess = t.loc[ok, "excess"].abs().sort_values(ascending=False).head(top_k).index.tolist()
        chosen = by_contrib + [s for s in by_excess if s not in by_contrib]

        for seg in chosen:
            r = t.loc[seg]
            share = r["contribution"] / delta if delta else float("nan")
            ex_share = r["excess"] / delta if delta else float("nan")
            txt = (f"Розріз {dim} = {seg}: внесок {r['contribution']:+.4f} "
                   f"({share:+.0%} від загальної зміни); "
                   f"надлишковий внесок понад пропорційну частку {r['excess']:+.4f} ({ex_share:+.0%})")
            z = float("nan")
            if agg == "mean":
                z = r["z"]
                txt += (f"; середнє в сегменті {r['before']:.4f} → {r['after']:.4f}; "
                        f"частка трафіку {r['share_before']:.1%} → {r['share_after']:.1%}")
                if not math.isnan(z):
                    txt += f"; z={z:+.1f}" + (" (шум)" if abs(z) < Z_NOISE else "")
            facts[f"F{i}"] = txt
            meta[f"F{i}"] = {"z": z, "excess_share": ex_share}
            i += 1
        if agg == "mean":
            zs = t.loc[ok, "z"].abs().dropna()
            if not zs.empty:
                candidates.append((zs.max(), dim, zs.idxmax()))
        elif by_excess:
            candidates.append((abs(t.loc[by_excess[0], "excess"]), dim, by_excess[0]))

    if candidates:
        _, dim, seg = max(candidates, key=lambda c: c[0])
        b_rest, a_rest = b[b[dim] != seg], a[a[dim] != seg]
        if not b_rest.empty and not a_rest.empty:
            d_rest = f(a_rest) - f(b_rest)
            left = d_rest / delta if delta else float("nan")
            facts[f"F{i}"] = (f"Без сегмента {dim} = {seg} загальна зміна {metric} становить {d_rest:+.4f} "
                              f"замість {delta:+.4f} (лишається {left:.0%} від початкової зміни)")
            i += 1

    def daily(x):
        g = x.groupby(x[date_col].dt.normalize())[metric]
        return (g.agg(["mean", "count"]) if agg == "mean" else g.agg(["sum", "count"])).set_axis(["v", "n"], axis=1)

    db, da = daily(b), daily(a)
    if agg == "mean":
        da, db = da[da["n"] >= MIN_N], db[db["n"] >= MIN_N]
    if not da.empty and not db.empty:
        peaks = da["v"].sort_values(ascending=False, key=lambda s: (s - db["v"].median()).abs()).head(3)
        peaks_txt = ", ".join(f"{day:%Y-%m-%d}: {v:.4f} (n={int(da.loc[day, 'n'])})" for day, v in peaks.items())
        facts[f"F{i}"] = (f"Динаміка по днях: медіана денного значення {db['v'].median():.4f} (before) → "
                          f"{da['v'].median():.4f} (after); дні періоду after з найбільшим відхиленням: {peaks_txt}")
    return facts, meta


def compute_drivers(df, date_col, metric, agg, before, after, dims, top_k=3, id_col=None):
    return _compute(df, date_col, metric, agg, before, after, dims, top_k, id_col)[0]


class Cause(BaseModel):
    hypothesis: str
    confidence: str
    supporting_fact_ids: list[str]
    how_to_check: str


class Explanation(BaseModel):
    headline: str
    top_drivers: str
    causes: list[Cause]
    caveats: str


ANOM_SYSTEM = """Ти — аналітик, який розбирає причини зміни метрики.
Тобі дають F0 (загальна зміна) та F1..Fn — розраховані внески сегментів, explain-away і динаміку по днях.
Правила:
1. Спершу назви сегменти з найбільшим НАДЛИШКОВИМ внеском (понад їхню пропорційну частку). Великий
   сегмент (напр. 70% трафіку) майже завжди має великий «внесок» — це ще не аномалія. Аномалія — це
   сегмент, де середнє змінилось значно сильніше, ніж загалом (порівнюй середнє «до → після» з F0).
2. Якщо F0 позначено як «в межах випадкового шуму» або z сегмента позначено «(шум)» — НЕ називай це
   причиною; прямо напиши, що зміна може бути випадковою.
3. Використай факт explain-away: якщо після виключення сегмента зміна майже зникає — сегмент і є причиною;
   якщо лишається більша частина — зміна системна (охоплює всі сегменти).
4. Враховуй динаміку по днях: чи зміна концентрується в окремих днях (інцидент/акція), чи рівномірна.
5. Розрізняй зміну всередині сегмента та зміну його частки трафіку (mix shift).
6. Розрізняй унікальні сутності (замовлення) та рядки (позиції): «замовленнями» називай лише величину
   «Унікальних … (усі в періоді)». Числа «з відомим …» та «позицій (рядків)» замовленнями не називай.
7. Формулюй причини як ГІПОТЕЗИ з посиланням на ID фактів. Не вигадуй чисел і не використовуй
   зовнішніх знань (географія, свята, сезонність), яких немає у фактах. Поле confidence познач як
   «low» — його розрахує код.
8. Для відсоткових метрик (частка, конверсія) пиши у відсотках і п.п. (5.1% → 13.8%, +8.8 п.п.),
   беручи числа з F0.
9. how_to_check — лише розрізи/колонки з фактів; не згадуй даних, яких немає (напр. «перевізники»).
10. У caveats вкажи, чого з цих даних встановити не можна.
11. Відповідай українською, без русизмів і друкарських помилок."""


def _confidence(ids, meta):
    zs = [abs(meta[i]["z"]) for i in ids if i in meta and not math.isnan(meta[i]["z"])]
    if not zs:
        return None
    zmax = max(zs)
    return "high" if zmax >= Z_HIGH else "medium" if zmax >= Z_NOISE else "low"


def explain_change(df, date_col, metric, agg, before, after, dims, id_col=None):
    facts, meta = _compute(df, date_col, metric, agg, before, after, dims, id_col=id_col)
    facts_text = "\n".join(f"{k}: {hide_raw_contribution(v)}" for k, v in facts.items())
    expl = ask_json(facts_text, Explanation, system=ANOM_SYSTEM)

    valid = []
    for c in expl.causes:
        if c.supporting_fact_ids and all(x in facts for x in c.supporting_fact_ids):
            c.confidence = _confidence(c.supporting_fact_ids, meta) or "n/a (немає z-тесту)"
            valid.append(c)
        else:
            print(f"Відкинуто причину з неіснуючими ID: {c.hypothesis[:60]}...")
    expl.causes = valid

    bad = unsupported_numbers(expl.model_dump_json(), facts_text)
    if bad:
        print(f"Непідтверджені числа: {bad}")
    return facts, expl