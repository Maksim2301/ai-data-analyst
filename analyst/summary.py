import json
import pandas as pd
from pydantic import BaseModel
from .llm import ask_json
from .verify import unsupported_numbers
from .utils import hide_raw_contribution

class ExecSummary(BaseModel):
    headline: str
    what_happened: str
    why_it_matters: str
    recommended_actions: list[str]
    caveats: str


SUM_SYSTEM = """Ти пишеш executive summary для нетехнічного керівника (CEO/продакт).
Правила:
1. Максимум ~150 слів; без жаргону; спершу висновок, потім деталі.
2. Використовуй ЛИШЕ числа з наданих метрик. Для часток пиши у відсотках і п.п. (5.1% → 13.8%, +8.8 п.п.),
   а не як 0.0506.
3. Розрізняй «замовлення» та «позиції/рядки»: слово «замовлень» лише для метрик, підписаних як
   «Унікальних замовлень». Не плутай їх.
4. Не стверджуй того, чого немає в метриках (скасування, лояльність, відтік, фінансові збитки). Якщо
   пишеш про наслідки — познач як припущення («ймовірно», «може»).
5. Якщо в метриках сказано, що зміна «в межах випадкового шуму» — прямо напиши, що висновків робити не слід.
6. Якщо зміна системна (охоплює всі сегменти, після виключення топ-сегмента майже не зменшується), не
   зводь її до одного сегмента.
7. recommended_actions — 2–3 конкретні дії, кожна прив'язана до сегмента чи метрики з даних.
8. У caveats вкажи обмеження даних і що причинність не доведена.
9. Відповідай українською, грамотно, без русизмів і друкарських помилок.
10. Не називай сегмент причиною лише за розміром. Спирайся на надлишковий внесок і на зміну середнього
    в сегменті (до → після). Якщо без сегмента більша частина зміни лишається — пиши, що зміна системна."""


def business_metrics(df, date_col, before, after, id_col=None, revenue_col=None, score_col=None) -> str:
    """Бізнес-метрики по двох періодах (рахує Pandas) — додаються до фактів для summary."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    def per(p):
        return d[(d[date_col] >= pd.Timestamp(p[0])) & (d[date_col] < pd.Timestamp(p[1]) + pd.Timedelta(days=1))]

    b, a = per(before), per(after)

    def line(name, vb, va, fmt="{:.0f}"):
        ch = f"{va / vb - 1:+.1%}" if vb else "n/a"
        return f"{name}: {fmt.format(vb)} → {fmt.format(va)} ({ch})"

    lines = []
    if id_col:
        lines.append(line("Унікальних замовлень", b[id_col].nunique(), a[id_col].nunique()))
    if revenue_col:
        lines.append(line("Виручка", b[revenue_col].sum(), a[revenue_col].sum()))
    if score_col:
        lines.append(line("Середня оцінка клієнтів", b[score_col].mean(), a[score_col].mean(), "{:.3f}"))
    return "\n".join(lines)


def executive_summary(metrics, audience: str = "CEO", extra: str = "") -> ExecSummary:
    if isinstance(metrics, str):
        text = hide_raw_contribution(metrics)
    else:
        text = json.dumps({k: hide_raw_contribution(v) if isinstance(v, str) else v
                           for k, v in metrics.items()}, ensure_ascii=False, indent=2)
    if extra:
        text += f"\n\nДОДАТКОВІ БІЗНЕС-МЕТРИКИ:\n{extra}"
    s = ask_json(f"АУДИТОРІЯ: {audience}\n\nМЕТРИКИ:\n{text}", ExecSummary, system=SUM_SYSTEM)
    bad = unsupported_numbers(s.model_dump_json(), text)
    if bad:
        print(f"Непідтверджені числа: {bad}")
    return s


def to_markdown(s: ExecSummary) -> str:
    acts = "\n".join(f"- {a}" for a in s.recommended_actions)
    return (f"## {s.headline}\n\n**Що сталося:** {s.what_happened}\n\n"
            f"**Чому це важливо:** {s.why_it_matters}\n\n**Що робити:**\n{acts}\n\n"
            f"*Обмеження:* {s.caveats}")