def show_eda(facts: dict, report) -> None:
    print(f"Датасет: {facts['rows']:,} рядків × {facts['columns_count']} колонок; "
          f"повних дублікатів: {facts['duplicate_rows']:,}")
    print(f"Оцінка якості даних: {report.quality_score}/10")
    print("\nКолонки:")
    print(f"  {'колонка':30}{'тип':13}{'пропуски %':>11}{'унік.':>9}  розподіл")
    for c in facts["columns"]:
        if c["kind"] == "numeric" and "median" in c:
            dist = f"min={c['min']}, медіана={c['median']}, max={c['max']}"
        elif c["kind"] == "datetime":
            dist = f"{c['min'][:10]} … {c['max'][:10]}"
        elif c["kind"] == "categorical":
            dist = ("ідентифікатор" if c.get("looks_like_id")
                    else ", ".join(f"{k} {v}%" for k, v in list(c["top_values_pct"].items())[:3]))
        else:
            dist = c["kind"]
        print(f"  {c['name'][:29]:30}{c['dtype'][:12]:13}{c['missing_pct']:>11}{c['n_unique']:>9}  {dist}")
    for k, v in facts.get("quality_breakdown", {}).items():
        print(f"   · {k}: {v}")
    print(f"\n{report.summary}\n")
    print("Ключові висновки:")
    for f in report.key_findings:
        print(f"  • {f}")
    order = {"high": 0, "medium": 1, "low": 2}
    print("\nПроблеми:")
    for i in sorted(report.issues, key=lambda x: order.get(x.severity.lower(), 3)):
        print(f"  [{i.severity.upper()}] {i.column}: {i.problem}\n      Доказ: {i.evidence}\n"
              f"      Рекомендація: {i.recommendation}")


def show_hypotheses(facts: dict, hypotheses) -> None:
    for n, h in enumerate(hypotheses, 1):
        print(f"\n{n}. [пріоритет {h.priority}] {h.statement}\n   Обґрунтування: {h.rationale}")
        for fid in h.supporting_fact_ids:
            print(f"   {fid}: {facts[fid]}")
        print(f"   Як перевірити: {h.how_to_validate}")


def show_explanation(facts: dict, expl) -> None:
    print(f"{expl.headline}\n\n{facts['F0']}\n\nГоловні драйвери: {expl.top_drivers}\n")
    for n, c in enumerate(expl.causes, 1):
        print(f"{n}. [впевненість (за z-тестом): {c.confidence}] {c.hypothesis}")
        for fid in c.supporting_fact_ids:
            if fid in facts:
                print(f"   {fid}: {facts[fid]}")
        print(f"   Як перевірити: {c.how_to_check}")
    extra = [k for k, v in facts.items() if v.startswith(("Без сегмента", "Динаміка по днях"))]
    if extra:
        print("\nДодаткові перевірки:")
        for k in extra:
            print(f"   {k}: {facts[k]}")
    print(f"\nОбмеження: {expl.caveats}")