import argparse
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

ORDER_DATES = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]
FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

EXTRA_TRANSLATION = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}
BRAZIL_LAT, BRAZIL_LNG = (-35, 6), (-75, -33)

def load_tables(raw: Path) -> dict:
    missing = [f for f in FILES.values() if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(f"У папці {raw} не знайдено файлів: {missing}")
    t = {k: pd.read_csv(raw / f) for k, f in FILES.items() if k not in ("orders", "items")}
    t["orders"] = pd.read_csv(raw / FILES["orders"], parse_dates=ORDER_DATES)
    t["items"] = pd.read_csv(raw / FILES["items"], parse_dates=["shipping_limit_date"])
    return t


def tables_overview(t: dict):
    print(f"\n{'=' * 24} ВИХІДНІ ТАБЛИЦІ {'=' * 24}")
    rows = [{"таблиця": k, "рядків": len(v), "колонок": v.shape[1], "колонки": ", ".join(v.columns)}
            for k, v in t.items()]
    print(pd.DataFrame(rows).to_string(index=False, max_colwidth=90))

def integrity_report(t: dict):
    o, it, pay = t["orders"], t["items"], t["payments"]
    print(f"\n{'=' * 24} ПЕРЕВІРКИ ЦІЛІСНОСТІ {'=' * 24}")

    no_items = o[~o["order_id"].isin(it["order_id"])]
    print(f"• Замовлень без позицій: {len(no_items):,} (за статусом: {no_items['order_status'].value_counts().to_dict()})"
          " → випадуть з аналізу виручки")
    no_pay = it.loc[~it["order_id"].isin(pay["order_id"]), "order_id"].nunique()
    n_rows = int((~it["order_id"].isin(pay["order_id"])).sum())
    print(f"• Замовлень із позиціями, але без платежу: {no_pay} ({n_rows} рядків-позицій) "
          "→ inner join з payments їх втрачає, тому тут використовується left join")
    multi_rev = int((t["reviews"].groupby("order_id").size() > 1).sum())
    print(f"• Замовлень із кількома відгуками: {multi_rev:,} → усереднюємо review_score")
    print(f"• payment_type='not_defined': {int((pay['payment_type'] == 'not_defined').sum())} → 'unknown'")
    carrier_bad = int((o["order_delivered_carrier_date"] < o["order_purchase_timestamp"]).sum())
    print(f"• Передано перевізнику РАНІШЕ за покупку: {carrier_bad} замовлень → дата анулюється")
    deliv_no_date = int(((o["order_status"] == "delivered") & o["order_delivered_customer_date"].isna()).sum())
    print(f"• Статус delivered, але без дати доставки: {deliv_no_date} → is_late = NaN")
    cats = set(t["products"]["product_category_name"].dropna()) - set(t["translation"]["product_category_name"])
    print(f"• Категорії без перекладу: {sorted(cats)} → додано вручну")
    geo = t["geolocation"]
    outside = int((~(geo["geolocation_lat"].between(*BRAZIL_LAT) & geo["geolocation_lng"].between(*BRAZIL_LNG))).sum())
    print(f"• Геоточок поза Бразилією: {outside} з {len(geo):,} → відкидаємо; "
          f"{geo['geolocation_zip_code_prefix'].nunique():,} індексів → медіана координат на індекс")

def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = (np.radians(x) for x in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def build_dataset(t: dict) -> pd.DataFrame:
    geo = t["geolocation"]
    geo = geo[geo["geolocation_lat"].between(*BRAZIL_LAT) & geo["geolocation_lng"].between(*BRAZIL_LNG)]
    geo = geo.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))

    def with_geo(df, zip_col, prefix):
        g = geo.rename(columns={"geolocation_zip_code_prefix": zip_col,
                                "lat": f"{prefix}_lat", "lng": f"{prefix}_lng"})
        return df.merge(g, on=zip_col, how="left")

    cust = with_geo(t["customers"], "customer_zip_code_prefix", "customer")
    sell = with_geo(t["sellers"], "seller_zip_code_prefix", "seller")[
        ["seller_id", "seller_state", "seller_lat", "seller_lng"]]

    pay = (t["payments"].sort_values("payment_value", ascending=False)
           .groupby("order_id", as_index=False)
           .agg(payment_type=("payment_type", "first"),
                payment_installments=("payment_installments", "max")))

    rv = t["reviews"].assign(has_comment=t["reviews"]["review_comment_message"].notna().astype(int))
    rev = rv.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"), has_review_comment=("has_comment", "max"))

    trans = pd.concat([t["translation"], pd.DataFrame(
        {"product_category_name": list(EXTRA_TRANSLATION), "product_category_name_english": list(EXTRA_TRANSLATION.values())})
    ], ignore_index=True)
    prod = t["products"].copy()
    prod["product_volume_cm3"] = prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    prod = prod[["product_id", "product_category_name", "product_weight_g",
                 "product_photos_qty", "product_volume_cm3"]].merge(trans, on="product_category_name", how="left")

    n_items = len(t["items"])
    df = (t["orders"].merge(t["items"], on="order_id", how="inner")
          .merge(cust, on="customer_id", how="left")
          .merge(sell, on="seller_id", how="left")
          .merge(pay, on="order_id", how="left")
          .merge(rev, on="order_id", how="left")
          .merge(prod, on="product_id", how="left"))
    assert len(df) == n_items, f"Merge змінив кількість рядків: {n_items} → {len(df)}. Перевірте дублікати ключів."

    df["distance_km"] = haversine_km(df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"])
    return df

def clean(df: pd.DataFrame, date_from: str, date_to: str, drop_bad_status: bool = True):
    steps = []

    def log(msg, before, after):
        steps.append(f"{msg}: {before:,} → {after:,} рядків")

    n = len(df)
    df = df.drop_duplicates(subset=["order_id", "order_item_id"])
    log("Видалення дублікатів (order_id + order_item_id)", n, len(df))

    if drop_bad_status:
        n = len(df)
        df = df[~df["order_status"].isin(["canceled", "unavailable"])]
        log("Виключення canceled/unavailable", n, len(df))

    n = len(df)
    lo, hi = pd.Timestamp(date_from), pd.Timestamp(date_to) + pd.Timedelta(days=1)
    df = df[(df["order_purchase_timestamp"] >= lo) & (df["order_purchase_timestamp"] < hi)].copy()
    log(f"Фільтр періоду {date_from} … {date_to}", n, len(df))

    ts = df["order_purchase_timestamp"]

    bad = df["order_delivered_carrier_date"] < ts
    steps.append(f"order_delivered_carrier_date раніше за покупку анульовано: {int(bad.sum()):,} рядків")
    df.loc[bad, "order_delivered_carrier_date"] = pd.NaT

    df["customer_city"] = df["customer_city"].str.strip().str.title()
    df["customer_state"] = df["customer_state"].str.strip().str.upper()
    df["seller_state"] = df["seller_state"].str.strip().str.upper()
    df["category"] = df["product_category_name_english"].fillna(df["product_category_name"]).fillna("unknown")
    top = df["category"].value_counts().nlargest(15).index
    df["category_grouped"] = df["category"].where(df["category"].isin(top), "other")
    df["payment_type"] = df["payment_type"].replace("not_defined", "unknown").fillna("unknown")
    df["purchase_date"] = ts.dt.normalize()
    df["purchase_month"] = ts.dt.to_period("M").astype(str)
    df["purchase_weekday"] = ts.dt.day_name()
    df["purchase_hour"] = ts.dt.hour
    df["revenue"] = df["price"] + df["freight_value"]
    df["freight_share"] = df["freight_value"] / df["revenue"]
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["delivery_days"] = (df["order_delivered_customer_date"] - ts).dt.total_seconds() / 86400
    df["delay_days"] = (df["order_delivered_customer_date"]
                        - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    neg = df["delivery_days"] < 0
    steps.append(f"Некоректний delivery_days (<0) занулено: {int(neg.sum()):,} рядків")
    df.loc[neg, ["delivery_days", "delay_days"]] = np.nan
    df["is_late"] = df["delay_days"].gt(0).astype(float).where(df["delay_days"].notna())
    q1, q3 = df["price"].quantile([0.25, 0.75])
    n_out = int((df["price"] > q3 + 1.5 * (q3 - q1)).sum())
    steps.append(f"Цінових викидів (IQR) залишено без змін: {n_out:,} ({n_out / len(df):.1%}), макс. price = {df['price'].max():,.2f}")
    df = df.drop(columns=["customer_zip_code_prefix", "product_category_name",
                          "product_category_name_english", "order_approved_at"])
    return df.reset_index(drop=True), steps

def preview(df: pd.DataFrame, title: str, n: int = 3):
    print(f"\n{'=' * 24} {title} {'=' * 24}")
    print(f"Розмір: {df.shape[0]:,} рядків × {df.shape[1]} колонок")
    print(f"\nПерші {n} рядки (транспоновано — кожна колонка в окремому рядку):")
    print(df.head(n).T.to_string())
    print(f"\nВипадкові {n} рядки:")
    print(df.sample(n, random_state=42).T.to_string())
    print("\nТипи колонок:")
    print(df.dtypes.astype(str).to_string())
    miss = df.isna().mean().mul(100).round(2)
    print("\nПропуски, %:")
    print(miss[miss > 0].sort_values(ascending=False).to_string() or "немає")
    print(f"\nПовних дублікатів рядків: {int(df.duplicated().sum()):,}")

def monthly_orders(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["order_id"].nunique()

def monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    k = df.groupby("purchase_month").agg(
        orders=("order_id", "nunique"), revenue=("revenue", "sum"),
        avg_review=("review_score", "mean"), late_rate=("is_late", "mean"))
    k["revenue_mom_%"] = (k["revenue"].pct_change() * 100).round(1)
    k["late_rate_pp"] = (k["late_rate"].diff() * 100).round(1)   # зміна у процентних пунктах
    return k.round(3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/olist_clean.csv")
    ap.add_argument("--sample-out", default="data/sample.csv")
    ap.add_argument("--from", dest="date_from", default="2017-01-01")
    ap.add_argument("--to", dest="date_to", default="2018-08-31")
    args = ap.parse_args()

    tables = load_tables(Path(args.raw))
    tables_overview(tables)
    integrity_report(tables)

    raw_df = build_dataset(tables)
    preview(raw_df, "ПІСЛЯ MERGE")
    print("\nЗамовлень по місяцях:")
    print(monthly_orders(raw_df).to_string())

    df, steps = clean(raw_df, args.date_from, args.date_to)
    print("\nКроки очищення:")
    print("\n".join(f"  • {s}" for s in steps))
    preview(df, "ПІСЛЯ ОЧИЩЕННЯ")
    print("\nKPI по місяцях:")
    print(monthly_kpis(df).to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    df.sample(min(10_000, len(df)), random_state=42).sort_values("order_purchase_timestamp") \
      .to_csv(args.sample_out, index=False)
    print(f"\nЗбережено: {args.out} ({len(df):,} рядків) та {args.sample_out}")


if __name__ == "__main__":
    main()
