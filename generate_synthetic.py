from pathlib import Path
import numpy as np
import pandas as pd

def make_synthetic(n: int = 150_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2026-08-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
    df = pd.DataFrame({
        "session_id": np.arange(n),
        "date": dates,
        "channel": rng.choice(["organic", "paid_search", "paid_social", "email", "referral"], n,
                              p=[.30, .25, .20, .15, .10]),
        "platform": rng.choice(["ios", "android", "web"], n, p=[.25, .35, .40]),
        "country": rng.choice(["UA", "PL", "DE", "US", "GB"], n, p=[.40, .20, .15, .15, .10]),
        "new_user": rng.choice([0, 1], n, p=[.40, .60]),
    })
    p = 0.06 + 0.02 * (df.channel == "email") - 0.015 * (df.channel == "paid_social") + 0.01 * (df.new_user == 0)
    mask = (df.date >= "2026-10-16") & (df.platform == "android") & (df.channel == "paid_social")
    p = np.where(mask, p * 0.2, p)

    df["converted"] = (rng.random(n) < p).astype(int)
    df["revenue"] = np.where(df.converted == 1, rng.lognormal(3.5, 0.6, n).round(2), 0.0)

    df.loc[rng.choice(n, int(n * 0.025), replace=False), "country"] = None
    df.loc[rng.choice(df.index[df.converted == 1], 40, replace=False), "revenue"] = 99999.0
    df = pd.concat([df, df.sample(200, random_state=1)], ignore_index=True)  # 200 дублікатів
    return df

if __name__ == "__main__":
    out = Path("data/synthetic_sessions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = make_synthetic()
    data.to_csv(out, index=False)
    print(data.shape, "→", out)
