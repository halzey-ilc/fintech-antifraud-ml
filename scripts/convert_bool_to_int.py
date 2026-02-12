from __future__ import annotations

import pandas as pd

#Временный файл для преобразования bool в int

def main() -> None:
    path = "data/raw/transactions.csv"
    df = pd.read_csv(path)

    bool_cols = list(df.select_dtypes(include=["bool"]).columns)
    for c in bool_cols:
        df[c] = df[c].astype("int8")

    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Converted bool->int: {bool_cols}")


if __name__ == "__main__":
    main()
