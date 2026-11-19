import argparse
from pathlib import Path
import pandas as pd

METHOD_NAMES = ["fedavg", "fedprox", "heterofl", "fedmeta", "splitfed"]

def infer_method(path: Path) -> str:
    name = str(path).lower()
    for method in METHOD_NAMES:
        if method in name:
            return method
    return "unknown"

def read_result_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "method" not in df.columns:
        df["method"] = infer_method(path)

    if "source_file" not in df.columns:
        df["source_file"] = str(path)

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="all_fed_results.csv")
    args = parser.parse_args()

    root = Path(args.root)
    files = list(root.rglob("*.csv"))

    rows = []

    for file in files:
        if args.out in file.name:
            continue

        try:
            df = read_result_file(file)
            rows.append(df)
            print(f"[OK] Added {file}")
        except Exception as e:
            print(f"[SKIP] {file}: {e}")

    if not rows:
        print("No CSV files found.")
        return

    final = pd.concat(rows, ignore_index=True)
    final.to_csv(args.out, index=False)

    print(f"\nSaved combined CSV to: {args.out}")

if __name__ == "__main__":
    main()
