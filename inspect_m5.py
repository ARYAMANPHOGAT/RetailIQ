
import csv
from pathlib import Path

folder = Path("data/m5")

for filename in [
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv"
]:
    path = folder / filename

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        print("\n" + "=" * 55)
        print(filename)
        print("Number of columns:", len(header))
        print("First 12 columns:", header[:12])
        print("Last 5 columns:", header[-5:])

        if filename == "sales_train_evaluation.csv":
            rows = sum(1 for _ in reader)
            print("Product-store series:", rows)
