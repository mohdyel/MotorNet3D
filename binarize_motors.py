# binarize_motors.py

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Binarize the 'Number of motors' column in a CSV.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_csv", help="Path where the modified CSV will be saved")
    args = parser.parse_args()

    # Load the CSV
    df = pd.read_csv(args.input_csv)

    # Replace in "Number of motors": 0 → 0, anything else → 1
    df["Number of motors"] = df["Number of motors"].apply(lambda x: 0 if x == 0 else 1)

    # Save to the output path
    df.to_csv(args.output_csv, index=False)
    print(f"Saved binary motor column to {args.output_csv}")

if __name__ == "__main__":
    main()
