"""
=============================================================
AI Food Quality Platform

Dataset Analyzer - Entry Point

Author : Dia Maria
=============================================================
"""

from pathlib import Path

from scripts.config.constants import RGB_DATASET

from .analyzer import analyze_dataset

import traceback


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Runs the Dataset Analyzer.
    """

    print("=" * 70)
    print("AI FOOD QUALITY PLATFORM")
    print("DATASET ANALYZER")
    print("=" * 70)

    dataset_path = Path(RGB_DATASET)

    print(f"\nDataset : {dataset_path}")

    try:

        results = analyze_dataset(dataset_path)

        print("\n")
        print("=" * 70)
        print("Dataset Analysis Completed Successfully")
        print("=" * 70)

        summary = results["summary"]

        print("\nDataset Summary")
        print("-" * 30)

        print(f"Excellent Categories : {summary['excellent']}")
        print(f"Good Categories      : {summary['good']}")
        print(f"Moderate Categories  : {summary['moderate']}")
        print(f"Problematic          : {summary['problematic']}")

        print("\nReport generated successfully.")

    except FileNotFoundError as error:

        print("\nDataset not found!")
        print(error)

    except Exception:

        print("\nUnexpected Error")
        traceback.print_exc()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()

