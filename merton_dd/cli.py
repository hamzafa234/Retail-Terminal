"""Interactive command-line entry point."""
from typing import Optional

from merton_dd import pipeline
from merton_dd.plotting import plot_default_probabilities


def prompt_ticker() -> Optional[str]:
    while True:
        ticker = input("Enter ticker symbol (or 'exit' to quit): ").strip().upper()
        if ticker == "EXIT":
            print("Exiting...")
            return None
        if ticker:
            return ticker
        print("Error: Please enter a valid ticker symbol\n")


def prompt_years(low: int = 1, high: int = 10) -> int:
    while True:
        raw = input(f"Enter number of years of data ({low}-{high}): ").strip()
        try:
            years = int(raw)
        except ValueError:
            print("Error: Please enter a valid number")
            continue
        if low <= years <= high:
            return years
        print(f"Error: Please enter a number between {low} and {high}")


def main() -> None:
    ticker = prompt_ticker()
    if ticker is None:
        return
    years = prompt_years()
    dates, probabilities = pipeline.run(ticker, years)
    plot_default_probabilities(dates, probabilities)


if __name__ == "__main__":
    main()
