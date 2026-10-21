import plotext as plt


def plot_default_probabilities(dates, probabilities) -> None:
    plt.date_form("Y-m-d")
    plt.plot([d.strftime("%Y-%m-%d") for d in dates], probabilities)
    plt.show()
