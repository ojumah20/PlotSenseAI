import pandas as pd
from plotsense.plot_generator.seaborn import generate_seaborn_lineplot

def test_generate_seaborn_lineplot():
    """Tests if the Seaborn line plot function runs without errors."""
    sample_data = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "sales": range(10)
    })
    generate_seaborn_lineplot(sample_data, x="date", y="sales")