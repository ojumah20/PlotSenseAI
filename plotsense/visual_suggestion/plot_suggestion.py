import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import functools

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")
    return df

#there are two other types of data : Boolean and text that will need to discuss
def detect_data_types(df):
    data_types = {
        'numerical': [],
        'categorical': [],
        'datetime': []
    }
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            data_types['numerical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            data_types['datetime'].append(col)
        else:
            data_types['categorical'].append(col)
    return data_types




def plot_decorator(func):
    """Decorator to log, handle errors, and measure execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"[LOG] Function '{func.__name__}' executed in {execution_time:.4f} seconds")
            print(f"[LOG] Recommended Chart: {result}")
            return result
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            return "Error in generating recommendation"
    return wrapper


@plot_decorator
def recommend_plot(df):
    data_types = detect_data_types(df)

    num_cols = data_types['numerical']
    cat_cols = data_types['categorical']
    time_cols = data_types['datetime']

    # Univariate Analysis
    if len(num_cols) == 1 and not cat_cols and not time_cols:
        return "Recommended: Histogram"

    if len(cat_cols) == 1 and not num_cols and not time_cols:
        return "Recommended: Bar Chart"

    # Bivariate Analysis
    if len(num_cols) == 2 and not cat_cols and not time_cols:
        return "Recommended: Scatter Plot"

    if len(num_cols) == 1 and len(cat_cols) == 1 and not time_cols:
        return "Recommended: Box Plot"

    if len(time_cols) == 1 and len(num_cols) == 1:
        return "Recommended: Line Chart"

    # Multivariate Analysis (Three or More Numerical Columns)
    if len(num_cols) >= 3 and not cat_cols and not time_cols:
        return "Recommended: Pairplot (Multivariate Analysis) or Heatmap"

    if len(num_cols) >= 3 and len(cat_cols) == 1 and not time_cols:
        return "Recommended: Pairplot (Grouped by Category), Box Plot (One num per cat), or Grouped Bar Chart"

    if len(num_cols) == 2 and len(cat_cols) == 1:
        return "Recommended: Regression Plot"

    if len(num_cols) >= 2 and len(cat_cols) >= 1:
        return "Recommended: Grouped Bar Chart (Multivariate Analysis) or Violin Plot"

    # Time-Series with Multiple Numerical Columns
    if len(time_cols) == 1 and len(num_cols) > 1:
        return "Recommended: Multi-Line Plot, Facet Grid, or Stacked Area Chart"

    # Aggregated/Proportional Data
    if len(cat_cols) == 1 and len(num_cols) == 1:
        total_sum = df[num_cols[0]].sum()
        if total_sum > 0 and total_sum <= 100:
            return "Recommended: Pie Chart"
        else:
            return "Recommended: Bar Chart"

    # High-Dimensional Data (More than 4+ Numerical Columns)
    if len(num_cols) > 4:
        return "Recommended: Parallel Coordinates Plot or PCA/t-SNE Visualization"

    return "No suitable chart found. Recommend using the AI version."




df = load_data('sample.csv')
print(recommend_plot(df))