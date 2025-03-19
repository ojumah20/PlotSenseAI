import pandas as pd
import functools

def load_data(file_path):
    """Loads data from CSV, Excel, or JSON file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, engine='openpyxl') 
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    raise ValueError("Unsupported file format")

def detect_data_types(df):
    """Detects data types and categorizes them into numerical, categorical, datetime, boolean, and text."""
    data_types = {'numerical': [], 'categorical': [], 'datetime': [], 'boolean': [], 'text': []}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            data_types['numerical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            data_types['datetime'].append(col)
        elif pd.api.types.is_bool_dtype(df[col]):
            data_types['boolean'].append(col)
        elif pd.api.types.is_string_dtype(df[col]):
            data_types['text'].append(col)
        else:
            data_types['categorical'].append(col)  # Fallback for non-numeric non-text

    return data_types

def plot_decorator(func):
    """Decorator to log and handle errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] {e}")
            return "Error in generating recommendation"
    return wrapper

@plot_decorator
def recommend_plot(df):
    """Recommends an appropriate plot based on data types in the dataset."""
    data_types = detect_data_types(df)

    num_cols = data_types['numerical']
    cat_cols = data_types['categorical']
    time_cols = data_types['datetime']
    bool_cols = data_types['boolean']
    text_cols = data_types['text']

    # Univariate Analysis
    if len(num_cols) == 1 and not (cat_cols or time_cols or text_cols):
        return "Recommended: Histogram"

    if len(cat_cols) == 1 and not (num_cols or time_cols or text_cols):
        return "Recommended: Bar Chart"

    # Bivariate Analysis
    if len(num_cols) == 2 and not (cat_cols or time_cols or text_cols):
        return "Recommended: Scatter Plot"

    if len(num_cols) == 1 and len(cat_cols) == 1:
        return "Recommended: Box Plot"

    if len(time_cols) == 1 and len(num_cols) == 1:
        return "Recommended: Line Chart"

    # Multivariate Analysis (Three or More Numerical Columns)
    if len(num_cols) >= 3 and not (cat_cols or time_cols or text_cols):
        return "Recommended: Pairplot (Multivariate Analysis) or Heatmap"

    if len(num_cols) >= 3 and len(cat_cols) == 1:
        return "Recommended: Pairplot (Grouped by Category), Box Plot, or Grouped Bar Chart"

    if len(num_cols) == 2 and len(cat_cols) == 1:
        return "Recommended: Regression Plot"

    if len(num_cols) >= 2 and len(cat_cols) >= 1:
        return "Recommended: Grouped Bar Chart or Violin Plot"

    # Time-Series with Multiple Numerical Columns
    if len(time_cols) == 1 and len(num_cols) > 1:
        return "Recommended: Multi-Line Plot, Facet Grid, or Stacked Area Chart"

    # Boolean Data
    if bool_cols:
        return "Recommended: Stacked Bar Chart or Count Plot for Boolean Data"

    # Text Data
    if text_cols:
        return "Recommended: Word Cloud, Text Frequency Bar Chart, or Sentiment Analysis"

    # Aggregated/Proportional Data
    if len(cat_cols) == 1 and len(num_cols) == 1:
        total_sum = df[num_cols[0]].sum()
        return "Recommended: Pie Chart" if 0 < total_sum <= 100 else "Recommended: Bar Chart"

    # High-Dimensional Data (More than 4+ Numerical Columns)
    if len(num_cols) > 4:
        return "Recommended: Parallel Coordinates Plot or PCA/t-SNE Visualization"

    return "No suitable chart found. Recommend using the AI version."
