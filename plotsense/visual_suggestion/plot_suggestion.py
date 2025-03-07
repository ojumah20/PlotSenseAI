import pandas as pd

def log_recommendation(func):
    """
    Decorator to log the recommendation process.
    """
    def wrapper(data):
        print("Analyzing data...")
        recommendation = func(data)
        print(f"Recommended Chart: {recommendation}")
        return recommendation
    return wrapper

@log_recommendation
def recommend_chart(data):
    """
    Recommends a chart type based on the dataset.
    """
    num_cols = data.select_dtypes(include=['number']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(num_cols) == 1 and not cat_cols:
        return "Histogram (for distribution analysis)"
   
    if len(num_cols) == 1 and len(cat_cols) == 1:
        return "Bar Chart (for category-wise comparisons)"
   
    if len(num_cols) == 2:
        return "Scatter Plot (for correlation analysis)"
   
    if len(num_cols) >= 2 and ('date' in data.columns or pd.api.types.is_datetime64_any_dtype(data.index)):
        return "Line Chart (for trends over time)"
   
    return "No suitable chart found."