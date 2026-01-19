import pandas as pd
import plotly.express as px
import json

def generate_charts(data):
    """
    Takes the nested JSON data and returns a dictionary of Plotly Figures.
    
    Args:
        data (dict): The input JSON dictionary containing dataset metrics.
        
    Returns:
        dict: Keys are dataset names (str), Values are Plotly Figure objects.
    """
    figures = {}

    # Iterate through each top-level dataset (e.g., 'bcopa', 'paws')
    for dataset_name, segments in data.items():
        
        # 1. Flatten the data for this specific dataset
        rows = []
        for segment_key, counts in segments.items():
            # Clean the segment name (e.g., 'tags_premise' -> 'Premise')
            clean_segment = segment_key.replace("tags_", "").replace("_", " ").title()
            
            for category, count in counts.items():
                # Filter out 0 counts to keep the chart clean
                if count > 0:
                    rows.append({
                        "Segment": clean_segment,
                        "Category": category.replace("_", " ").title(), # Clean 'di_karaniwang_ayos'
                        "Count": count
                    })
        
        # If no data exists for this dataset (e.g. all zeros), skip it
        if not rows:
            continue

        # 2. Create DataFrame
        df = pd.DataFrame(rows)

        # 3. Create the Plotly Figure
        # We use a Stacked Bar chart to compare Ayos within each segment
        fig = px.bar(
            df,
            x="Segment",
            y="Count",
            color="Category",
            title=f"Sentence Structure Distribution: {dataset_name.upper()}",
            text_auto=True, # Show values inside bars
            color_discrete_map={
                "Karaniwang Ayos": "#2E86C1",     # Blue
                "Di Karaniwang Ayos": "#E74C3C",  # Red
                "Ambiguous": "#95A5A6"            # Grey (if it appears)
            }
        )
        
        # 4. Refine Layout
        fig.update_layout(
            barmode='stack', 
            xaxis_title=None, # Clean look
            yaxis_title="Count",
            legend_title="Structure Type",
            # Make it responsive
            autosize=True,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Add to our result dictionary
        figures[dataset_name] = fig

    return figures



def _read_json(path):
# Open and read a JSON file
    with open(path, "r") as file:
        data = json.load(file)
    return data

