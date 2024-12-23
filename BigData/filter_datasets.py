import pandas as pd

def filter_columns(input_file, output_file, columns_to_keep):
    """
    Reads a CSV file, keeps specified columns, and saves to a new file.

    :param input_file: str, path to the input CSV file
    :param output_file: str, path to the output CSV file
    :param columns_to_keep: list of str, columns to retain in the output file
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_file)

        # Keep only the specified columns
        filtered_df = df[columns_to_keep]

        # Save the filtered dataset to a new CSV
        filtered_df.to_csv(output_file, index=False)

        print(f"Filtered dataset saved to {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Define input and output files along with columns to keep
file_configs = [
    {"input": "airports.csv", "output": "filtered_airports.csv", "columns": ["iata"]},
    {"input": "carriers.csv", "output": "filtered_carriers.csv", "columns": ["Code"]},
    {"input": "plane-data.csv", "output": "filtered_plane_data.csv", "columns": ["tailnum"]}
]

# Process each file
for config in file_configs:
    filter_columns(config["input"], config["output"], config["columns"])