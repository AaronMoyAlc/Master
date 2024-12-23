import pandas as pd

def process_2008_data(input_file_2008, input_file_plane, output_file):
    """
    Processes the 2008.csv file by:
    1. Dropping specified columns.
    2. Removing rows with null values.
    3. Adding a new column IssueDate from plane-data.csv immediately after TailNum.

    :param input_file_2008: str, path to the 2008.csv file
    :param input_file_plane: str, path to the plane-data.csv file
    :param output_file: str, path to save the processed CSV file
    """
    try:
        # Load the 2008 dataset
        df_2008 = pd.read_csv(input_file_2008)

        # Columns to drop
        columns_to_drop = [
            "Year", "FlightNum", "ActualElapsedTime", "AirTime", "Distance", "TaxiIn", "TaxiOut",
            "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay",
            "SecurityDelay", "LateAircraftDelay"
        ]

        # Drop specified columns
        df_2008 = df_2008.drop(columns=columns_to_drop, errors='ignore')

        # Remove rows with null values
        df_2008 = df_2008.dropna()

        # Load the plane-data.csv
        plane_data = pd.read_csv(input_file_plane)

        # Create a dictionary for tailnum to issue_date mapping
        tailnum_to_issue_date = plane_data.set_index("tailnum")["issue_date"].to_dict()

        # Map IssueDate to the 2008 dataset
        df_2008["IssueDate"] = df_2008["TailNum"].map(tailnum_to_issue_date)

        # Reorder columns to place IssueDate immediately after TailNum
        tailnum_index = df_2008.columns.get_loc("TailNum")
        cols = list(df_2008.columns)
        cols.insert(tailnum_index + 1, cols.pop(cols.index("IssueDate")))
        df_2008 = df_2008[cols]

        # Save the processed dataset to a new file
        df_2008.to_csv(output_file, index=False)

        print(f"Processed dataset saved to {output_file}")
    except Exception as e:
        print(f"Error processing the dataset: {e}")

# Input and output file paths
input_2008_file = "2008.csv"
input_plane_file = "plane-data.csv"
output_file = "processed_2008.csv"

# Run the function
process_2008_data(input_2008_file, input_plane_file, output_file)