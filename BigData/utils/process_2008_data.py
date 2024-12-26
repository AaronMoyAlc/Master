from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit


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
        # Initialize Spark session
        spark = SparkSession.builder.appName("Process2008Data").getOrCreate()

        # Load the 2008 dataset
        df_2008 = spark.read.csv(input_file_2008, header=True, inferSchema=True)

        # Columns to drop
        columns_to_drop = [
            "Year",
            "FlightNum",
            "ActualElapsedTime",
            "AirTime",
            "Distance",
            "TaxiIn",
            "TaxiOut",
            "CancellationCode",
            "Diverted",
            "CarrierDelay",
            "WeatherDelay",
            "NASDelay",
            "SecurityDelay",
            "LateAircraftDelay",
        ]

        # Drop specified columns
        df_2008 = df_2008.drop(*columns_to_drop)

        # Remove rows with null values
        df_2008 = df_2008.na.drop()

        # Load the plane-data.csv
        plane_data = spark.read.csv(input_file_plane, header=True, inferSchema=True)

        # Select only the necessary columns from plane-data.csv
        plane_data = plane_data.select(
            col("tailnum").alias("TailNum"), col("issue_date").alias("IssueDate")
        )

        # Join df_2008 with plane_data on TailNum
        df_2008 = df_2008.join(plane_data, on="TailNum", how="left")

        # Save the processed dataset to a new file
        df_2008.write.csv(output_file, header=True, mode="overwrite")

        print(f"Processed dataset saved to {output_file}")

        # Stop the Spark session
        spark.stop()
    except Exception as e:
        print(f"Error processing the dataset: {e}")
