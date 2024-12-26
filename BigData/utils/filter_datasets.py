from pyspark.sql import SparkSession


def filter_columns(input_file, output_file, columns_to_keep):
    """
    Reads a CSV file with PySpark, keeps specified columns, and saves to a new file.

    :param input_file: str, path to the input CSV file
    :param output_file: str, path to the output CSV file
    :param columns_to_keep: list of str, columns to retain in the output file
    """
    try:
        # Initialize Spark session
        spark = SparkSession.builder.appName("FilterColumns").getOrCreate()

        # Load the dataset
        df = spark.read.csv(input_file, header=True, inferSchema=True)

        # Keep only the specified columns
        filtered_df = df.select(*columns_to_keep)

        # Save the filtered dataset to a new CSV
        filtered_df.write.csv(output_file, header=True, mode="overwrite")

        print(f"Filtered dataset saved to {output_file}")

        # Stop the Spark session
        spark.stop()
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
