from pyspark.sql import SparkSession
from models import estimate_lr, estimate_kmeans, train_and_evaluate
from eda import eda
from preprocessing import preprocessing
from filter_datasets import filter_columns
from process_2008_data import process_2008_data


def main():
    spark = SparkSession.builder.appName("proyecto").getOrCreate()
    file_configs = [
        {
            "input": "airports.csv",
            "output": "filtered_airports.csv",
            "columns": ["iata"],
        },
        {
            "input": "carriers.csv",
            "output": "filtered_carriers.csv",
            "columns": ["Code"],
        },
        {
            "input": "plane-data.csv",
            "output": "filtered_plane_data.csv",
            "columns": ["tailnum"],
        },
    ]

    # Process each file
    for config in file_configs:
        filter_columns(config["input"], config["output"], config["columns"])
    # Input and output file paths
    input_2008_file = "2008.csv"
    input_plane_file = "plane-data.csv"
    output_file = "processed_2008.csv"
    # original_col = [Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,UniqueCarrier,FlightNum,TailNum,ActualElapsedTime,CRSElapsedTime,AirTime,ArrDelay,DepDelay,Origin,Dest,Distance,TaxiIn,TaxiOut,Cancelled,CancellationCode,Diverted,CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay]
    # Run the function
    process_2008_data(input_2008_file, input_plane_file, output_file)
    df = spark.read.csv(output_file, header=True, inferSchema=True)
    numerical_cols = [
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "DepTime",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "ArrDelay",
        "DepDelay",
    ]
    categorical_cols = ["UniqueCarrier", "TailNum", "Origin", "Dest"]
    eda(df, numerical_cols, categorical_cols)

    # Preprocesamiento de datos
    df = preprocessing(df)

    # trainig the models
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Modelo 1: Regresión Lineal
    lr = LinearRegression(featuresCol="features", labelCol="ArrDelay")
    param_grid_lr = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1, 0.5])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build()
    )

    best_lr_model, rmse_lr = train_and_evaluate(
        lr, param_grid_lr, train_data, test_data
    )

    # Modelo 2: Bosques Aleatorios
    rf = RandomForestRegressor(featuresCol="features", labelCol="ArrDelay")
    param_grid_rf = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100])
        .addGrid(rf.maxDepth, [5, 10])
        .build()
    )

    best_rf_model, rmse_rf = train_and_evaluate(
        rf, param_grid_rf, train_data, test_data
    )

    # Modelo 3: Gradient Boosted Trees
    gbt = GBTRegressor(featuresCol="features", labelCol="ArrDelay")
    param_grid_gbt = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [10, 50])
        .addGrid(gbt.maxDepth, [5, 10])
        .build()
    )

    best_gbt_model, rmse_gbt = train_and_evaluate(
        gbt, param_grid_gbt, train_data, test_data
    )

    # Comparar modelos
    results = [
        ("Linear Regression", rmse_lr),
        ("Random Forest", rmse_rf),
        ("Gradient Boosted Trees", rmse_gbt),
    ]
    results_sorted = sorted(results, key=lambda x: x[1])  # Ordenar por menor RMSE

    print("Model Comparison (RMSE):")
    for model_name, rmse in results_sorted:
        print(f"{model_name}: {rmse:.3f}")

    # Elegir el mejor modelo
    best_model_name, best_rmse = results_sorted[0]
    print(f"\nBest Model: {best_model_name} with RMSE = {best_rmse:.3f}")


if __name__ == "__main__":
    main()
