from pyspark.sql import SparkSession
from utils.models import (
    train_and_evaluate,
    perform_kmeans_clustering,
)
from utils.eda import eda
from utils.preprocessing import preprocessing
from utils.filter_datasets import filter_columns
from utils.process_2008_data import process_2008_data
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor


def main():
    spark = SparkSession.builder.appName("Flight_delays_predictor").getOrCreate()
    file_configs = [
        {
            "input": "csv/airports.csv",
            "output": "csv/filtered_airports.csv",
            "columns": ["iata"],
        },
        {
            "input": "csv/carriers.csv",
            "output": "csv/filtered_carriers.csv",
            "columns": ["Code"],
        },
        {
            "input": "csv/plane-data.csv",
            "output": "csv/filtered_plane_data.csv",
            "columns": ["tailnum"],
        },
    ]

    # Process each file
    for config in file_configs:
        filter_columns(config["input"], config["output"], config["columns"])
    # Input and output file paths
    input_2008_file = "csv/2008.csv"
    input_plane_file = "csv/plane-data.csv"
    output_file = "csv/processed_2008.csv"
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
    preprocessed_df = preprocessing(df)

    # trainig the models
    train_data, test_data = preprocessed_df.randomSplit([0.8, 0.2], seed=42)

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

    # Estimar los centroides de los clusters
    # Realizar clustering

    k_values = [3, 4, 5, 7, 9]  # Lista de valores de k a probar
    clustering_results = perform_kmeans_clustering(preprocessed_df, k_values)

    # Obtener los resultados del mejor modelo según el Silhouette Score
    best_k = max(
        clustering_results, key=lambda k: clustering_results[k]["silhouette_score"]
    )
    best_model = clustering_results[best_k]["model"]
    print(
        f"Mejor modelo con k={best_k} y Silhouette Score={clustering_results[best_k]['silhouette_score']:.3f}"
    )

    # terminamos la sesión de Spark
    spark.stop()


if __name__ == "__main__":
    main()
