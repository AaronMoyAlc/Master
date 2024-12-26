from pyspark.sql.functions import (
    col,
    year,
    avg,
    when,
    lit,
    coalesce,
    to_date,
    concat_ws,
    months_between,
)
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler


def preprocessing(df):
    # obtenemos la fecha de salida del vuelo
    # Año fijo del dataset
    year = lit(2008)  # Ajusta esto según el año de tus datos

    # Crear la columna FlightDate a partir de Year, Month y DayofMonth
    data = df.withColumn(
        "FlightDate", to_date(concat_ws("-", year, col("Month"), col("DayofMonth")))
    )

    # Calcular la antigüedad del avión en años (PlaneAge)
    data = data.withColumn(
        "PlaneAge",
        (months_between(col("FlightDate"), col("IssueDate"))).cast("double"),
    )

    # eliminamos la columna FlightDate
    data = data.drop("FlightDate")

    # Rellenar nulos en PlaneAge con la media
    avg_age = data.select(avg("PlaneAge")).first()[0]
    data = data.withColumn(
        "PlaneAge", when(col("PlaneAge").isNull(), avg_age).otherwise(col("PlaneAge"))
    )

    # Convertir columnas categóricas a índices numéricos
    categorical_columns = ["UniqueCarrier", "Origin", "Dest"]
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_Index")
        for col in categorical_columns
    ]
    for indexer in indexers:
        data = indexer.fit(data).transform(data)

    # Eliminar columnas originales categóricas
    data = data.drop(*categorical_columns)

    # Normalizar los valores
    feature_columns = [
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "DepTime",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "DepDelay",
        "Cancelled",
        "PlaneAge",
        "UniqueCarrier_Index",
        "Origin_Index",
        "Dest_Index",
    ]

    # Impute nulls with 0 before assembling the features
    # This will prevent the MinMaxScaler from failing
    for col_name in feature_columns:
        data = data.withColumn(col_name, coalesce(col(col_name), lit(0)))

    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="features_assembled"
    )
    data = assembler.transform(data)

    scaler = MinMaxScaler(inputCol="features_assembled", outputCol="features")
    scaler_model = scaler.fit(data)
    data = scaler_model.transform(data)

    # Seleccionar columnas finales (incluye la normalizada y la variable objetivo)
    df = data.select("features", "ArrDelay")

    # Mostrar algunas filas del conjunto preprocesado
    df.show(truncate=False)
    return df
