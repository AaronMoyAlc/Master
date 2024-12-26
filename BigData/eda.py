from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, when, count, isnan, isnull, mean, round


def eda(df, numerical_cols, categorical_cols):
    """
    Análisis exploratorio de datos
    """
    # Mostrar esquema de las columnas
    df.printSchema()

    # Mostrar los primeros registros
    df.show(5)

    # 1. Resumen estadístico de las columnas numéricas
    df.select(numerical_cols).describe().show()

    # 2. Inspección de valores nulos o faltantes
    missing_data = df.select(
        [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
    )
    print("Cantidad de valores nulos por columna:")
    missing_data.show()

    # 3. Inspección de columnas categóricas

    for col_name in categorical_cols:
        print(f"Distribución de valores únicos para la columna {col_name}:")
        df.groupBy(col_name).count().orderBy("count", ascending=False).show(5)

    # 4. Inspección específica de la variable objetivo (ArrDelay)
    print("Estadísticas descriptivas de la variable objetivo (ArrDelay):")
    df.select("ArrDelay").describe().show()

    # 5. Identificar correlaciones básicas (opcional, solo entre columnas numéricas)

    assembler = VectorAssembler(inputCols=numerical_cols, outputCol="features")
    vector_df = assembler.transform(df).select("features")
    correlation_matrix = Correlation.corr(vector_df, "features").head()[0]
    print("Matriz de correlación:")
    print(correlation_matrix)
