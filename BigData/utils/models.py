# funciones entrenar los modelos con pyspark
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator


def estimate_lr(df, input_col, label_col):
    """
    Estima los coeficientes de la regresión lineal
    df: dataframe
    input_col: columnas de entrada
    label_col: target
    """
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    # ensambla las columnas de entrada
    assembler = VectorAssembler(inputCols=input_col, outputCol="features")
    # normaliza las columnas
    normalizer = Normalizer(inputCol="features", outputCol="features_norm")
    # regresión lineal
    lr = LinearRegression(
        featuresCol="features_norm",
        labelCol=label_col,
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
    )
    # pipeline
    pipeline = Pipeline(stages=[assembler, normalizer, lr])
    # ajusta el modelo
    model = pipeline.fit(train)
    transformed = model.transform(test)
    transformed.show()
    return model


def estimate_kmeans(df, input_col, k):
    """
    Estima los centroides de los clusters
    """
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    # ensambla las columnas de entrada
    assembler = VectorAssembler(inputCols=input_col, outputCol="features")
    # kmeans
    kmeans = KMeans(featuresCol="features", k=k)
    # pipeline
    pipeline = Pipeline(stages=[assembler, kmeans])
    # ajusta el modelo
    model = pipeline.fit(train)
    tranformed = model.transform(test)
    tranformed.show()
    return model


# Entrenar y evaluar modelos
def train_and_evaluate(model, param_grid, training_data, test_data):
    """
    Entrena y evalúa un modelo dado utilizando validación cruzada.
    """
    evaluator = RegressionEvaluator(
        labelCol="ArrDelay", predictionCol="prediction", metricName="rmse"
    )

    # Configurar validación cruzada
    crossval = CrossValidator(
        estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3
    )

    # Entrenar el modelo
    cv_model = crossval.fit(training_data)

    # Evaluar en datos de prueba
    predictions = cv_model.bestModel.transform(test_data)
    rmse = evaluator.evaluate(predictions)

    print(f"\nResultados del modelo {type(model).__name__}:")
    predictions.select("ArrDelay", "prediction").show(
        20, truncate=False
    )  # Mostrar predicciones y etiquetas reales
    print(f"RMSE: {rmse:.3f}")
    return cv_model.bestModel, rmse


def perform_kmeans_clustering(data: DataFrame, k_values: list):
    """
    Realiza clustering usando KMeans con múltiples valores de k y evalúa los resultados.

    Args:
        data (DataFrame): Dataset preprocesado con columna "features".
        k_values (list): Lista de valores de k (número de clústeres) a probar.

    Returns:
        dict: Resultados con centroides, WSSSE (Within Set Sum of Squared Errors), y evaluaciones para cada k.
    """
    results = {}
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    for k in k_values:
        # Configurar KMeans con k clústeres
        kmeans = KMeans(featuresCol="features", k=k, seed=42)
        model = kmeans.fit(data)

        # Predecir clústeres y evaluar
        predictions = model.transform(data)
        silhouette_score = evaluator.evaluate(predictions)

        # Obtener WSSSE y centroides
        wssse = model.summary.trainingCost
        centroids = model.clusterCenters()

        # Guardar resultados
        results[k] = {
            "model": model,
            "predictions": predictions,
            "silhouette_score": silhouette_score,
            "wssse": wssse,
            "centroids": centroids,
        }

        # Imprimir resultados
        print(f"KMeans con k={k}")
        print(f"Silhouette Score: {silhouette_score:.3f}")
        print(f"WSSSE: {wssse:.3f}")
        print("Centroides:")
        for idx, centroid in enumerate(centroids):
            print(f"  Centroid {idx}: {centroid}")
        print()

    return results
