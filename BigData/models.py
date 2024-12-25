# funciones entrenar los modelos con pyspark
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.regression import LinearRegression

# kmeans
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline


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


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor


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
        10, truncate=False
    )  # Mostrar predicciones y etiquetas reales

    return cv_model.bestModel, rmse
