from pyspark.shell import spark
from pyspark.sql.functions import date_format, col
from pyspark.sql.functions import window, desc
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler

from spark_definitive_guide import path_to_data

staticDataFrame = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(f"{path_to_data}/retail-data/by-day/*.csv")

staticDataFrame.createOrReplaceTempView("retail_data")
staticSchema = staticDataFrame.schema

# COMMAND ----------


staticDataFrame \
    .selectExpr("CustomerId", "(UnitPrice * Quantity) as total_cost", "InvoiceDate") \
    .groupBy(col("CustomerId"), window(col("InvoiceDate"), "1 day")) \
    .sum("total_cost") \
    .sort(desc("sum(total_cost)")) \
    .show(5)

# COMMAND ----------

streamingDataFrame = spark.readStream \
    .schema(staticSchema) \
    .option("maxFilesPerTrigger", 1) \
    .format("csv") \
    .option("header", "true") \
    .load(f"{path_to_data}/retail-data/by-day/*.csv")

# COMMAND ----------

purchaseByCustomerPerHour = streamingDataFrame \
    .selectExpr("CustomerId", "(UnitPrice * Quantity) as total_cost", "InvoiceDate") \
    .groupBy(col("CustomerId"), window(col("InvoiceDate"), "1 day")) \
    .sum("total_cost")

# COMMAND ----------

purchaseByCustomerPerHour.writeStream \
    .format("memory") \
    .queryName("customer_purchases") \
    .outputMode("complete") \
    .start()

# COMMAND ----------

spark.sql("""
  SELECT *
  FROM customer_purchases
  ORDER BY `sum(total_cost)` DESC
  """) \
    .show(5)

# COMMAND ----------


preppedDataFrame = staticDataFrame \
    .na.fill(0) \
    .withColumn("day_of_week", date_format(col("InvoiceDate"), "EEEE")) \
    .coalesce(5)

# COMMAND ----------

trainDataFrame = preppedDataFrame \
    .where("InvoiceDate < '2011-07-01'")
testDataFrame = preppedDataFrame \
    .where("InvoiceDate >= '2011-07-01'")

# COMMAND ----------


indexer = StringIndexer() \
    .setInputCol("day_of_week") \
    .setOutputCol("day_of_week_index")

# COMMAND ----------


encoder = OneHotEncoder() \
    .setInputCol("day_of_week_index") \
    .setOutputCol("day_of_week_encoded")

# COMMAND ----------


vectorAssembler = VectorAssembler() \
    .setInputCols(["UnitPrice", "Quantity", "day_of_week_encoded"]) \
    .setOutputCol("features")

# COMMAND ----------


transformationPipeline = Pipeline() \
    .setStages([indexer, encoder, vectorAssembler])

# COMMAND ----------

fittedPipeline = transformationPipeline.fit(trainDataFrame)

# COMMAND ----------

transformedTraining = fittedPipeline.transform(trainDataFrame)

# COMMAND ----------


kmeans = KMeans() \
    .setK(20) \
    .setSeed(1)

# COMMAND ----------

kmModel = kmeans.fit(transformedTraining)

# COMMAND ----------

transformedTest = fittedPipeline.transform(testDataFrame)

# COMMAND ----------


spark.sparkContext.parallelize([Row("1"), Row("2"), Row("3")]).toDF()

# COMMAND ----------
