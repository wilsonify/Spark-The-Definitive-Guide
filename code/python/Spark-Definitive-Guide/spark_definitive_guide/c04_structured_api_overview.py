from pyspark.shell import spark
from pyspark.sql.types import *

df = spark.range(500).toDF("number")
df.select(df["number"] + 10)

# COMMAND ----------

spark.range(2).collect()

# COMMAND ----------


b = ByteType()

# COMMAND ----------
