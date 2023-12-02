from __future__ import print_function


def main():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .master("local") \
        .appName("Word Count") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    print(spark.range(5000).where("id > 500").selectExpr("sum(id)").collect())


if __name__ == '__main__':
    main()
