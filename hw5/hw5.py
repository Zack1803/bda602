import sys
from pyspark.sql import SparkSession

# Create a spark session
spark = (
    SparkSession.builder.config(
        "spark.jars",
        "/Users/zack/Documents/SDSU/Fall 2022/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar",
    )
    .master("local")
    .appName("HW5")
    .getOrCreate()
)


# Create a function to load the data from MariaDB
def load_data(query):

    table_data = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball")
        .option("driver", "com.mysql.jdbc.Driver")
        .option("query", query)
        .option("user", "root")
        .option("password", "believe")
        .load()
    )
    return table_data


def convert_to_pandas_df(spark_df):
    df = spark_df.toPandas()
    print(df.head())





def main():

    # Loading the Features Table
    features_sql = """ SELECT * FROM FEATURES """
    Features = load_data(features_sql)
    print(type(Features))

    #Converting Spark Dataframe to Pandas Dataframe
    convert_to_pandas_df(Features)



if __name__ == "_main_":
    sys.exit(main())