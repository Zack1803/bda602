# pre commit hooks

import sys

from pyspark import StorageLevel, keyword_only
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

# Create a spark session
spark = (
    SparkSession.builder.config(
        "spark.jars",
        "/Users/zack/Documents/SDSU/Fall 2022/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar",
    )
    .master("local")
    .appName("HW3")
    .getOrCreate()
)

# Define Transformer Class


class RollingAverageTransform(Transformer):
    @keyword_only
    def __init__(self):
        super(RollingAverageTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(data):
        rolling_avg = rolling_average_calculation(spark, data)
        return rolling_avg


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


# Intermediate Calculation


def intermediate_data():
    game_sql = """ \
        SELECT game_id, \
        local_date \
        FROM game"""

    battercounts_sql = """ \
        SELECT game_id, \
        batter,\
        atbat,\
        hit \
        FROM batter_counts"""

    game = load_data(game_sql)  # getting the game table data
    batter_counts = load_data(battercounts_sql)  # Loading the batter_counts data

    intermediate_df = batter_counts.join(game, on="game_id")  # joining the tables

    return intermediate_df


# Function to calculate rolling average


def rolling_average_calculation(spark, data):
    rolling_average_sql = """ \
    SELECT a.batter, \
    a.local_date, \
    (sum(b.Hit)/NULLIF(sum(b.atBat),0)) as rolling_avg \
    FROM rolling_average_intermediate as a \
    JOIN rolling_average_intermediate as b \
    ON a.batter = b.batter \
    AND a.local_date > b.local_date \
    AND b.local_date BETWEEN a.local_date - INTERVAL 100 DAY and a.local_date \
    group by a.batter,a.local_date \
    order by a.local_date DESC """

    data.createOrReplaceTempView("rolling_average_intermediate")
    data.persist(StorageLevel.DISK_ONLY)
    rolling_average_intermediate = spark.sql(rolling_average_sql)

    return rolling_average_intermediate


def main():

    # Loading the game and batter counts data
    intermediate_df = intermediate_data()
    # Create an object of transformer class
    obj = RollingAverageTransform
    # Passing the data to transformer
    result = obj._transform(intermediate_df)

    # Displaying the first 10 rows of the result
    result.show()


if __name__ == "__main__":
    sys.exit(main())
