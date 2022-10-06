import os
import sys

from pyspark import StorageLevel
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext


def set_conf():
    # Creating a configuration file for my Spark object
    # Need spark.jars setting for connection to mariadb

    conf = SparkConf().setAppName("App")
    conf = (conf.setMaster('local[*]')
            .set('spark.jars',
                 '/Users/zack/Documents/SDSU/Fall 2022/BDA 602/src/bda602/hw3/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar'))



    return conf


def main():

    # Making Spark Object
    sc = pyspark.SparkContext.getOrCreate(conf=set_conf())
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.master("local[*]").getOrCreate()


    # Making connection to MariaDB
    mysql_db_driver_class = "com.mysql.jdbc.Driver"
    host_name = "localhost"
    port_no = "3306"
    user_name = "root"
    password = "believe"
    database_name = "baseball?zeroDateTimeBehavior=convertToNull"

    #test_query = ""

    # Making JDBC URL
    mysql_jdbc_url = ('jdbc:mysql://' + host_name +
                      ':' + port_no + '/' + database_name)

    # Reading DataTable from jdbc
    game_df = sqlContext.read \
        .format("jdbc") \
        .option("url", mysql_jdbc_url) \
        .option("driver", mysql_db_driver_class) \
        .option("dbtable", "game") \
        .option("user", user_name) \
        .option("password", password) \
        .load()

    batter_counts_df = sqlContext.read \
        .format("jdbc") \
        .option("url", mysql_jdbc_url) \
        .option("driver", mysql_db_driver_class) \
        .option("dbtable", "batter_counts") \
        .option("user", user_name) \
        .option("password", password) \
        .load()

    # game_df.show()
    # batter_counts_df.show()
    game_df.createOrReplaceTempView("game")
    game_df.persist(StorageLevel.DISK_ONLY)
    batter_counts_df.createOrReplaceTempView("batter_counts")
    batter_counts_df.persist(StorageLevel.DISK_ONLY)

    # Simple SQL
    intermediate_table_df = spark.sql(
        """
    SELECT 
     BC.batter,
     G.local_date, 
     BC.hit,
     BC.atBat
    FROM batter_counts BC
    JOIN game G
    ON BC.game_id = G.game_id
        """
    )
    #intermediate_table.show()
    intermediate_table_df.createOrReplaceTempView("rolling_average_intermediate")
    intermediate_table_df.persist(StorageLevel.DISK_ONLY)

    rolling_average = spark.sql(
        """
    select a.batter, a.local_date, (sum(b.Hit)/NULLIF(sum(b.atBat),0)) as rolling_avg
    from rolling_average_intermediate as a
    join rolling_average_intermediate as b
    on a.batter = b.batter and a.local_date > b.local_date and b.local_date between  a.local_date - INTERVAL 100 DAY and a.local_date
    group by a.batter,a.local_date
    order by a.local_date DESC
        """
    )
    rolling_average.show()
    return

if __name__ == "__main__":
    sys.exit(main())