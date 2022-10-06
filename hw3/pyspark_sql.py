import sys
import tempfile

import requests
from pyspark import StorageLevel
from pyspark.sql import SparkSession


def main():
    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # Nice way to write a tmp file onto the system
    temp_csv_file = tempfile.mktemp()
    with open(temp_csv_file, mode="wb") as f:
        data_https = requests.get(
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        )
        f.write(data_https.content)

    titanic_df = spark.read.csv(temp_csv_file, inferSchema="true", header="true")
    titanic_df.createOrReplaceTempView("titanic")
    titanic_df.persist(StorageLevel.DISK_ONLY)

    # Simple SQL
    results = spark.sql("SELECT * FROM titanic")
    results.show()

    # More complicated queries
    results_female = spark.sql(
        """
        SELECT SUM(survived)/COUNT(*) AS survived_female_ratio
            FROM (
                SELECT
                        survived
                    FROM titanic
                    WHERE sex = 'female') AS TMP
        """
    )
    results_female.show()

    results_male = spark.sql(
        """
        SELECT SUM(survived)/COUNT(*) AS survived_male_ratio
            FROM (
                SELECT
                        survived
                    FROM titanic
                    WHERE sex = 'male') AS TMP
        """
    )
    results_male.show()

    results_overall = spark.sql(
        """
        SELECT
                sex
                , COUNT(*) AS cnt
                , AVG(survived) AS survived_ratio
            FROM titanic
            GROUP BY sex
        """
    )
    results_overall.show()
    return


if __name__ == "__main__":
    sys.exit(main())
