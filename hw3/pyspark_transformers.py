import sys
import tempfile

import requests
from pyspark import StorageLevel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import CountVectorizer
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
    titanic_df.persist(StorageLevel.MEMORY_ONLY)

    # Create a column that has all the "words" we want to encode for modeling
    # Each word must be in an array (hence SPLIT)
    titanic_df = spark.sql(
        """
        SELECT
                *
                , SPLIT(CONCAT(
                    CASE WHEN class IS NULL THEN ""
                    ELSE class END,
                    " ",
                    CASE WHEN sex IS NULL THEN ""
                    ELSE sex END,
                    " ",
                    CASE WHEN embarked IS NULL THEN ""
                    ELSE embarked END,
                    " ",
                    CASE WHEN who IS NULL THEN ""
                    ELSE who END
                ), " ") AS categorical
            FROM titanic
        """
    )
    titanic_df.show()

    # Count Vectorizer
    count_vectorizer = CountVectorizer(
        inputCol="categorical", outputCol="categorical_vector"
    )
    count_vectorizer_fitted = count_vectorizer.fit(titanic_df)
    titanic_df = count_vectorizer_fitted.transform(titanic_df)
    titanic_df.show()

    # Random Forest
    random_forest = RandomForestClassifier(
        labelCol="survived",
        featuresCol="categorical_vector",
        numTrees=100,
        predictionCol="pred_survived",
        probabilityCol="prob_survived",
        rawPredictionCol="raw_pred_survived",
    )
    random_forest_fitted = random_forest.fit(titanic_df)
    titanic_df = random_forest_fitted.transform(titanic_df)
    titanic_df.show()

    return


if __name__ == "__main__":
    sys.exit(main())
