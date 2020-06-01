# Script Executed on the Cluster
# ALS Cross validation on MovieLens Dataset

# author: Gianvito Losapio


from pyspark.sql import SparkSession

import time

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pyspark.sql.functions as func
import numpy as np

from pyspark.sql import Row

# Custom import
from utils.ALS_utils import *
from utils.io import *


# -----------------------------------------------

if __name__ == "__main__":
    
    # SPARK SESSION
    spark = SparkSession \
        .builder \
        .appName("ALS") \
        .getOrCreate()


    # INPUT CSV HDFS PATH
    HADOOP_PATH = 'hdfs:///home/user30/Exam/datasets/'

    uri_train = HADOOP_PATH + "training.csv"
    uri_test = HADOOP_PATH + "test.csv"

    
    # OUTPUT CSV with top10 recommendations for each user
    uri_csv = 'hdfs:///home/user30/Exam/outputs/als.csv'

    
    # TEXT FILE where performance results will be written
    uri_text = "/home/user30/Exam/outputs/als_results.txt"
    f = open(uri_text,"w")

    # Load and split
    training_df, validation_df, test_df = load_data(spark, uri_train, uri_test)
    
    
    # CROSS-VALIDATION
    num_iterations = 10
    ranks = [8, 14, 20, 50]
    reg_params = [0.001, 0.01, 0.1]


    # grid search and select best model
    f.write('-------------------------------------------\n')
    f.write('Starting Hold-out Cross Validation ...\n')
    start_time = time.time()
    best_model, errors = holdout_cv_ALS(training_df, validation_df, num_iterations, reg_params, ranks, f)

    f.write('Total Runtime: {:.2f} seconds \n'.format(time.time() - start_time))


    # TODO: SAVE CHECKPOINTS
    #
    # ..................


    # Predictions
    num_movies = 10
    predictions_df = predict(best_model, num_movies).cache()


    # RMSE, Accuracy
    rmse_val = compute_rmse(best_model, validation_df)
    rmse_test = compute_rmse(best_model, test_df)
    accuracy_val = compute_accuracy(predictions_df, validation_df)
    accuracy_test = compute_accuracy(predictions_df, test_df)
    f.write('Validation RMSE: {} \n'.format(rmse_val))
    f.write('Test RMSE: {} \n'.format(rmse_test))
    f.write('Validation Accuracy: {} \n'.format(accuracy_val))
    f.write('Test Accuracy: {} \n'.format(accuracy_test))

    f.close()

    print("Cross-Validation completed successfully")

    # Save
    save_predictions(predictions_df, uri_csv)
    print("Predictions saved. SUCCESS")