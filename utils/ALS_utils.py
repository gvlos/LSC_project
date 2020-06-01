from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pyspark.sql.functions as func
import numpy as np

def remove_bias(predictions, validation):
    """
    Remove bias in ALS predictions (as suggested in the paper)
    Shift each predicted rating by the difference between the mean
    of ratings in the validation set and the mean of predicted ratings 
    
    Input:
    recs = predicted recommendation rdd (output of ALS_model.transform(validation))
    validation = validation set rdd
    
    Output:
    recs_shifted = predicted reccomendation rdd with shifted ratings
    """
    
    mean_valid = float(validation.select("rating").summary("mean").first()[1])
    mean_pred = float(predictions.select("prediction").summary("mean").first()[1])
    
    tau = mean_valid - mean_pred
    
    if tau != 0:
        preds_shifted = predictions.withColumn('prediction', predictions.prediction + tau)
    else:
        preds_shifted = predictions

    return preds_shifted



def holdout_cv_ALS(train_data, validation_data, num_iters, reg_params, ranks, file):
    """
    Hold-out cross validation for ALS.
    Parameters evaluated: reg_param, ranks
    
    Input:
    train_data
    validation_data
    num_iters
    reg_param = list of lambdas
    ranks = list of number of features (p)
    
    Output:
    best_model = best ALS model
    errs = rmse evolution -> (i,j) = lambda,p)
    
    """
    # initial
    min_rmse = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    
    errors = np.zeros((len(reg_params),len(ranks)))
        
    # Outer loop: try different number of features (the most relevant hyperparameter to tune
    # which highly affects the performance of the methods in both time/space complexity
    # and accuracy)
    for i, reg in enumerate(reg_params):
        
        # Inner loop: try different lambda
        for j, rank in enumerate(ranks):
            
            # create model
            als = ALS(maxIter=num_iters, regParam=reg, rank=rank, userCol="userId",
                      itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

            # fit
            model = als.fit(train_data)
            
            # make prediction
            predictions = model.transform(validation_data)
            
            # remove bias
            preds_shifted = remove_bias(predictions, validation_data)
            
            # get the RMSE
            rmse = evaluator.evaluate(preds_shifted)

            file.write('{} latent factors and regularization = {}: validation RMSE is {} \n'.format(rank, reg, rmse))
            
            errors[i,j] = rmse
            
            if rmse < min_rmse:
                min_rmse = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    
    file.write('\nThe best model has {} latent factors and regularization = {} \n'.format(best_rank, best_regularization))
    return best_model, errors



def predict(ALS_model, num_movies):
    return ALS_model.recommendForAllUsers(num_movies)


def compute_rmse(ALS_model, test_df):
    
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    
    predictions = ALS_model.transform(test_df)
    
    # remove bias
    preds_shifted = remove_bias(predictions, test_df)
            
    # get the RMSE
    return evaluator.evaluate(preds_shifted)
    
    
    
def compute_accuracy(predictions_df, test_df):
    
    liked_df = test_df.filter(test_df.rating >= 3).groupBy('userId') \
                .agg(func.collect_list('movieId').alias('liked'))
    
    top_rec_df = predictions_df.rdd.map(lambda row: \
                               [row['userId'], [elem[0] for elem in row['recommendations']]]) \
                                .toDF(['userId','top_recommendation'])
    
    accuracy_df = top_rec_df.join(liked_df,'userId').rdd \
                        .map(lambda row: [row['userId'], 
                          len(list(set(row['top_recommendation']) & set(row['liked']))) \
                          /min(len(row['top_recommendation']),len(row['liked']))]) \
                        .toDF(['userId','accuracy'])
    
    
    return accuracy_df.select("accuracy").summary("mean").first()[1]