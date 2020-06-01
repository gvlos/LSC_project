from pyspark.sql import Row


def load_data(spark, uri_train, uri_test):
    
    # read csv
    lines_train = spark.read.text(uri_train).rdd
    lines_test = spark.read.text(uri_test).rdd

    # Remove header
    header = lines_train.first() 
    lines_train = lines_train.filter(lambda line: line != header)
    lines_test = lines_test.filter(lambda line: line != header)

    # only need ["userId", "movieId", "rating", "timestamp"]
    parts_train = lines_train.map(lambda row: row.value.split(',')[0:4])
    parts_test = lines_test.map(lambda row: row.value.split(',')[0:4])

    # create RDD with schema
    ratingsRDD_train = parts_train.map(lambda row: Row(userId=int(row[0]), movieId=int(row[1]),
                                    rating=float(row[2]), timestamp=int(row[3])))
    ratingsRDD_test = parts_test.map(lambda row: Row(userId=int(row[0]), movieId=int(row[1]),
                                    rating=float(row[2]), timestamp=int(row[3])))

    # Training - Validation split
    training_RDD, validation_RDD = split_cold_strategy(ratingsRDD_train)

    # Convert to dataframes
    training_df = training_RDD.toDF()
    validation_df = validation_RDD.toDF()
    test_df = ratingsRDD_test.toDF()

    return training_df, validation_df, test_df


def split_cold_strategy(ratingsRDD_train):

    usersWithMaxTimestamp = ratingsRDD_train \
            .map(lambda row:(row['userId'], row['timestamp'])).reduceByKey(max)
    usersWithMinTimestamp = ratingsRDD_train \
                .map(lambda row:(row['userId'], row['timestamp'])).reduceByKey(min)

    minMaxTimestampForEachUser = usersWithMinTimestamp.join(usersWithMaxTimestamp)
    training_tresh = minMaxTimestampForEachUser \
                    .map(lambda row: (row[0],float(((row[1][1] - row[1][0])/5)*4 + row[1][0])))

    training_RDD = ratingsRDD_train.map(lambda row:(row['userId'], (row['movieId'], row['rating'], row['timestamp']))) \
                        .join(training_tresh) \
                        .map(lambda row:(row[0], row[1][0][0], row[1][0][1], row[1][0][2], row[1][1])) \
                        .filter(lambda row:row[3] <= row[4]) \
                        .map(lambda row: Row(userId=int(row[0]), movieId=int(row[1]),
                                    rating=float(row[2]), timestamp=int(row[3])))

    validation_RDD = ratingsRDD_train.subtract(training_RDD)

    return training_RDD, validation_RDD





def serialize(predictions_df):
    return predictions_df.rdd.map(lambda row: (row['userId'], \
                        '|'.join([str(elem[0]) + ':' + str(round(elem[1],3)) \
                        for elem in row['recommendations']]))).\
                        toDF(['userid','recommendations'])


def save_predictions(predictions_df, hdfs_uri):
    
    pred_serialized_df = serialize(predictions_df)
    
    pred_serialized_df.write.format("csv") \
    .mode("overwrite") \
    .option("header", "true") \
    .save(hdfs_uri)