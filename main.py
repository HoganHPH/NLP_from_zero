import tensorflow as tf

from sentiment_analysis.dnn import sa_dnn_runner, get_prediction_from_tweet



if __name__ == '__main__':
    # sa_dnn_runner()
    model = tf.keras.models.load_model('model/SA_DNN.h5')
    tweet = "I am very sad today"
    prediction = get_prediction_from_tweet(tweet=tweet, model=model)
    if prediction >= 0.5:
        print("POSITIVE")
    else: print("NEGATIVE")
    