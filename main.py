import tensorflow as tf

from sentiment_analysis.dnn import sa_dnn_runner, get_prediction_from_tweet
from sentiment_analysis.sa_scratch_rnn import sa_scratch_rnn_running


if __name__ == '__main__':
    """
    Training sentiment analysis by DNN
    """
    # sa_dnn_runner()
    
    """
    Infer sentiment analysis by DNN
    """
    # model = tf.keras.models.load_model('model/SA_DNN.h5')
    # tweet = "I am very sad today"
    # prediction = get_prediction_from_tweet(tweet=tweet, model=model)
    # if prediction >= 0.5:
    #     print("POSITIVE")
    # else: print("NEGATIVE")
    
    """
    Training sentiment analysis by RNN from scratch
    """
    sa_scratch_rnn_running()
    