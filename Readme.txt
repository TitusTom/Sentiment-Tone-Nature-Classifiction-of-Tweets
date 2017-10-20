Readme for data_cleaning_and_evaluation.py script
-------------------------------------------------

Files included
---------------  
1.data_cleaning_and_evaluation.py 
2.pull_twitter_data.py


Requriements
------------
Recommended package: Anaconda 2(or Anaconda 3 with python2 enviorment)
link:https://www.continuum.io/downloads

1. Python 2.7+
   Python 2.7 was used for devolopment and testing for    data_cleaning_and_evaluation.py .
2.Tweepy library for python(http://www.tweepy.org/)
3.Numpy library for python(http://www.scipy.org/scipylib/download.html)
4.ScikitLearn library for python(http://scikit-learn.org/stable/install.html)
5.Nltk for python(http://www.nltk.org/install.html),Stopword Corpus also required


Working
------------
The data_cleaning_and_evaluation.py script requires one of the csv files(2-4) in the same directory to work.

tweet_processed_list function is used to preprocess the tweets and prepare it for the classifictaion.

write_processed is used to save the tweets after the preprocessing step.

The evaluate_tweet function uses SVMS to predict the label and outputs the confusion matrix and accuracy of the prediction.modify argument 2 to change the label used as ground truth.

As is the pyton script uses the "Is racist?" feature as the ground truth for the SVM.