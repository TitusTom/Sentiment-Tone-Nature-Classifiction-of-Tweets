import re
import pandas as pd
from sklearn import svm
from string import punctuation
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#globals
gt_filename = "input_file.csv"
#Gt label : 0 - tone, 1 - nature 2 - sentiment.
gt_flag = 0

# Load captions and file names.
def gtFileLoad(gt_filename):
    ground_truth_file = pd.read_csv(gt_filename, names \
    =['tweets', 'tone', 'nature', 'sentiment'], skiprows=1)
    ground_truth_file = ground_truth_file.convert_objects(convert_numeric=True)
    print 'Ground truth file loaded'
    return ground_truth_file

#function to create list of unique words and its frequencies                                                                                                                        
def createWordDictionary(x_data):                                                                                       
    word_dictionary = dict()
    for tweet in x_data:
        words = tweet.split(' ')
        for word in words:
            if word != '':
                if word in word_dictionary:
                    word_dictionary[word]+=1
                else:
                    word_dictionary[word]=1
    return  word_dictionary

#preprocess list of tweets
def preporcess(tweet_list):
    stopwords_list  = stopwords.words('english')

    tweet_list_processed =[]
    for tweet in tweet_list:
        # Normalizing case.
        tweet_process=tweet.lower()
        # Removing punctiation chars.
        for p in list(punctuation):                                                                                     
            tweet_process=tweet_process.replace(p,'')

        words = tweet_process.split(' ')
        tweet_processed =''

        #Process words in tweet. 
        for word in words:                                                                                              
            # Removing unicode chars.
            if "\u" in word:                                                                                            
                word.replace(word,'')
            # remove RT
            if word =='rt':
                word = word.replace (word ,'')                                                                        
            # Starting lemmatizer"
            # lematizies if verb.
            w1=WordNetLemmatizer()                                                                                      
            l = w1.lemmatize(word,'v')                                                                                  
            # lematizies if Noun.
            if l==word:
                l = w1.lemmatize(word,'n')                                                                              
            word = l
            # Removing weblinks
            if len(re.findall("htt",word)) >0:
                word = word.replace(word,'')
            # removing Stopwords.   
            if word in stopwords_list:                                                                                  
                word = word.replace(word,'')
            tweet_processed+=word
            tweet_processed+=' '

        tweet_list_processed.append(tweet_processed)
    print "Tweets Processed!"
    return tweet_list_processed

def classifyTweets(ground_truth_file, gt_flag):
    
    #load Gt and Labels
    tweet_list_l = ground_truth_file['tweets']
    if gt_flag == 0:
        tweet_gt_v = ground_truth_file['tone']           
    elif gt_flag == 1:
        tweet_gt_v = ground_truth_file['nature']     
    else: 
        tweet_gt_v = ground_truth_file['sentiment']   
    tweet_train, tweet_test,GT_train,GT_test= train_test_split(tweet_list_l,tweet_gt_v, test_size=0.30)
    TF_V = TfidfVectorizer(ngram_range=(1,1))

    classifiersvm = svm.SVC(kernel='rbf',gamma=1,C=2)
    c_svm=Pipeline([('C', TF_V), ('svc', classifiersvm)])
    c_svm.fit(tweet_train,GT_train)
    result=c_svm.predict(tweet_test)

    print('Accuracy:')
    print(accuracy_score(GT_test,result))
    print('Confusion matrix:')
    print(confusion_matrix(GT_test,result))
    print('Classification Report:')
    print(classification_report(GT_test,result))
    

print "Hello I am a simple script that uses cleans and classifies tweets labeled for Tone, Nature or Sentiment."
print "========================================================================================================="
if gt_flag == 0:
    print 'Classfying tweet tone'
elif gt_flag == 1:
    print 'Classfying tweet nature'
else: 
    print 'Classfying tweet sentiment'   
ground_truth_file = gtFileLoad(gt_filename)
print"========================================================================================================="


print "Begin preprocessing tweets"
print "========================================================================================================="
ground_truth_file['tweets'] = preporcess(ground_truth_file['tweets'])                                                                           
print "========================================================================================================="
print "Begin classifiaction"
print "========================================================================================================="
classifyTweets(ground_truth_file, gt_flag)
print "========================================================================================================="
print "script complete"                                                                       


