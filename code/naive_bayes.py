import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB

def pre_processing(df, stop_words):
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    df['text'] = df['text'].str.replace('\d', '')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words)) 
    df['text'] = df['text'].apply(lambda x: " ".join([Word(x).lemmatize() for x in x.split()]))
    return df

def main():	
	#Loading train and test data	
	dataset_train = pd.read_csv('G:\machine_learning\sentiment\Train.csv')
	dataset_test = pd.read_csv('G:\machine_learning\sentiment\Test.csv')

	stop_words = stopwords.words('english')

	#preparing train and test dataset
	dataset_train_v2 = pre_processing(dataset_train,stop_words)
	dataset_test_v2 = pre_processing(dataset_test, stop_words)

	train_reviews = dataset_train_v2.text[:]
	train_sentiments = dataset_train_v2.label[:]
	test_reviews = dataset_test_v2.text[:]
	test_sentiments = dataset_test_v2.label[:]

	#Encoding the text data
	vec = TfidfVectorizer(min_df = 4, max_df = 0.9)
	train_vec = vec.fit_transform(train_reviews)
	test_vec = vec.transform(test_reviews)

	#Initializing and fitting data to SVM model
	mnb_model = MultinomialNB()
	mnb_model.fit(train_vec,train_sentiments)
	prediction_mnb = mnb_model.predict(test_vec)

	#Printing the accuracy score
	accr_score = accuracy_score(test_sentiments,prediction_mnb)
	print("Accuracy score of the model:",accr_score)

	#Printing the confusion matrix
	mnb_report=classification_report(test_sentiments,prediction_mnb,target_names=['Positive','Negative'])
	print(mnb_report)

if __name__ == "__main__":
    main()