import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from spacy.util import minibatch
import random

def pre_processing(df, stop_words):
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    df['text'] = df['text'].str.replace('\d', '')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words)) 
    df['text'] = df['text'].apply(lambda x: " ".join([Word(x).lemmatize() for x in x.split()]))
    return df
	
def prediction(nlp_model, test_data):
	docs = [nlp_model.tokenizer(text) for text in test_data]
	textcat = nlp_model.get_pipe('textcat')
	scores, _ = textcat.predict(docs)
	predicted_labels = scores.argmax(axis=1)
	prediction_result = [textcat.labels[label] for label in predicted_labels]
	return prediction_result
	
def train_model(nlp_model, train_data):
	random.seed(1)
	spacy.util.fix_random_seed(1)
	nlp_optimizer = nlp_model.begin_training()
	losses = {}
	for epoch in range(10):
		random.shuffle(train_data)
		batches = minibatch(train_data, size=8)
		for batch in batches:
			texts, labels = zip(*batch)
			nlp_model.update(texts, labels, sgd=nlp_optimizer, losses=losses)
			
def main():
	#Loading train and test data	
	dataset_train = pd.read_csv('G:\machine_learning\sentiment\Train.csv')
	dataset_test = pd.read_csv('G:\machine_learning\sentiment\Test.csv')

	stop_words = stopwords.words('english')

	#preparing train and test dataset
	train_nlp = pre_processing(dataset_train,stop_words)
	test_nlp = pre_processing(dataset_test, stop_words)

	train_nlp['label'] = np.where(train_nlp['label'] == 0, "negative", "positive")
	test_nlp['label'] = np.where(test_nlp['label'] == 0, "negative", "positive") 

	train_text  = train_nlp['text'].values
	train_labels = [{'cats': {'negative': label == 'negative',
							  'positive': label == 'positive'}} 
					for label in train_nlp['label']]

	train_data = list(zip(train_text, train_labels))

	nlp_model = spacy.blank('en')
	text_cat = nlp_model.create_pipe("textcat", config={"exclusive_classes": True, "architecture": "bow"})
	nlp_model.add_pipe(text_cat)

	text_cat.add_label("negative")
	text_cat.add_label("positive")
	test_data = list(test_nlp['text'])
	
	#Training the Model
	train_model(nlp_model, train_data)
	
	#Predicting results for test data
	prediction_result = prediction(nlp_model, test_data)
	
	test_data_labels = list(test_nlp['label'])
	accr_score = accuracy_score(test_data_labels,prediction_result)
	print("Accuracy score of the model:",accr_score)

if __name__ == "__main__":
    main()
