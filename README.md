Implementation of a simple chatbot in python:

we implement a simple chatbot using deep learning that is trained on the data of intent.json file .

Following are the steps involved in our implementation:

1) Pre-Process the input text and label into numeric form which can be feed into our deep learning model.
	* The input data for our model is the patterns in each intent in our intents.json file.
	* the label data for our model is the tag corresponding to the pattern.
	* We perform word_tokenization, stemming and one hot encoding to convert out input from text form to numeric form.
	* word_tokenization involves spliting the sentence into words and removing special characters.
	* stemming involves converting each word to into stemmed form(eg., [running, runner] -> run)
	* one-hot encoding means creating a 1/0 list for each sentence that represents the presence or absence of each word in that sentence.
	 this means if for example we have 6 unique words on including all the sentences, then each of the sentence will be represented as one-hot encoded array of length 6.
	 	
	 say the 6 unique words are ["is", "happy", "run", "difficult", "journey", "science"].

	 Then the sentence "science is difficult" will be represented as
	 	[1, 0, 0, 1, 0, 1] which can be fed into our model.

	* for labels, we consider all unique tags and for a one-hot array with 1 at the index of the intent's tag and zero on other places.

2) We build out model using the tensorflow keras api.Our model consists of three fully connected dense layers. Note that the last layer has 'softmax' activation function to output the result as probabilities for different labels.We train the model on the input data.

3) create a utility function chat() that gets an input from the user , converts the sentence to one-hot array and predict the tag that the input sentence is associated.


To run the script: 
	1) we have to install python3, tensorflow, nltk, numpy and pickle.
	2) once we have insalled the packages and have the intents.json file, run the script as
		python ChatBot.py