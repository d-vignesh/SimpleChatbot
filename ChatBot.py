import json
import pickle
import random

import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np 

import tensorflow as tf 
from keras import models
from keras import layers

nltk.download("punkt")
stemmer = LancasterStemmer()

with open("intents.json") as file:
		data = json.load(file)

try:
	with open("data.pickle", "rb") as file:
		words, labels, training, output = pickle.load(f)
except:

	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	training = []
	output = []

	for idx, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = [0 for _ in range(len(labels))]
		output_row[labels.index(docs_y[idx])] = 1

		training.append(bag)
		output.append(output_row)

	training = np.array(training)
	output = np.array(output)

	with open("data.pickle", "wb") as file:
		pickle.dump((words, labels, training, output), file)


try:
	model = models.load_model("chatbot_model.h5")
except Exception as e:
	model = models.Sequential([
			layers.Dense(8, input_shape=(len(training[0]),)),
			layers.Dense(8, activation="relu"),
			layers.Dense(len(labels), activation="softmax")
		])

	model.compile(
		optimizer = "adam",
		loss = "categorical_crossentropy",
		metrics = ["accuracy"]
		)

	model.fit(training, output, epochs=100, batch_size=8, verbose=1)
	model.save("chatbot_model.h5")

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(w.lower()) for w in s_words]

	for w in s_words:
		if w in words:
			bag[words.index(w)] = 1

	return np.array(bag).reshape(1, 46)

def chat():
	print("start talking with the bot (type q to stop)")
	while True:
		inp = input("you : ")
		if inp.lower() == "q":
			break

		results = model.predict(bag_of_words(inp, words))
		result_idx = np.argmax(results)
		tag = labels[result_idx]

		for intent in data["intents"]:
			if tag == intent["tag"]:
				responses = intent["responses"]

		print(random.choice(responses))
		

chat()