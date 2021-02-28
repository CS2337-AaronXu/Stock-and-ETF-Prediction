import lstm
import dataRetrieval
import predictGains
import time
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def plot_results(predicted_data, true_data, fileName):
	fig = plt.figure(facecolor='white', figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.savefig(fileName)

def plot_results_multiple(predicted_data, true_data, prediction_len):
	fig = plt.figure(facecolor='white', figsize=(30,10))
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	for i, data in enumerate(predicted_data):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data, label='Prediction')
		plt.legend()
		plt.savefig('./out/multipleResults.jpg')

def plotMetrics(history):
	losses = []
	mses = []
	for key, value in history.items():
		if(key == 'loss'):
			losses = value
	plt.figure(figsize=(6, 3))
	plt.plot(losses)
	plt.ylabel('error')
	plt.xlabel('iteration')
	plt.title('testing error over time')
	plt.savefig('losses.png')

def trainModel(newModel, epochs = 1, seq_len = 50):
	if newModel:
		global_start_time = time.time()

		print('> Data Loaded. Compiling LSTM model...')

		model = lstm.build_model([1, 50, 100, 1])

		model.save('./../model/lstm.h5')

		print('> Training duration (s) : ', time.time() - global_start_time)
	else:
		print('> Data Loaded. Loading LSTM model...')

		model = load_model('./../model/lstm.h5')

	return model

def run():
	dataRetrieval.retrieveStockData('AMZN', '2015-05-15', '2020-05-15', './data/lstm/AMZN.csv')
	dataRetrieval.retrieveStockData('GOOG', '2015-05-15', '2020-05-15', './data/lstm/GOOG.csv')
	dataRetrieval.retrieveStockData('IBM', '2015-05-15', '2020-05-15', './data/lstm/IBM.csv')
	dataRetrieval.retrieveStockData('MSFT', '2015-05-15', '2020-05-15', './data/lstm/MSFT.csv')


	stockFile = 'C:\\Users\\aaron\\PycharmProjects\\pythonProject\\data\\lstm\\AMZN.csv'
	epochs = 10
	seq_len = 100
	batch_size=512

	print('> Loading data... ')

	X_train, y_train, X_test, y_test = lstm.load_data(stockFile, seq_len, True)

	X_train = pad_sequences(X_train, maxlen=seq_len)
	X_test = pad_sequences(X_test, maxlen=seq_len)

	print('> X_train seq shape: ', X_train.shape)
	print('> X_test seq shape: ', X_test.shape)

	model = trainModel(True)

	print('> LSTM trained, Testing model on validation set... ')

	training_start_time = time.time()

	hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05, validation_data=(X_test, y_test))

	print('> Testing duration (s) : ', time.time() - training_start_time)

	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)

	print('> Plotting Losses....')
	plotMetrics(hist.history)

	print('> Plotting point by point prediction....')
	predicted = lstm.predict_point_by_point(model, X_test)

	print(predicted)
	plot_results(predicted, y_test, './out/ppResults.jpg')

	print('> Plotting full sequence prediction....')
	predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	plot_results(predicted, y_test, './out/sResults.jpg')

	print('> Plotting multiple sequence prediction....')

	predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	plot_results_multiple(predictions, y_test, 50)

if __name__=='__main__':
	run()