#!/usr/bin/env python
from model import Chinese
import utils




def _wordSeg(train, test, numlinestest = -1, numlinestrain = 100000, ngram = 4, logfile = 'hujialogs.txt'):
	traindata = utils._input(train, numlinestrain)
	testdata = utils._input(test, numlinestest)

	model = Chinese(ngram)
	model.traindata, model.trainlabels = model._extractFeatures(traindata, ngram)
	model.testdata, model.testlabels = model._extractFeatures(testdata, ngram)
	csrMatrix = model._constructCSR()

	lracc = model._LogisticReg()
	print 'logistic regression accuracy on test set is ', lracc

	nbaccu = model._NBMultinomial()
	print 'NB Multinomial accuracy on test set is ', nbaccu

	svmaccu = model._LinearSVM()
	print 'svm accuracy is on test set is ', svmaccu

	with open(logfile, 'a') as f:
		f.write('train: %d lines \ntest: %d lines \nLogistic regression accuracy on test set is %.4f\
				\nNB Multinomial accuracy on test set is %.4f\nSVM accuracy on test set is %.4f\n\n\n'\
				 % (numlinestrain, numlinestest, lracc, nbaccu, svmaccu))
	f.close()



if __name__ == '__main__':

	trainfile = 'training.txt'
	testfile = 'test.txt'
	ngram = 4
	logfile = 'hujialogs.txt'
	numLinesTrain = [1000, 3000]
	numLinesTest = [100, 300, 500]
	for numTrain in numLinesTrain:
		for numTest in numLinesTest:
			_wordSeg(trainfile, testfile, numTest, numTrain, ngram, logfile)