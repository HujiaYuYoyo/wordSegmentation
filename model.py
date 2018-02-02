#!/usr/bin/env python
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

class Chinese(object):
	def __init__(self, ngram):
		self.traindata = []
		self.testdata = []
		self.ngram = ngram
		self.features = []
		self.trainlabels = []
		self.testlabels = []
		self.trainmatrix = []
		self.testmatrix = []


	def _extractFeatures(self, data, ngram):
		# given text data, convert to list of n grams and extract features
		# features stored in self.features
		# keeps a pointer that keeps track of where the current moving window starts
		beginptr = 0
		words = []
		labels = []
		while beginptr < len(data):
			tempptr = beginptr
			currwords = []
			labeled = 0
			while len(currwords) < ngram:
				if tempptr < len(data):
					# if current word is not space - if the current moving window has not been labeled
					# and current pointer points to the middle of the ngram window 
					# then this word is not segmented - label is 0
					if data[tempptr] != ' ':
						if labeled == 0 and len(currwords) == ngram / 2:
							currlabel = 0
							labeled = 1
						currwords.append(data[tempptr])
					else:
						# if current word is space
						# current moving window has not been labeled and current pointer 
						# points at middle if the ngram window
						# this word is segmented - label is 1
						if labeled == 0 and len(currwords) == ngram / 2:
							currlabel = 1
							labeled = 1
					# current pointer moves forward by 1
					tempptr += 1
				else:
					# reached the end of data
					break
			# upon exiting - only add last moving window if it is a complete window of ngram
			if len(currwords) == self.ngram:
				labels.append(currlabel)
				words.append(currwords)
			assert len(words) == len(labels)
			self._convertFeatures(currwords)
			# pointer that points to the beginning of moving window increases by 1
			beginptr += 1
			# keeping incrementing moving window until the first character is not space
			while beginptr < len(data) and data[beginptr] == ' ':
				beginptr += 1
		print 'Extracted a total of %d words segments.' % len(words)
		return words, labels

	def _convertFeatures(self, currwords):
		# convert each [ABCD] to [AB, B, BC, C, CD]
		if len(currwords) == self.ngram:
			self.features.append([''.join((currwords[0], currwords[1])), currwords[1], \
								  ''.join((currwords[1], currwords[2])), currwords[2], \
								  ''.join((currwords[2], currwords[3]))])
		

	def _constructCSR(self):
		# constructs a csr matrix of dimension [# of ngrams, size of vocab]
		# each row is a n gram
		indices = []
		data = []
		indptr = [0]
		vocab = {}
		for d in self.features:
			for term in d:
				index = vocab.setdefault(term, len(vocab))
				indices.append(index)
				data.append(1)
			indptr.append(len(indices))

		res = csr_matrix((data, indices, indptr), dtype = int).toarray()
		assert np.sum(res) == 5 * len(res)
		print 'Finished constructing CSR matrix ... \nIt is a matrix of {}.'.format(res.shape)
		self.trainmatrix = res[:len(self.traindata)]
		self.testmatrix = res[len(self.traindata):]
		assert len(self.trainmatrix) == len(self.trainlabels)
		assert len(self.testmatrix) == len(self.testlabels)
		return res


	def _LogisticReg(self, max_iteration = 10, penalty = 'l1', solver = 'saga'):
		lr = LogisticRegression(solver = solver, 
		                        multi_class= 'multinomial', 
		                        C=1,
		                        penalty = penalty,
		                        fit_intercept=True,
		                        max_iter=max_iteration,
		                        random_state=42, 
		                       )
		lr.fit(self.trainmatrix, self.trainlabels)

		y_pred = lr.predict(self.testmatrix)
		return np.sum(y_pred == self.testlabels) *1./ self.testmatrix.shape[0]


	def _NBMultinomial(self):

		clf = MultinomialNB().fit(self.trainmatrix, self.trainlabels)
		predicted = clf.predict(self.testmatrix)
		return np.mean(predicted == self.testlabels)


	def _LinearSVM(self, loss = 'hinge', penalty = 'l2', max_iter = 10):
		
		clf = SGDClassifier(loss=loss, penalty=penalty,
		              alpha=1e-3, random_state=42,
		              max_iter=max_iter, tol=None).fit(self.trainmatrix, self.trainlabels)
		predicted = clf.predict(self.testmatrix)
		return np.mean(predicted == self.testlabels)
