import copy
import sys
import time
import numpy as np
import pandas as pd

from LoadData import load
from DataPre import DataPreprocessing
from MICriterion import Mutual_Info, mRMR_sel, MaxRel_sel
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import LeaveOneOut
from two import MBmain
from third import DAGThird

firstNodesArr = []
secondNodesArr = []
recallarr = []
precisionarr = []
f1arr = []
recallfinally = 0
precisionfinally = 0
f1finally = 0
TParr = []
FParr = []
FNarr = []

for index in range(100):
	df2 = np.loadtxt(open("xxx.csv", "rb"), delimiter=",")
	num_cols = df2.shape[1]
	random_col_index = np.random.choice(range(num_cols))
	df2[:, [0, random_col_index]] = df2[:, [random_col_index, 0]]

	# MBSecond
	second_Columns = MBmain(df2, 8)

	if len(second_Columns) == 0:
		ans = set()
	else:
		second_Columns.append(0)
		second_Columns.sort()
		secondDatasets = df2[:, second_Columns]

		# DAGThird
		DAGMatrix = DAGThird(secondDatasets)

		num_nodes = len(DAGMatrix[0])
		ans = set()
		for i in range(num_nodes):
			if DAGMatrix[0][i] != 0 and i != 0:
				ans.add(i)
				for j in range(num_nodes):
					if DAGMatrix[j][i] != 0 and j != 0:
						ans.add(j)
			if DAGMatrix[i][0] != 0 and i != 0:
				ans.add(i)

		original_indices = []
		for cols in range(8):
			original_indices.append(cols)
		new_indices = list(ans)
		matched_indices = [original_indices[index] for index in new_indices]

	dataSetslink = np.loadtxt(open("./skeleton_asia8.csv", "rb"), delimiter=",")
	dataSetslink[:, [0, random_col_index]] = dataSetslink[:, [random_col_index, 0]]
	datalink = dataSetslink

	num_nodes2 = len(datalink[0])
	ans2 = set()
	for i in range(num_nodes2):
		if datalink[0][i] != 0 and i != 0:
			ans2.add(i)
			for j in range(num_nodes2):
				if datalink[j][i] != 0 and j != 0:
					ans2.add(j)
		if datalink[i][0] != 0 and i != 0:
			ans2.add(i)

	ans = set(matched_indices)

	TP = len(ans.intersection(ans2))
	FP = len(ans.difference(ans2))
	FN = len(ans2.difference(ans))
	if (TP + FN) == 0:
		recall = 0
	else:
		recall = TP / (TP + FN)

	if (TP + FP) == 0:
		precision = 0
	else:
		precision = TP / (TP + FP)

	if (precision + recall == 0):
		f1 = 0
	else:
		f1 = 2 * (precision * recall) / (precision + recall)

	recallarr.append(recall)
	precisionarr.append(precision)
	f1arr.append(f1)

	recallfinally += recall
	precisionfinally += precision
	f1finally += f1

	firstNodesArr.append(len(ans))
	secondNodesArr.append(len(ans2))
	TParr.append(TP)
	FParr.append(FP)
	FNarr.append(FN)

print("recall", recallfinally / 100)
print("precision", precisionfinally / 100)
print("f1", f1finally / 100)

