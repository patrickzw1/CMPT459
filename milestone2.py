import json
import math
import time
import csv
import graphviz 
import numpy as np
import pandas as pd
from io import BytesIO
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Count the number of values occurrence in dictionary
# Input type: dic. Example: price_data.values()
# Output type: dic
def count_values(value):
	result = {}
	for i in value:
		result[i] = result.get(i, 0) + 1
	return result

# Take the hour out from created column
# Input type: dic. Example: data.get("created", "")
# Output type: list
def extrac_hour(data):
	time = []
	hour = []
	values = data.values()
	for i in values:
		time.append(i.split(' ')[1])
	for i in time:
		hour.append(i.split(':')[0])
	return hour

# Count the same address and use the index in the list for the category
# Input type: data -> train data or test data, column -> column name, df -> train data frame or test data frame
# Output type: list
def calculate_address_category(data, column, df):
	address = data.get(column, '')
	count = count_values(address.values())
	
	# After count, put all the address into list
	# get a list of distinct addresses
	# Ex: ['Suffolk Street', 'Thompson Street' ...]
	address_collection = []
	for key, value in count.items():
		address_collection.append(key)

	# Get the address from data frame and 
	# compare to address_collection to get the 
	# index and the index is the category
	# Ex: [0, 1, 2, 1, 0, 500 ...]
	address_category = []
	for i in df[column]:
		address_category.append(address_collection.index(i))

	return address_category

def listToString(s):  
	# initialize an empty string 
	str1 = ""    
	# traverse in the string   
	for ele in s: 
		str1 += " "
		str1 += ele   
	# return string   
	return str1

def count_feature_number(feature_names):
	def f2(s):
		result_list = []
		for ele in feature_names:
			if ele in s:
				result_list.append(1)
			else:
				result_list.append(0)
		return result_list
	result = f2
	return result

def count_feature(data):
	corpus = data['features'].values
	with open("decision_tree_text_features.txt", "r") as f:
		feature_names = f.read().splitlines()

	for i in range(len(feature_names)):
		feature_names[i] = feature_names[i].capitalize()

	count_f = count_feature_number(feature_names) 
	feature_result = list(map(count_f, corpus))

	count_feature = []
	for i in feature_result:
		count_feature.append(i[0])
	return count_feature

def x_ysquare(data,value):
	square = []
	for i in data:
		square.append((i-value)**2)
	return square
def distance(data1,data2):
	f_distance = []
	for x in range(len(data1)):
		f_distance.append(math.sqrt(data1[x]+data2[x]))
	return f_distance

def decision_tree():
	print ("Running Decision Tree...\n")
	with open('test.json') as json_file:
		test_data = json.load(json_file)

	with open('train.json') as json_file:
		data = json.load(json_file)
		
		data_frame = pd.DataFrame(data)
		test_data_frame = pd.DataFrame(test_data)

		# Count the number of photos in the training set
		count_photo = []
		for i in data_frame['photos']:
			count_photo.append(len(i))
		# Count the number of photos in the test set
		count_test_photo = []
		for i in test_data_frame['photos']:
			count_test_photo.append(len(i))
		# Append the data as columns into data_frame
		data_frame['count_photos'] = count_photo
		test_data_frame['count_photos'] = count_test_photo

		# Append the address as columns into data_frame
		data_frame['address_category'] = calculate_address_category(data, 'display_address', data_frame)
		test_data_frame['address_category'] = calculate_address_category(test_data, 'display_address', test_data_frame)

		# Extracting hour and append them into data frame
		hour = extrac_hour(data.get("created", ""))
		test_hour = extrac_hour(test_data.get("created", ""))
		data_frame['hour'] = hour
		test_data_frame['hour'] = test_hour

		# Append the features word counting as columns into data_frame
		data_frame['count_feature'] = count_feature(data_frame)
		test_data_frame['count_feature'] = count_feature(test_data_frame)

		# Get the target and features
		target_values = data_frame['interest_level'].values.reshape(-1, 1)
		features = ['price', 'address_category', 'hour', 'latitude', 'longitude', 'bathrooms', 'count_photos', 'count_feature']

		# Set up the classifier
		clf = DecisionTreeClassifier(min_samples_split=5, max_depth=6)
		# Using cross_val_score to calculate the score of the training set
		score = cross_val_score(clf, data_frame[features], target_values, cv=10)
		
		print ("The score of the training set is: \n", score)
		print ("Mean of score: ", score.mean())
		print ("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

		# Train the model
		clf.fit(data_frame[features], target_values)

		# Get the predict probability of test set
		predict = clf.predict_proba(test_data_frame[features])

		# Put the prediction into data frame with listing id
		# Export it to csv file for submitting to Kaggle
		result = pd.DataFrame({"listing_id": test_data_frame["listing_id"].values, "high": predict[:, 0], "medium": predict[:, 2], "low": predict[:, 1]})
		result.to_csv('decision_tree_result.csv', index=False)
		print ("\nFile: decision_tree_result.csv generated.")
		# Draw the decision tree into pdf
		dot_data = tree.export_graphviz(clf, out_file=None,
										filled=True, rounded=True,
										special_characters=True)
		graph = graphviz.Source(dot_data)
		graph.render("Decision Tree")
		print ("\nFile: Decision Tree.pdf generated.")

def SVM():
	print ('\nRunning SVM...\n')
	#open tain.jason
	with open('train.json') as json_file:
		data = json.load(json_file)
		data_frame = pd.DataFrame(data)

		print ("Extracting features from text...\n")

		corpus = data_frame['features'].values
		with open("text_features.txt", "r") as f:
			feature_names = f.read().splitlines()

		for i in range(len(feature_names)):
			feature_names[i] = feature_names[i].capitalize()

		# print(corpus)
		# print(feature_names)
		count_f = count_feature_number(feature_names) 
		result = list(map(count_f, corpus))
		# print(result)

		myfeatures = result
		mytarget = data_frame['interest_level'].values
		# X_train, X_test, y_train, y_test = train_test_split(myfeatures, mytarget, test_size = 0.2)
		clf = svm.SVC(kernel = 'linear')
		clf.fit(myfeatures, mytarget)
		y_pred = clf.predict(myfeatures)
		print(y_pred)

		print("Accuracy:",metrics.accuracy_score(mytarget, y_pred))

		#start cross validation by k-fold
		from sklearn.model_selection import KFold # import KFold
		X = np.array(result) # create an array
		y = np.array(mytarget) # Create another array
		kf = KFold(n_splits=5) # Define the split - into 2 folds 
		kf.get_n_splits(myfeatures) # returns the number of splitting iterations in the cross-validator
		clf2 = svm.SVC(kernel='linear',probability=True) # Linear Kernel
		scores = []
		proba = []
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			#Create a svm Classifier

			#Train the model using the training sets
			clf2.fit(X_train, y_train)

			#Predict the response for test dataset
			y_pred = clf2.predict(X_test)
			scores.append(metrics.accuracy_score(y_test, y_pred))
			
			y_prob = clf2.predict_proba(myfeatures)
			proba.append(y_prob)
		print("done")
			
	#test the model by test data
	with open('test.json') as json_file:
		data_test = json.load(json_file)
		data_frame_test = pd.DataFrame(data_test)

		corpus_test = data_frame_test['features'].values
		with open("text_features.txt", "r") as f:
			feature_names_test = f.read().splitlines()

		for i in range(len(feature_names_test)):
			feature_names_test[i] = feature_names_test[i].capitalize()

		# print(corpus)
		# print(feature_names)
		count_f_test = count_feature_number(feature_names_test) 
		result_test = list(map(count_f_test, corpus_test))

		myfeatures_test = result_test
		y_pred_test = clf2.predict(myfeatures_test)
	
		y_prob_test = clf2.predict_proba(myfeatures_test)

		y_prob_test2 = y_prob_test
		#y_prob_test
		y_prob_test = []
		for i in range(len(y_prob_test2)):
			temp = []
			temp.append(y_prob_test2[i][0])
			temp.append(y_prob_test2[i][2])   
			temp.append(y_prob_test2[i][1])
			y_prob_test.append(temp)

		listing_ids_test = list(data_frame_test['listing_id'].values)
		i_test = 0
		output_test = []
		for i in range(len(y_prob_test)):
			temp_test = []
			temp_test.append(listing_ids_test[i])

			temp_test.extend(y_prob_test[i])
			output_test.append(temp_test)
			i_test = i_test + 1
			
		with open('SVM_result.csv', 'w', newline='') as mycsvfile:
			thedatawriter = csv.writer(mycsvfile, delimiter=',', lineterminator = '\n')
			thedatawriter.writerows(output_test)      
		with open('SVM_result.csv',newline='') as f:
			r = csv.reader(f)
			data = [line for line in r]
		with open('SVM_result.csv','w',newline='') as f:
			w = csv.writer(f)
			w.writerow(['listing_id', 'high', 'medium', 'low'])
			w.writerows(data)

def logistic_regression():
	print ("\nRunning Logistic Regression...\n")
	with open('test.json') as json_file:
		test_data = json.load(json_file)

	with open('train.json') as json_file:
		data = json.load(json_file)

		data_frame = pd.DataFrame(data)
		test_data_frame = pd.DataFrame(test_data)

		city_latitude = 40.730610
		city_longitude = -73.935242

		x_square = x_ysquare(data_frame['latitude'],city_latitude)
		y_square = x_ysquare(data_frame['longitude'],city_longitude)

		x_square_test = x_ysquare(test_data_frame['latitude'],city_latitude)
		y_square_test = x_ysquare(test_data_frame['longitude'],city_longitude)
		
		# print (max(calculate_address_category(data, 'display_address', data_frame)))

		data_frame['address_category'] = calculate_address_category(data, 'display_address', data_frame)
		test_data_frame['address_category'] = calculate_address_category(test_data, 'display_address', test_data_frame)

		data_frame['center_distance'] = distance(x_square,y_square)
		test_data_frame['center_distance'] = distance(x_square_test,y_square_test)

		hour = extrac_hour(data.get("created", ""))
		test_hour = extrac_hour(test_data.get("created", ""))
		data_frame['hour'] = hour
		test_data_frame['hour'] = test_hour

		target_values = data_frame["interest_level"].values.ravel()
		features = ['price', 'address_category', 'hour', 'latitude', 'longitude', 'bathrooms','center_distance']
		# kf = KFold(n_splits=10)

		lr = LogisticRegression(multi_class ='multinomial', max_iter=5000)
		lr.fit(data_frame[features], target_values)

		score = cross_val_score(lr, data_frame[features], target_values, cv=10)
		
		print (score)
		print (score.mean())

		print("=================")

		accuracy = cross_val_score(lr, data_frame[features], target_values, cv=10,scoring='balanced_accuracy')
		print(accuracy)
		print (accuracy.mean())

		predict = lr.predict_proba(test_data_frame[features])

		result = pd.DataFrame({"listing_id": test_data_frame["listing_id"].values, "high": predict[:, 0], "medium": predict[:, 2], "low": predict[:, 1]})
		result.to_csv('logistic_regression_result.csv', index=False)

if __name__ == "__main__":
	decision_tree()
	logistic_regression()
	SVM()
