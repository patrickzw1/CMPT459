import json
import math
import time
import csv
import numpy as np
import pandas as pd
from io import BytesIO
from sklearn import tree
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Count the number of values occurrence in dictionary
# Input type: dic. Example: price_data.values()
# Output type: dic
def count_values(value):
	result = {}
	for i in value:
		result[i] = result.get(i, 0) + 1
	return result

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

#calculate the bounds of the outliers
def get_bounds(variable):
	values = list(variable.values())
	values.sort()
	length_p = len(values)
	inedx_Q1 = int((length_p - 1)/4)
	Q1 = (values[inedx_Q1] + values[inedx_Q1 + 1])/2
	inedx_Q3 = int((length_p - 1)*3/4)
	Q3 = (values[inedx_Q3] + values[inedx_Q3 + 1])/2
	IQR = Q3 - Q1
	lower_bound = Q1 - 1.5*IQR             #100
	upper_bound = Q3 + 1.5*IQR
	return lower_bound, upper_bound	

# Replace the outlier in price with upper_bound
# Input: data frame and data
# Output: a column of price with outlier replaced
def clear_outlier_of_pirce(data_frame, data):
	lower_bound, upper_bound = get_bounds(data['price'])
	price = data_frame['price']
	median = price.median()
	for i in price:
		if i > upper_bound:
			data_frame['price'] = data_frame['price'].replace(i, upper_bound)

	return data_frame['price']

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

def deal_price(data):
	price = []
	for i in data:
		if i > 100000:
			price.append(data.mean())
		else:
			price.append(i)
	return price

def count_features(data):
	numner = []
	for i in data:
		numner.append(len(i))
	return numner;

def listToString(s):  
    # initialize an empty string 
    str1 = ""    
    # traverse in the string   
    for ele in s: 
        str1 += " "
        str1 += ele   
    # return string   
    return str1

def random_forest():
	print ("Running Random Forest...\n")
	with open('test.json') as json_file:
		test_data = json.load(json_file)

	with open('train.json') as json_file:
		data = json.load(json_file)
		
		data_frame = pd.DataFrame(data)
		test_data_frame = pd.DataFrame(test_data)

		# Clear the outliers in price
		data_frame['new_price'] = clear_outlier_of_pirce(data_frame, data)
		test_data_frame['new_price'] = clear_outlier_of_pirce(test_data_frame, data)

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

		# Count the number of words in description
		data_frame["num_description_words"] = data_frame["description"].apply(lambda x: len(x.split(" ")))
		test_data_frame["num_description_words"] = test_data_frame["description"].apply(lambda x: len(x.split(" ")))

		# Extracting hour and append them into data frame
		hour = extrac_hour(data.get("created", ""))
		test_hour = extrac_hour(test_data.get("created", ""))
		data_frame['hour'] = hour
		test_data_frame['hour'] = test_hour

		# Extracting year, month and day from created
		data_frame["created"] = pd.to_datetime(data_frame["created"])
		data_frame["created_month"] = data_frame["created"].dt.month
		data_frame["created_day"] = data_frame["created"].dt.day
		
		test_data_frame["created"] = pd.to_datetime(test_data_frame["created"])
		test_data_frame["created_month"] = test_data_frame["created"].dt.month
		test_data_frame["created_day"] = test_data_frame["created"].dt.day
		
		# Append the features word counting as columns into data_frame
		data_frame['count_feature'] = count_feature(data_frame)
		test_data_frame['count_feature'] = count_feature(test_data_frame)

		# The idea of this part before line 229 is from Resource: https://www.kaggle.com/den3b81/improve-perfomances-using-manager-features
		# Based on the interest_level to get the manager skill
		temp = pd.concat([data_frame.manager_id,pd.get_dummies(data_frame['interest_level'])], axis = 1).groupby('manager_id').mean()
		temp.columns = ['high_frac','low_frac', 'medium_frac']
		temp['count'] = data_frame.groupby('manager_id').count().iloc[:,1]
		temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

		# Get ixes for unranked managers and ranked ones
		unranked_managers_ixes = temp['count']<20
		ranked_managers_ixes = ~unranked_managers_ixes

		# compute mean values from ranked managers and assign them to unranked ones
		mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
		temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
		data_frame = data_frame.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
		# Adding the manager skill into test data frame
		test_data_frame = test_data_frame.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
		new_manager_ixes = test_data_frame['high_frac'].isnull()
		test_data_frame.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values

		# Get the target and features
		target_values = data_frame['interest_level'].values
		features = ['address_category', 'new_price', 'hour', 'latitude', 'longitude', 'bathrooms', 'count_photos', 'count_feature',
					'created_month', 'created_day', 'num_description_words', 'bedrooms', 'manager_skill']

		# Set up the classifier
		clf = RandomForestClassifier(max_depth=52, min_samples_split=5, random_state=0)
		# Using cross_val_score to calculate the score of the training set
		score = cross_val_score(clf, data_frame[features], target_values, cv=10)
		print ("The score of the training set is: \n", score)
		print ("Mean of score: ", score.mean())
		print ("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
	
		# Train the model
		clf.fit(data_frame[features], target_values)

		# Get the predict probability of test set
		predict = clf.predict_proba(test_data_frame[features])

		# Calculate the features importance for each features
		feature_imp = pd.Series(clf.feature_importances_,index=features).sort_values(ascending=False)
		print ("\nFeature importance for each feature:\n" + str(feature_imp))

		# Put the prediction into data frame with listing id
		# Export it to csv file for submitting to Kaggle
		result = pd.DataFrame({"listing_id": test_data_frame["listing_id"].values, "high": predict[:, 0], "medium": predict[:, 2], "low": predict[:, 1]})
		result.to_csv('random_forest.csv', index=False)
		print ("\nFile: random_forest.csv generated.")
		print ("Random Forest finished.\n")

def n_neighbors():
	print ("Running n_neighbors...")
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

		data_frame['address_category'] = calculate_address_category(data, 'street_address', data_frame)
		test_data_frame['address_category'] = calculate_address_category(test_data, 'street_address', test_data_frame)

		data_frame['center_distance'] = distance(x_square,y_square)
		test_data_frame['center_distance'] = distance(x_square_test,y_square_test)

		data_frame['price_remove_outlier'] = deal_price(data_frame['price'])
		test_data_frame['price_remove_outlier'] = deal_price(test_data_frame['price'])

		data_frame['feature_number'] = count_features(data_frame['features'])
		test_data_frame['feature_number'] = count_features(test_data_frame['features'])

		data_frame['hour'] = extrac_hour(data.get("created", ""))
		test_data_frame['hour'] = extrac_hour(test_data.get("created", ""))

		target_values = data_frame["interest_level"].values.ravel()
		features = ['price_remove_outlier', 'address_category', 'hour', 'latitude', 'longitude','bathrooms','bedrooms','feature_number','center_distance']
		# kf = KFold(n_splits=10)

		neigh = KNeighborsClassifier(n_neighbors = 1000, weights = 'distance',p = 1)
		neigh.fit(data_frame[features], target_values)

		score = cross_val_score(neigh, data_frame[features], target_values, cv=10)
		
		print (score)
		print (score.mean())

		print("=================")

		accuracy = cross_val_score(neigh, data_frame[features], target_values, cv=10,scoring='balanced_accuracy')
		print(accuracy)
		print (accuracy.mean())

		predict = neigh.predict_proba(test_data_frame[features])

		result = pd.DataFrame({"listing_id": test_data_frame["listing_id"].values, "high": predict[:, 0], "medium": predict[:, 2], "low": predict[:, 1]})
		result.to_csv('n_neighbors.csv', index=False)
		print ("\nFile: n_neighbors.csv generated.")
		print ("Finished n_neighbors")

def Gradient_Boosting_Classifier():      
    print ('\nRunning xgbooster...\n')
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

        # encode string class values as integers
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(mytarget)
        label_encoded_y = label_encoder.transform(mytarget)
        mytarget = label_encoded_y        

	    #scaler = MinMaxScaler()
	    #myfeatures = scaler.fit_transform(myfeatures)
	    #mytarget = scaler.transform(mytarget)

        grid = {}
        myfeatures = np.array(myfeatures)
        xgb_clf = XGBClassifier(**grid)
	    #xgb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_features=1)          
        xgb_clf.fit(myfeatures, mytarget)

        y_pred = xgb_clf.predict(myfeatures)
        print(y_pred)

        print("Accuracy:",metrics.accuracy_score(mytarget, y_pred))

        #start cross validation by k-fold
        print("start cross validation by k-fold")

        X = np.array(result) # create an array
        y = np.array(mytarget) # Create another array
        kf = KFold(n_splits=5) # Define the split - into 2 folds 
        kf.get_n_splits(myfeatures) # returns the number of splitting iterations in the cross-validator

        xgb_clf2 = XGBClassifier(**grid)
	    #xgb_clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_features=1)          
	    #xgb_clf.fit(myfeatures, mytarget)

        scores = []
        proba = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #Create a svm Classifier

            #Train the model using the training sets
            xgb_clf2.fit(X_train, y_train)

            #Predict the response for test dataset
            y_pred = xgb_clf2.predict(X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

            y_prob = xgb_clf2.predict_proba(myfeatures)
            proba.append(y_prob)
        print("done")

    #test the model by test data
    print("start test the model by test data")
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
        myfeatures_test = np.array(myfeatures_test)
        y_pred_test = xgb_clf2.predict(myfeatures_test)

        y_prob_test = xgb_clf2.predict_proba(myfeatures_test)

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

        with open('XGBoost.csv', 'w', newline='') as mycsvfile:
            thedatawriter = csv.writer(mycsvfile, delimiter=',', lineterminator = '\n')
            thedatawriter.writerows(output_test)      
        with open('XGBoost.csv',newline='') as f:
            r = csv.reader(f)
            data = [line for line in r]
        with open('XGBoost.csv','w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['listing_id', 'high', 'medium', 'low'])
            w.writerows(data)
    print("done")
        
if __name__ == "__main__":
	random_forest()
	n_neighbors()
	Gradient_Boosting_Classifier()