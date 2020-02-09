import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image, ImageStat
import requests
from io import BytesIO
import math

# count the number of values occurrence in dictionary
# input type: dic. Example: price_data.values()
# output type: dic
def count_values(value):
	result = {}
	for i in value:
		result[i] = result.get(i, 0) + 1
	return result

# draw the histogram
# input type: value -> list. Example: price_data.values()
def draw_histogram(data, bins=10, title='title', xlabel='x', ylabel='y'):
	n_bins = bins
	plt.hist(data.values(), n_bins, color='#449F44')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

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

#return a function with one argument with lower_bound and upper_bounds given
def filter_outlier(lower_bound,upper_bound):
	def f2(num):
		if ((num >= lower_bound) and (num <= upper_bound)):
			return True
		else:
			return False
	result = f2
	return result

#find number of outliers for a variable	
def count_outlier(variable):
	lower_bound, upper_bound = get_bounds(variable)
	values = list(variable.values())
	count = 0
	for i in range(0, len(values)):
		if ((values[i] < lower_bound) or (values[i] > upper_bound)):
			count += 1
		else:
			continue	
	return count

#draw the histogram of the price
def clean_outlier(variable, title='title', xlabel='x', ylabel='y'):
	
	values = list(variable.values())

	values.sort()

	lower_bound, upper_bound = get_bounds(variable)

	variable_filter_outlier = filter_outlier(lower_bound, upper_bound)

	filtered_variable = list(filter(variable_filter_outlier, values))

	plt.hist(filtered_variable, bins = 50, color = '#449F44')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

def extrac_hour(data):
	time = []
	hour = []
	values = data.values()
	for i in values:
		time.append(i.split(' ')[1])
	for i in time:
		hour.append(i.split(':')[0])
	return hour

def find_proportion_target_variables(data):
	created = data['created']
	times = list(created.values())
	def hour_filter(time):
		return time[11] + time[12]
	result = list(map(hour_filter, times))
	list_hour = []
	most_common_5 = dict(Counter(result).most_common(5))
	# print(most_common_5)
	plt.pie(most_common_5.values(), labels=most_common_5, autopct='%.1f%%')
	plt.title("Proportion of Hours")
	plt.show()

def missing_value(data_frame, field):
	count = 0
	interest_level = ['medium', 'high', 'low']

	if field == 'latitude' or field == 'longitude':
		for i in data_frame[field]:
			if i == 0:
				count += 1
	elif field == 'building_id':
		for i in data_frame[field]:
			if i == '0':
				count += 1
	elif field == 'description' or field == 'display_address' or field == 'street_address':
		for i in data_frame[field]:
			if i == '':
				count += 1
	elif field == 'features' or field == 'photos':
		for i in data_frame[field]:
			if i == []:
				count += 1
	elif field == 'bedrooms' or field == 'bathrooms':
		for i in data_frame[field]:
			if i == 0:
				count += 1
	elif field == 'interest_level':
		for i in data_frame[field]:
			if i not in interest_level:
				count += 1
	else:
		None

	print ("The missing values in " + field + " are: " + str(count))
	
	return count

def plot_outliers(data):
	mydata1 = {}
	mydata2 = {}
	mydata3 = {}		
	price = data['price']
	latitude = data['latitude']
	longitude = data['longitude']
	mydata1['price'] = list(price.values())

	labels, data = [*zip(*mydata1.items())]  # 'transpose' items to parallel key, value lists

	# or backwards compatable    
	labels, data = mydata1.keys(), mydata1.values()

	plt.boxplot(data)
	plt.xticks(range(1, len(labels) + 1), labels)
	plt.title("Price Outliers")
	plt.show()

	mydata2['latitude'] = list(latitude.values())
	labels, data = [*zip(*mydata2.items())]  # 'transpose' items to parallel key, value lists

	# or backwards compatable    
	labels, data = mydata2.keys(), mydata2.values()

	plt.boxplot(data)
	plt.xticks(range(1, len(labels) + 1), labels)
	plt.title("Latitude Outliers")
	plt.show()


	mydata3['longitude'] = list(longitude.values())
	labels, data = [*zip(*mydata3.items())]  # 'transpose' items to parallel key, value lists

	# or backwards compatable    
	labels, data = mydata3.keys(), mydata3.values()

	plt.boxplot(data)
	plt.xticks(range(1, len(labels) + 1), labels)
	plt.title("Longitude Outliers")
	plt.show()	

def number_of_outliers(data):
	#dic to store variables as key and their outliers as values
	my_data = {}

	#find outliers num of price
	price = data['price']
	num_price_outliers = count_outlier(price)
	my_data['price'] = num_price_outliers

	#find outliers num of latitude
	latitude = data['latitude']
	num_latitude_outliers = 0
	for l in latitude:
		if (latitude[l] < -90) or (latitude[l]  > 90) or (latitude[l]  == 0):
			num_latitude_outliers += 1
	my_data['latitude'] = num_latitude_outliers

	#find longitude outlier nums
	longitude = data['longitude']
	num_longitude_outliers = 0
	for i in longitude:
		if (longitude[i] < -180) or (longitude[i] > 80):
			num_longitude_outliers += 1
	my_data['longitude'] = num_longitude_outliers
	for v in my_data:
		print("For " + str(v),", number of outliers is: " + str(my_data[v]) + "\n")

def brightness( im_file ):
	im = im_file
	stat = ImageStat.Stat(im)
	return stat.rms[0]

def main():
	with open('train.json') as json_file:
		data = json.load(json_file)

		price_data = data.get("price", "")
		latitude_data = data.get("latitude", "")
		longitude_data = data.get("longitude", "")
		created_data = data.get("created", "")

		# =============================== 1.1 =============================
		# Plot the histogram of price, latitude and longitude
		print ("============================== Part 1.1 ===================================\n")
		print ("Drawing histogram with outliers. Please close window to continue.")

		draw_histogram(price_data, 10, "Price (with outliers)", "Price ($)", "Frequecny")
		draw_histogram(latitude_data, 10, "Latitude (with outliers)", "Latitude", "Frequecny")
		draw_histogram(longitude_data, 10, "Latitude (with outliers)", "longitude", "Frequecny")

		print ("Now drawing values without the outliers.")

		clean_outlier(price_data, "Price (without outliers)", "Price ($)", "Frequecny")
		clean_outlier(latitude_data, "Latitude (without outliers)", "Latitude", "Frequecny")
		clean_outlier(longitude_data, "Longitude (without outliers)", "Longitude", "Frequecny")

		print ("\n============================== Part 1.1 End ===================================\n")
		# =============================== 1.2 =============================
		# Plot the hour graph and give the top 5 busiest hours
		print ("============================== Part 1.2 ===================================\n")
		hour_freq = {}
		hour = extrac_hour(created_data)
		hour_counting = count_values(hour)

		for i in sorted(hour_counting):
			hour_freq[i] = hour_counting[i]

		print ("Now Drawing the hour-wise listing trend graph. Close the window to continue.\n")
		plt.bar(hour_freq.keys(), hour_freq.values(), color='g')
		plt.title('Hour-wise Listing Trend')
		plt.xlabel('Hours')
		plt.ylabel('Values')
		plt.show()

		top_5 = sorted(hour_counting, key=hour_counting.get, reverse=True)[:5]
		print ("The top 5 busiest hours of posting is: ", top_5)
		print ("\n============================== Part 1.2 End ===================================\n")
		# =============================== 1.3 =============================
		# Plot the proportion of target values
		print ("============================== Part 1.3 ===================================\n")

		print ("Drawing the proportion of target values.")
		find_proportion_target_variables(data)

		print ("============================== Part 1.3 End ===================================\n")
		# =============================== 2.1 =============================
		# Find the number of missing values
		
		print ("============================== Part 2.1 ===================================\n")
		missing_values = 0

		data_frame = pd.DataFrame(data)

		print ("\nFinding null values in all fields... Now showing the result...\n")
		print (data_frame.isnull().max())
		print ("\n")

		for i in data_frame.columns:
			missing_values += missing_value(data_frame, i)

		print ("\nThe total number of missing values are: " + str(missing_values))

		print ("============================== Part 2.1 End ===================================\n")
	
		print ("============================== Part 2.2 ===================================\n")
		print ("Found the outliers: ")
		
		number_of_outliers(data)

		plot_outliers(data)

		print ("============================== Part 2.2 End ===================================\n")

		# =============================== 3.1 =============================
		#Convert image to greyscale, return RMS pixel brightness.
		print ("============================== Part 3.1 ===================================\n")
		print ("Extracting brightness from image, Please wait...")
		
		brightness_list = []
		photos = data['photos']
		for i in photos:	
			b = 0
			if(len(photos[i])>0):
				img = requests.get(photos[i][0])
				img = Image.open(BytesIO(img.content))
				print (brightness(img))
				brightness_list.append(brightness(img))
		print ("Data finished processing.")
		
		print ("============================== Part 3.1 End ===================================\n")

		# =============================== 3.2 =============================

		print ("============================== Part 3.2 ===================================\n")
		print ("Extracting features from text...\n")
		f = open("stopwords.txt", "r")
		content = f.read()
		stopwords = frozenset(content.split("\n"))

		feature_content = []
		for i in data_frame['features']:
			feature_content += i

		count_vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word')
		word_counts = count_vectorizer.fit_transform(feature_content)

		tfidf_transformer = TfidfTransformer()
		tfidf_results = tfidf_transformer.fit_transform(word_counts)
		print ("Data stored in the tfidf_results, and ready for further use in model training.\n" + 
				"See comment then uncomment it to print details of matrix.")
		print ("============================== Part 3.2 End ===================================\n")
		# The way to print the tfidf matrix
		# df = pd.DataFrame(tfidf_results.todense(), columns=count_vectorizer.get_feature_names())
		# for i in df.columns:
		# 	print ("words in features (columns): "  + str(i))
		# 	row_index = 0
		# 	for j in df[i]:
		# 		row_index += 1
		# 		if j > 0:
		# 			print ("row " + str(row_index) + ": " + str(j))

if __name__ == "__main__":
	main()