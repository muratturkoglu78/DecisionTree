import math
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

class Node(object):
	def __init__(self, attribute, value_):
		self.attr = attribute
		self.value_ = value_
		self.left = None
		self.right = None
		self.leaf = False
		self.infogain= None

# The value chosen splits the test data such that information gain is maximized
def select_value(df, attribute, predict_attr):
	# Convert dataframe column to a list and round each value
	values = df[attribute].tolist()
	values = [ float(x) for x in values]
	# Remove duplicate values by converting the list to a set, then sort the set
	values = set(values)
	values = list(values)
	values.sort()
	max_ig = float("-inf")
	tres_val = 0
	# try all values that are half-way between successive values in this sorted list
	for i in range(0, len(values) - 1):
		tres = (values[i] + values[i+1])/2
		ig = info_gain(df, attribute, predict_attr, tres)
		if ig > max_ig:
			max_ig = ig
			tres_val = tres
	# Return the threshold value that maximizes information gained
	return tres_val

def info_entropy(df, predict_attr):
	# Dataframe and number of positive/negatives examples in the data
	p_df = df[df[predict_attr] == 1]
	n_df = df[df[predict_attr] == 2]
	p = float(p_df.shape[0])
	n = float(n_df.shape[0])
	# Calculate entropy
	if p  == 0 or n == 0:
		I = 0
	else:
		I = ((-1*p)/(p + n))*math.log(p/(p+n), 2) + ((-1*n)/(p + n))*math.log(n/(p+n), 2)
	return I

# the weighted average of the entropy after an attribute test
def remainder(df, df_subsets, predict_attr):
	# number of test data
	num_data = df.shape[0]
	remainder = float(0)
	for df_sub in df_subsets:
		if df_sub.shape[0] > 1:
			remainder += float(df_sub.shape[0]/num_data)*info_entropy(df_sub, predict_attr)
	return remainder

# the information gain from the attribute test based on a given threshold
def info_gain(df, attribute, predict_attr, threshold):
	sub_1 = df[df[attribute] < threshold]
	sub_2 = df[df[attribute] > threshold]
	# Determine information content, and subract remainder of attributes from it
	ig = info_entropy(df, predict_attr) - remainder(df, [sub_1, sub_2], predict_attr)
	return ig

# Returns the number of positive and negative data
def num_class(df, predict_attr):
	p_df = df[df[predict_attr] == 1]
	n_df = df[df[predict_attr] == 2]
	return p_df.shape[0], n_df.shape[0]

# Chooses the attribute and its threshold with the highest info gain
# from the set of attributes
def choose_attr(df, attributes, predict_attr):
	max_info_gain = float("-inf")
	best_attr = None
	threshold = 0
	# Test each attribute (note attributes maybe be chosen more than once)
	for attr in attributes:
		thres = select_value(df, attr, predict_attr)
		ig = info_gain(df, attr, predict_attr, thres)
		if ig > max_info_gain:
			max_info_gain = ig
			best_attr = attr
			threshold = thres
	return best_attr, threshold, ig

# Builds the Decision Tree based on training data, attributes to train on,
# and a prediction attribute
def build_tree(df, cols, predict_attr, level):
	# Get the number of positive and negative examples in the training data
	p, n = num_class(df, predict_attr)
	# If train data has all positive or all negative values
	# then we have reached the end of our tree
	if p == 0 or n == 0 or level == 4:
		# Create a leaf node indicating it's prediction
		leaf = Node(None,None)
		leaf.leaf = True
		if p > n:
			leaf.predict = 1
		else:
			leaf.predict = 2
		return leaf
	else:
		# Determine attribute and its threshold value with the highest
		# information gain
		best_attr, threshold, ifg = choose_attr(df, cols, predict_attr)
		# Create internal tree node based on attribute and it's threshold
		tree = Node(best_attr, threshold)
		tree.infogain = ifg
		tree.value_ = threshold
		sub_1 = df[df[best_attr] < threshold]
		sub_2 = df[df[best_attr] > threshold]
		# Recursively build left and right subtree
		tree.left = build_tree(sub_1, cols, predict_attr, level + 1)
		tree.right = build_tree(sub_2, cols, predict_attr, level + 1)
		return tree

def csvToDf(csv_file_name):
	df = pd.read_csv(csv_file_name, header=None)
	df.columns = ['X', 'Y', 'Type']
	df.drop(['Type'], axis=1 )
	cols = df.columns
	df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
	return df

def print_tree(root, level, direction):
	if root.leaf:
		if (root.predict == 1):
			arrPrint.append('level = ' + str(level) + ' direction = ' + direction + ' Predict : butterfly')
		else:
			arrPrint.append('level = ' + str(level) + ' direction = ' + direction + ' Predict : bird')
	else:
		arrPrint.append('level = ' + str(level) + ' direction = ' + direction + ' ' + root.attr + '=' + str(root.value_) + ' infogain = ' + str(root.infogain))
	if root.left:
		print_tree(root.left, level + 1, direction + '-' + 'left')
	if root.right:
		print_tree(root.right, level + 1, direction + '-' + 'right')

def main():
	# 'build_tree'
	df_train = csvToDf('Data.txt')
	attributes =  ['X', 'Y']
	root = build_tree(df_train, attributes, 'Type', 1)
	print_tree(root, 1, 'Root')
	arrPrint.sort()
	for a in arrPrint:
		print (a)
	pass

arrPrint = []
main()

