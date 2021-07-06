'''
Import all the necessary libraries
'''
import os
from os import path
import logging
import datetime
from datetime import datetime
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data('data/bank_data.csv')
		logging.info('{time_frame} | {message}'.format
					 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
					  message = "Testing import_data: SUCCESS"))
	except FileNotFoundError:
		logging.error("Testing import_eda: The file wasn't found")
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")

def perform_eda(df):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		cls.perform_eda(df)
		# logging.info("Testing import_data: SUCCESS")
		logging.info('{time_frame} | {message}'.format
					 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
					  message = "Testing test_eda: SUCCESS"))
	except FileNotFoundError:
		logging.error("Testing test_eda: The file wasn't found")
		# raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError:
		logging.error("Testing test_eda: The file doesn't appear to have rows and columns")

def test_encoder_helper(df,category_lst):
	'''
	 This is the test_encoder_helper function where I am converting
	 the categorical data into a neumeric data
     And finaly return the X value.
	'''
	try:
		cls.encoder_helper(df,category_lst)
		logging.info('{time_frame} | {message}'.format
					 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
					  message = "Testing encoder_helper: SUCCESS"))
	except FileNotFoundError:
		logging.error("Testing encoder_helper: The file wasn't found")
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError:
		logging.error("Testing encoder_helper: The file doesn't appear to have rows and columns")
		# raise err

	try:
		assert len(category_lst) > 0
		logging.info('{time_frame} | {message}'.format
					 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
					  message = "Testing encoder_helper: SUCCESSFUL"))
	except AssertionError:
		logging.error("Testing encoder_helper: ERROR")
		# raise err

def test_perform_feature_engineering(df, response):
	'''
	working with the perform_feature_engineering from churn_library
	& try to handle the exception The file
	does or doesn't appear to have rows and columns.
	'''
	try:
		cls.perform_feature_engineering(df, response)
		logging.info('{time_frame} | {message}'.format
					 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
					  message = "Testing perform_feature_engineering: SUCCESS"))
	except FileNotFoundError:
		logging.error("Testing perform_feature_engineering: The file wasn't found")
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError:
		logging.error("Testing perform_feature_engineering: The file"
					  " doesn't appear to have rows and columns")

def test_train_models(X_train,Y_train,X_test,Y_test):

	'''
        Split the df & return X_train_data, X_test_data, y_train_data, y_test_data
    '''
	try:
		cls.train_models(X_train,Y_train,X_test,Y_test)
		logging.info('{time_frame} | {message}'.format
					 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
					  message = "Testing train_models: SUCCESS"))
	except FileNotFoundError:
			logging.error("Testing train_models: The file wasn't found")
	try:
		exists = os.path.isfile('models/rfc_model.pkl')
		if exists:
			logging.info('{time_frame} | {message}'.format
						 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
						  message = "Path found"))
		else:
			logging.info('{time_frame} | {message}'.format
						 (time_frame=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
						  message = "Path not found"))
	except AssertionError:
		logging.error("Testing train_models: The file doesn't appear to have rows and columns")

if __name__ == "__main__":
	test_import('data/bank_data.csv')
	perform_eda(df)
	test_encoder_helper(cls.data, cls.cat_columns)
	test_perform_feature_engineering(cls.X_train,cls.response)
	test_train_models(cls.X_train, cls.X_test , cls.y_train, cls.y_test)
