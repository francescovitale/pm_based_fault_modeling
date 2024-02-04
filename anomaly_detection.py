import sys
import os
import pandas as pd
import random

input_dir = "Input/AD/"
input_training_dir = input_dir + "Training/"
input_training_data_dir = input_training_dir + "Data/"

input_inference_dir = input_dir + "Inference/"
input_inference_data_dir = input_inference_dir + "Data/"
input_inference_model_dir = input_inference_dir + "Model/"

output_dir = "Output/AD/"
output_training_dir = output_dir + "Training/"
output_training_model_dir = output_training_dir + "Model/"

output_inference_dir = output_dir + "Inference/"
output_inference_data_dir = output_inference_dir + "Data/"

def read_model():

	pass

def read_data():
	data = {}
	data["Training"] = {}
	for observation in os.listdir(input_training_data_dir):
		data["Training"][observation] = {}
		for window_file in os.listdir(input_training_data_dir + observation):
			data["Training"][observation][window_file.split(".csv")[0]] = pd.read_csv(input_training_data_dir + observation + "/" + window_file)

	data["Inference"] = {}
	for observation in os.listdir(input_inference_data_dir):
		data["Inference"][observation] = {}
		for window_file in os.listdir(input_inference_data_dir + observation):
			data["Inference"][observation][window_file.split(".csv")[0]] = pd.read_csv(input_inference_data_dir + observation + "/" + window_file)

	return data

def write_data(windows, observation):

	isExist = os.path.exists(output_inference_data_dir + observation)
	if not isExist:
		os.makedirs(output_inference_data_dir + observation)
	
	for window in windows:
		windows[window].to_csv(output_inference_data_dir + observation + "/" + window + ".csv", index = False)

	return None
	

try:
	tp_percentage = float(sys.argv[1])
except:
	print("Enter the right number of input arguments.")
	sys.exit()

data = read_data()

# Anomaly detection: to implement



for observation in data["Inference"]:
	n_windows_to_retain = int(len(data["Inference"][observation])*tp_percentage)
	window_names = list(data["Inference"][observation].keys())
	windows_to_retain = random.sample(window_names, n_windows_to_retain)
	temp = {}
	for window_name in windows_to_retain:
		temp[window_name] = data["Inference"][observation][window_name]
	

	write_data(temp, observation)








