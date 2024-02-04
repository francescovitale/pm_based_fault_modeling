import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import pandas as pd
import os

from random import seed
from random import random
from random import randint

import math

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

input_dir = "Input/ELE/"
input_training_dir = input_dir + "Training/"
input_training_data_dir = input_training_dir + "Data/"
input_inference_dir = input_dir + "Inference/"
input_inference_data_dir = input_inference_dir + "Data/"

output_dir = "Output/ELE/"
output_training_dir = output_dir + "Training/"
output_training_eventlogs_dir = output_training_dir + "EventLogs/"
output_inference_dir = output_dir + "Inference/"
output_inference_eventlogs_dir = output_inference_dir + "EventLogs/"

def read_data():
	observations = {}

	observations["Training"] = {}
	for observation in os.listdir(input_training_data_dir):
		observations["Training"][observation] = {}
		for window_file in os.listdir(input_training_data_dir + observation):
			observations["Training"][observation][window_file.split(".csv")[0]] = pd.read_csv(input_training_data_dir + observation + "/" + window_file)

	observations["Inference"] = {}
	for observation in os.listdir(input_inference_data_dir):
		observations["Inference"][observation] = {}
		for window_file in os.listdir(input_inference_data_dir + observation):
			observations["Inference"][observation][window_file.split(".csv")[0]] = pd.read_csv(input_inference_data_dir + observation + "/" + window_file)

	return observations

def extract_state_transitions(data, sensor_types):
	state_transitions = []
	cluster_columns = []
	if "S" in sensor_types:
		cluster_columns.append("CS")
	for i in range(1,7+1):
		if "J"+str(i) in sensor_types:
			cluster_columns.append("C"+str(i))

	current_state = {}
	for idx,sample in data.iterrows():
		if len(current_state) == 0:
			for cluster in cluster_columns:
				current_state[cluster] = sample[cluster]
		else:
			next_state = {}
			for cluster in cluster_columns:
				next_state[cluster] = sample[cluster]
			for cluster in next_state:
				if next_state[cluster] != current_state[cluster]:
					if cluster == "CS":
						new_state_transition = "S-" + str(int(current_state[cluster])) + "_" + str(int(next_state[cluster]))
					else:
						new_state_transition = "J" + str(cluster[1]) + "-" + str(int(current_state[cluster])) + "_" + str(int(next_state[cluster]))
					state_transitions.append([new_state_transition,idx])
					current_state[cluster] = next_state[cluster]

	return state_transitions

def timestamp_builder(number):
	
	SSS = number
	ss = int(math.floor(SSS/1000))
	mm = int(math.floor(ss/60))
	hh = int(math.floor(mm/24))
	
	SSS = SSS % 1000
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)+"."+str(SSS)

def build_event_log(windows_state_transitions):
	
	event_log = []
	for idx,window_state_transitions in enumerate(windows_state_transitions):
		caseid = idx
		for state_transitions in window_state_transitions:
			event_timestamp = timestamp_builder(state_transitions[1])
			state_transition = state_transitions[0]
			event = [caseid, state_transition, event_timestamp]
			event_log.append(event)
	'''
	for state_transition in windows_state_transitions:
		caseid = 0
		event_timestamp = timestamp_builder(state_transition[1])
		state_transition = state_transition[0]
		event = [caseid, state_transition, event_timestamp]
		event_log.append(event)
	'''
	event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Timestamp'])
	event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
		
	return event_log	

def save_log(name, event_log, mode):

	if mode == "Training":
		eventlogs_dir = output_training_eventlogs_dir
	elif mode == "Inference":
		eventlogs_dir = output_inference_eventlogs_dir

	xes_exporter.apply(event_log, eventlogs_dir + name + ".xes")

	return None	

try:
	sensor_types = []
	n_sensor_types = int(sys.argv[1])
	for i in range(1, n_sensor_types+1):
		sensor_types.append(sys.argv[1+i])
except:
	print("Enter the right number of input arguments.")
	sys.exit()

observations = read_data()

for mode in observations:
	windows_state_transitions = {}
	event_logs = {}
	for observation in observations[mode]:
		windows_state_transitions[observation] = []
		for window in observations[mode][observation]:
			windows_state_transitions[observation].append(extract_state_transitions(observations[mode][observation][window], sensor_types))
		event_logs[observation] = build_event_log(windows_state_transitions[observation])
		save_log(observation, event_logs[observation], mode)
	
	'''
	if observation.split("_")[0] == "V" or observation.split("_")[0] == "W":
		n_windows_to_sample = int(len(observations[observation])*tp_percentage)
		keys = random.sample(observations[observation].keys(), n_windows_to_sample)
		sample_d = {k: d[k] for k in keys}
	'''










