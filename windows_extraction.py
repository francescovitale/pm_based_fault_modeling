import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import pandas as pd
import numpy as np
import math
import os
from RoADDataset import Dataset

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

input_dir = "Input/WE/"

output_dir = "Output/WE/"
output_training_dir = output_dir + "Training/"
output_training_data_dir = output_training_dir + "Data/"

output_inference_dir = output_dir + "Inference/"
output_inference_data_dir = output_inference_dir + "Data/"


n_clusters = -1
n_components = -1
clustering_technique = "kmeans"

def per_joint_state_extraction(training_observation, collision_observation, control_observation, weight_observation, velocity_observation, n_samples_per_window, n_clusters, sensor_types):

	training_datasets_columns = ["ID", "AP", "C", "F", "PA", "P", "PF", "RP", "V", "J1XA", "J1YA", "J1ZA", "J1XAV", "J1YAV", "J1ZAV", "J1QOC1", "J1QOC2", "J1QOC3", "J1QOC4", "J1T", "J2XA", "J2YA", "J2ZA", "J2XAV", "J2YAV", "J2ZAV", "J2QOC1", "J2QOC2", "J2QOC3", "J2QOC4", "J2T", "J3XA", "J3YA", "J3ZA", "J3XAV", "J3YAV", "J3ZAV", "J3QOC1", "J3QOC2", "J3QOC3", "J3QOC4", "J3T", "J4XA", "J4YA", "J4ZA", "J4XAV", "J4YAV", "J4ZAV", "J4QOC1", "J4QOC2", "J4QOC3", "J4QOC4", "J4T", "J5XA", "J5YA", "J5ZA", "J5XAV", "J5YAV", "J5ZAV", "J5QOC1", "J5QOC2", "J5QOC3", "J5QOC4", "J5T", "J6XA", "J6YA", "J6ZA", "J6XAV", "J6YAV", "J6ZAV", "J6QOC1", "J6QOC2", "J6QOC3", "J6QOC4", "J6T", "J7XA", "J7YA", "J7ZA", "J7XAV", "J7YAV", "J7ZAV", "J7QOC1", "J7QOC2", "J7QOC3", "J7QOC4", "J7T"]

	other_datasets_columns = ["ID", "AP", "C", "F", "PA", "P", "PF", "RP", "V", "J1XA", "J1YA", "J1ZA", "J1XAV", "J1YAV", "J1ZAV", "J1QOC1", "J1QOC2", "J1QOC3", "J1QOC4", "J1T", "J2XA", "J2YA", "J2ZA", "J2XAV", "J2YAV", "J2ZAV", "J2QOC1", "J2QOC2", "J2QOC3", "J2QOC4", "J2T", "J3XA", "J3YA", "J3ZA", "J3XAV", "J3YAV", "J3ZAV", "J3QOC1", "J3QOC2", "J3QOC3", "J3QOC4", "J3T", "J4XA", "J4YA", "J4ZA", "J4XAV", "J4YAV", "J4ZAV", "J4QOC1", "J4QOC2", "J4QOC3", "J4QOC4", "J4T", "J5XA", "J5YA", "J5ZA", "J5XAV", "J5YAV", "J5ZAV", "J5QOC1", "J5QOC2", "J5QOC3", "J5QOC4", "J5T", "J6XA", "J6YA", "J6ZA", "J6XAV", "J6YAV", "J6ZAV", "J6QOC1", "J6QOC2", "J6QOC3", "J6QOC4", "J6T", "J7XA", "J7YA", "J7ZA", "J7XAV", "J7YAV", "J7ZAV", "J7QOC1", "J7QOC2", "J7QOC3", "J7QOC4", "J7T", "L"]

	supply_columns = ["AP", "C", "F", "PA", "P", "PF", "RP", "V"]

	
	joint_columns=[["J1XA", "J1YA", "J1ZA", "J1XAV", "J1YAV", "J1ZAV", "J1QOC1", "J1QOC2", "J1QOC3", "J1QOC4", "J1T"], 
	["J2XA", "J2YA", "J2ZA", "J2XAV", "J2YAV", "J2ZAV", "J2QOC1", "J2QOC2", "J2QOC3", "J2QOC4", "J2T"],
	["J3XA", "J3YA", "J3ZA", "J3XAV", "J3YAV", "J3ZAV", "J3QOC1", "J3QOC2", "J3QOC3", "J3QOC4", "J3T"],
	["J4XA", "J4YA", "J4ZA", "J4XAV", "J4YAV", "J4ZAV", "J4QOC1", "J4QOC2", "J4QOC3", "J4QOC4", "J4T"],
	["J5XA", "J5YA", "J5ZA", "J5XAV", "J5YAV", "J5ZAV", "J5QOC1", "J5QOC2", "J5QOC3", "J5QOC4", "J5T"],
	["J6XA", "J6YA", "J6ZA", "J6XAV", "J6YAV", "J6ZAV", "J6QOC1", "J6QOC2", "J6QOC3", "J6QOC4", "J6T"],
	["J7XA", "J7YA", "J7ZA", "J7XAV", "J7YAV", "J7ZAV", "J7QOC1", "J7QOC2", "J7QOC3", "J7QOC4", "J7T"]]
	
	'''
	joint_columns=[["J1XA", "J1YA", "J1ZA"], 
	["J2XA", "J2YA", "J2ZA"],
	["J3XA", "J3YA", "J3ZA"],
	["J4XA", "J4YA", "J4ZA"],
	["J5XA", "J5YA", "J5ZA"],
	["J6XA", "J6YA", "J6ZA"],
	["J7XA", "J7YA", "J7ZA"]]
	'''

	cluster_columns = ["CS", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

	dfs = {}
	dfs_columns = []
	if "S" in sensor_types:
		dfs_columns = dfs_columns + supply_columns
	for i in range(0,7):
		if "J"+str(i+1) in sensor_types:
			dfs_columns = dfs_columns + joint_columns[i]
	if "S" in sensor_types:
		dfs_columns = dfs_columns + [cluster_columns[0]]
	for i in range(0,7):
		if "J"+str(i+1) in sensor_types:
			dfs_columns = dfs_columns + [cluster_columns[i+1]]

	dfs["T"] = pd.DataFrame(columns = dfs_columns)
	#dfs["C"] = pd.DataFrame(columns = dfs_columns)
	#dfs["CTRL"] = pd.DataFrame(columns = dfs_columns)
	dfs["W"] = pd.DataFrame(columns = dfs_columns)
	dfs["V"] = pd.DataFrame(columns = dfs_columns)

	# Training observation
	temp = pd.DataFrame(columns = training_datasets_columns, data = training_observation)
	reuse_parameters = 0
	supply_df = temp[supply_columns]
	supply_df, supply_clustering_parameters = cluster_dataset(supply_df, reuse_parameters, None)
	supply_df = supply_df.rename(columns={"Cluster": "CS"})
	joint_dfs = []
	joint_clustering_parameters = []
	for idx,set in enumerate(joint_columns):
		joint_df = temp[set]
		joint_df, clustering_parameters = cluster_dataset(joint_df, reuse_parameters, None)
		joint_df = joint_df.rename(columns={"Cluster": "C" + str(idx+1)})
		joint_dfs.append(joint_df)
		joint_clustering_parameters.append(clustering_parameters)
	if "S" in sensor_types:
		for column in supply_df:
			dfs["T"][column] = supply_df[column]
	for idx,joint_df in enumerate(joint_dfs):
		if "J"+str(idx+1) in sensor_types:
			for column in joint_df:
				dfs["T"][column] = joint_df[column]
	dfs["T"] = split_data(dfs["T"], n_samples_per_window)

	'''
	# Collision observation
	temp = pd.DataFrame(columns = other_datasets_columns, data = collision_observation)
	reuse_parameters = 1
	supply_df = temp[supply_columns]
	supply_df, ignore = cluster_dataset(supply_df, reuse_parameters, supply_clustering_parameters)
	supply_df = supply_df.rename(columns={"Cluster": "CS"})
	joint_dfs = []
	for idx,set in enumerate(joint_columns):
		joint_df = temp[set]
		joint_df, ignore = cluster_dataset(joint_df, reuse_parameters, joint_clustering_parameters[idx])
		joint_df = joint_df.rename(columns={"Cluster": "C" + str(idx+1)})
		joint_dfs.append(joint_df)
	if "S" in sensor_types:
		for column in supply_df:
			dfs["C"][column] = supply_df[column]
	for idx,joint_df in enumerate(joint_dfs):
		if "J"+str(idx+1) in sensor_types:
			for column in joint_df:
				dfs["C"][column] = joint_df[column]
	dfs["C"] = linear_split_data(dfs["C"], n_samples_per_window)

	# Control observation
	temp = pd.DataFrame(columns = other_datasets_columns, data = control_observation)
	reuse_parameters = 1
	supply_df = temp[supply_columns]
	supply_df, ignore = cluster_dataset(supply_df, reuse_parameters, supply_clustering_parameters)
	supply_df = supply_df.rename(columns={"Cluster": "CS"})
	joint_dfs = []
	for idx,set in enumerate(joint_columns):
		joint_df = temp[set]
		joint_df, ignore = cluster_dataset(joint_df, reuse_parameters, joint_clustering_parameters[idx])
		joint_df = joint_df.rename(columns={"Cluster": "C" + str(idx+1)})
		joint_dfs.append(joint_df)
	if "S" in sensor_types:
		for column in supply_df:
			dfs["CTRL"][column] = supply_df[column]
	for idx,joint_df in enumerate(joint_dfs):
		if "J"+str(idx+1) in sensor_types:
			for column in joint_df:
				dfs["CTRL"][column] = joint_df[column]
	dfs["CTRL"] = linear_split_data(dfs["CTRL"], n_samples_per_window)
	'''
	# Weight observation
	temp = pd.DataFrame(columns = other_datasets_columns, data = weight_observation)
	reuse_parameters = 1
	supply_df = temp[supply_columns]
	supply_df, ignore = cluster_dataset(supply_df, reuse_parameters, supply_clustering_parameters)
	supply_df = supply_df.rename(columns={"Cluster": "CS"})
	joint_dfs = []
	for idx,set in enumerate(joint_columns):
		joint_df = temp[set]
		joint_df, ignore = cluster_dataset(joint_df, reuse_parameters, joint_clustering_parameters[idx])
		joint_df = joint_df.rename(columns={"Cluster": "C" + str(idx+1)})
		joint_dfs.append(joint_df)
	if "S" in sensor_types:
		for column in supply_df:
			dfs["W"][column] = supply_df[column]
	for idx,joint_df in enumerate(joint_dfs):
		if "J"+str(idx+1) in sensor_types:
			for column in joint_df:
				dfs["W"][column] = joint_df[column]
	dfs["W"] = split_data(dfs["W"], n_samples_per_window)

	# Velocity observation
	temp = pd.DataFrame(columns = other_datasets_columns, data = velocity_observation)
	reuse_parameters = 1
	supply_df = temp[supply_columns]
	supply_df, ignore = cluster_dataset(supply_df, reuse_parameters, supply_clustering_parameters)
	supply_df = supply_df.rename(columns={"Cluster": "CS"})
	joint_dfs = []
	for idx,set in enumerate(joint_columns):
		joint_df = temp[set]
		joint_df, ignore = cluster_dataset(joint_df, reuse_parameters, joint_clustering_parameters[idx])
		joint_df = joint_df.rename(columns={"Cluster": "C" + str(idx+1)})
		joint_dfs.append(joint_df)
	if "S" in sensor_types:
		for column in supply_df:
			dfs["V"][column] = supply_df[column]
	for idx,joint_df in enumerate(joint_dfs):
		if "J"+str(idx+1) in sensor_types:
			for column in joint_df:
				dfs["V"][column] = joint_df[column]
	dfs["V"] = split_data(dfs["V"], n_samples_per_window)

	return dfs

def cluster_dataset(dataset, reuse_parameters, clustering_parameters_in):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	
	if reuse_parameters == 0:
		if clustering_technique == "agglomerative":
			cluster_configuration = AgglomerativeClustering(n_clusters=n_clusters, affinity='cityblock', linkage='average')
			cluster_labels = cluster_configuration.fit_predict(clustered_dataset)
		elif clustering_technique == "kmeans":
			kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(clustered_dataset)
			cluster_labels = kmeans.labels_

		clustered_dataset["Cluster"] = cluster_labels
		cluster_labels = cluster_labels.tolist()
		used = set();
		clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

		instances_sets = {}
		centroids = {}
		
		for cluster in clusters:
			instances_sets[cluster] = []
			centroids[cluster] = []
			
		
		temp = clustered_dataset
		for index, row in temp.iterrows():
			instances_sets[int(row["Cluster"])].append(row.values.tolist())
		
		n_features_per_instance = len(instances_sets[0][0])-1
		
		for instances_set_label in instances_sets:
			instances = instances_sets[instances_set_label]
			for idx, instance in enumerate(instances):
				instances[idx] = instance[0:n_features_per_instance]
			for i in range(0,n_features_per_instance):
				values = []
				for instance in instances:
					values.append(instance[i])
				centroids[instances_set_label].append(np.mean(values))
				
		clustering_parameters = centroids
			
	else:
		clusters = []
		for index, instance in clustered_dataset.iterrows():
			min_value = float('inf')
			min_centroid = -1
			for centroid in clustering_parameters_in:
				centroid_coordinates = np.array([float(i) for i in clustering_parameters_in[centroid]])
				dist = np.linalg.norm(instance.values-centroid_coordinates)
				if dist<min_value:
					min_value = dist
					min_centroid = centroid
			clusters.append(min_centroid)
		
		clustered_dataset["Cluster"] = clusters
		

	return clustered_dataset, clustering_parameters

def split_data(timeseries, n_samples_per_window):

	windows = []
	temp_timeseries = timeseries.copy()

	# The following code could be used if a recurrent pattern of action IDs is recognizable from the system.
	'''
	idx_pairs = []
	start_idx = -1
	current_activity = -1

	for row_idx, row in temp_timeseries.iterrows():
		if row_idx == 0:
			prev_activity = row["ID"]
		else:
			prev_activity = current_activity
		current_activity = row["ID"]

		if current_activity == 0 and start_idx == -1:
			start_idx = row_idx

		if current_activity == 0 and start_idx != -1 and prev_activity != current_activity:
			end_idx = row_idx
			idx_pairs.append([start_idx, end_idx])
			start_idx = -1
			prev_activity = -1

	print(idx_pairs)
	'''

	'''
	PFP_ids = [1, 5, 9, 13, 15, 17, 19, 21, 23, 25, 27, 29]
	PTP_ids = [2, 6, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30]
	MOP_ids = [3, 4, 7, 8, 11, 12]

	idx_pairs = []
	start_idx = -1
	current_activity = -1
	for row_idx, row in temp_timeseries.iterrows():
		if row_idx == 0:
			prev_activity = row["ID"]
		else:
			prev_activity = current_activity
		current_activity = row["ID"]

		if start_idx == -1 and current_activity != 0:
			start_idx = row_idx

		elif start_idx != -1 and prev_activity != current_activity:
			end_idx = row_idx-1
			idx_pairs.append([start_idx, end_idx])
			start_idx = -1
			prev_activity = -1

	for elem in idx_pairs:
		windows.append(temp_timeseries.iloc[list(range(elem[0],elem[1]))].drop(axis=1, columns=["ID"]))

	'''

	n_windows = math.floor(len(timeseries)/n_samples_per_window)
	for i in range(0,n_windows):
		windows.append(timeseries.head(n_samples_per_window))
		timeseries = timeseries.iloc[n_samples_per_window:]

	return windows

def save_behavior(observation, folder_name, mode):

	if mode == "Training":
		dir = output_training_data_dir
	elif mode == "Inference":
		dir = output_inference_data_dir

	isExist = os.path.exists(dir + folder_name)
	if not isExist:
		os.makedirs(dir + folder_name)

	for idx,window in enumerate(observation):
		window.to_csv(dir + folder_name + "/" + folder_name + "_" + str(idx) + ".csv", index = False)



try:
	sensor_types = []
	n_samples_per_window = int(sys.argv[1])
	n_clusters = int(sys.argv[2])
	n_sensors = int(sys.argv[3])
	for i in range(1,n_sensors+1):
		sensor_types.append(sys.argv[3+i])
except:
	print("Enter the correct number of input arguments.")

dataset = Dataset(normalize=True)
training_observations = dataset.sets['training']
collision_observations = dataset.sets['collision']
control_observations = dataset.sets['control']
weight_observations = dataset.sets['weight']
velocity_observations = dataset.sets['velocity']

dfs = per_joint_state_extraction(training_observations[0], collision_observations[0], control_observations[0], weight_observations[0], velocity_observations[0], n_samples_per_window, n_clusters, sensor_types)

for observation in dfs:
	if observation == "T":
		save_behavior(dfs[observation], observation, "Training")
	else:
		save_behavior(dfs[observation], observation, "Inference")
