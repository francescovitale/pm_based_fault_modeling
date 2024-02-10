import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

input_dir = "Input/"
output_dir = "Output/"

def read_data():

	data = {}

	for filename in os.listdir(input_dir):
		data[filename.split("_")[0]] = pd.read_csv(input_dir + filename)

	return data

def split_data(data):

	non_pareto_optimal_data = {}
	pareto_optimal_data = {}

	for observation in data:
		non_pareto_optimal_data[observation] = data[observation].copy()
		pareto_optimal_data[observation] = data[observation].copy()
	
		cluster_number_values = list(set(list(data[observation]["ClusterNumber"].values)))
		
		pareto_optimal_indices = []
		non_pareto_optimal_indices = []

		for cluster_number in cluster_number_values:
			df = data[observation].copy()
			df = df[df["ClusterNumber"] == cluster_number].drop(columns=["ClusterNumber", "NoiseThreshold"], axis=0)
			df_sorted = df.sort_values(by=['Precision', 'Fitness'], ascending=False)
			is_pareto_optimal = np.ones(df_sorted.shape[0], dtype=bool)
			current_max_1 = float('-inf')
			for i in range(df_sorted.shape[0]):
				if (df_sorted.iloc[i]['Fitness'] <= current_max_1):
					is_pareto_optimal[i] = False
				else:
					current_max_1 = df_sorted.iloc[i]['Fitness']

			pareto_optimal_indices = pareto_optimal_indices + list(df_sorted[is_pareto_optimal].index)
			non_pareto_optimal_indices = non_pareto_optimal_indices + list(df_sorted[~is_pareto_optimal].index)
		
		non_pareto_optimal_data[observation] = non_pareto_optimal_data[observation].loc[non_pareto_optimal_indices].reset_index(drop=True)
		pareto_optimal_data[observation] = pareto_optimal_data[observation].loc[pareto_optimal_indices].reset_index(drop=True)

	return non_pareto_optimal_data, pareto_optimal_data

def scatter_data(non_pareto_optimal_data, pareto_optimal_data):
	fig = plt.figure(figsize=(10,3.5))
	for idx_obs, observation in enumerate(non_pareto_optimal_data):
		
		cluster_number_values = list(set(list(non_pareto_optimal_data[observation]["ClusterNumber"])))
		for idx,cluster_number in enumerate(cluster_number_values):
			ax = fig.add_subplot(2, len(cluster_number_values), idx+1+(idx_obs*3))
			non_pareto_optimal_points = non_pareto_optimal_data[observation].loc[non_pareto_optimal_data[observation]['ClusterNumber'] == cluster_number]
			pareto_optimal_points = pareto_optimal_data[observation].loc[pareto_optimal_data[observation]['ClusterNumber'] == cluster_number]
			non_pareto_optimal_fitness_values = list(non_pareto_optimal_points["Fitness"])
			non_pareto_optimal_precision_values = list(non_pareto_optimal_points["Precision"])
			pareto_optimal_fitness_values = list(pareto_optimal_points["Fitness"])
			pareto_optimal_precision_values = list(pareto_optimal_points["Precision"])
			pareto_optimal_noise_threshold_values = list(pareto_optimal_points["NoiseThreshold"])
			ax.set_ylim(0.65, 1.05)
			ax.set_xlim(0, 0.65)
			if idx == 0:
				ax.set_ylabel("Fitness")
			if idx_obs > 0:
				ax.set_xlabel("Precision")

			ax.scatter(non_pareto_optimal_precision_values, non_pareto_optimal_fitness_values, marker = "o", label = "Non-optimal")
			ax.scatter(pareto_optimal_precision_values, pareto_optimal_fitness_values, marker = "x", label = "Optimal")
			if observation == "V":
				ax.set_title("V " + str(cluster_number) + " clusters", size = 8, fontweight = "bold")
			else:
				ax.set_title("W " + str(cluster_number) + " clusters", size = 8, fontweight = "bold")
			if idx +(idx_obs*3) == len(cluster_number_values) + 3 - 1:
				ax.legend(loc="upper right", borderpad=0.2, prop={'size': 7})
			
			for i, txt in enumerate(pareto_optimal_noise_threshold_values):
				if i == 0:
					ax.annotate(txt, (pareto_optimal_precision_values[i], pareto_optimal_fitness_values[i]), fontsize=7, weight='bold')
				elif prev_precision != pareto_optimal_precision_values[i] and prev_fitness != pareto_optimal_fitness_values[i]:
					ax.annotate(txt, (pareto_optimal_precision_values[i], pareto_optimal_fitness_values[i]), fontsize=7, weight='bold')
				prev_precision = pareto_optimal_precision_values[i]
				prev_fitness = pareto_optimal_fitness_values[i]
			ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
			ax.set_yticks([0.7, 0.8, 0.9, 1.0])

	fig.tight_layout()
	plt.savefig(output_dir + "FITNESS_PRECISION.png", bbox_inches='tight')
	plt.close(fig)
	

	return None

data = read_data()
non_pareto_optimal_data, pareto_optimal_data = split_data(data)
scatter_data(non_pareto_optimal_data, pareto_optimal_data)


