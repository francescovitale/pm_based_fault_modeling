import pandas as pd
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

input_dir = "Input/"
output_dir = "Output/"

def read_metrics():

	fitness = {}
	precision = {}

	fitness["V"] = {}
	precision["V"] = {}
	fitness["W"] = {}
	precision["W"] = {}
	for filename in os.listdir(input_dir):
		temp = pd.read_csv(input_dir + filename)
		for r_value in temp:
			if filename.split(".csv")[0] == "V_fitness":
				fitness["V"][r_value] = sum(temp[r_value])/len(temp[r_value])
			elif filename.split(".csv")[0] == "V_precision":
				precision["V"][r_value] = sum(temp[r_value])/len(temp[r_value])
			elif filename.split(".csv")[0] == "W_fitness":
				fitness["W"][r_value] = sum(temp[r_value])/len(temp[r_value])
			elif filename.split(".csv")[0] == "W_precision":
				precision["W"][r_value] = sum(temp[r_value])/len(temp[r_value])

	return fitness, precision

def plot(fitness, precision):

	R_values = list(fitness[list(fitness.keys())[0]].keys())
	fig = plt.figure(figsize=(16,3))

	for idx,observation in enumerate(fitness):
		ax = fig.add_subplot(1,2,idx+1)
		ax.set_ylim([0.25, 1])
		ax.set_xlim([0, 8])
		ax.set_title(observation, fontsize="medium", weight='bold')
		ax.set_xlabel('Recall', fontsize="medium")
		if idx == 0:
			ax.set_ylabel('Score', fontsize="medium")
		#ax.set_xticks(R_values)
		fitness_values = list(fitness[observation].values())
		precision_values = list(precision[observation].values())

		
		ax.plot(R_values, fitness_values, marker='|', mfc="k", mec = "k", linewidth=4.0, markersize=10, label = "F")
		ax.plot(R_values, precision_values, marker='x', mfc="k", mec = "k", linewidth=4.0, markersize=10, label = "P")

		if idx == 1:
			ax.legend(loc="center right")
			
		
	fig.tight_layout()
	plt.savefig(output_dir + "F_P_COMPARISON.png", bbox_inches='tight')

	return None

def bar_plot(metrics):

	# plotting the values obtained for 6 clusters

	V_keys = list(metrics["V"]["6"].keys())
	V_keys = [float(x) for x in V_keys]
	V_values = list(metrics["V"]["6"].values())
	W_keys = list(metrics["W"]["6"].keys())
	W_keys = [float(x) for x in W_keys]
	W_values = list(metrics["W"]["6"].values())

	fig = plt.figure(figsize=(8,5))
	ax = fig.add_subplot(1,1,1)
	ax.set_ylim([0, 1])
	ax.set_title('Fitness (6 clusters)', fontsize="medium", weight='bold')
	ax.set_xlabel('True positive percentage', fontsize="medium")
	ax.set_ylabel('Score', fontsize="medium", labelpad=-20.0)
	ax.set_xticks(list(range(0,len(V_keys))))
	positions = list(range(0,len(V_keys)))
	for idx,position in enumerate(positions):
		positions[idx] = position-0.15
	ax.bar(positions, V_values, width=0.2, color = "seagreen", label="V")
	positions = list(range(0,len(V_keys)))
	for idx,position in enumerate(positions):
		positions[idx] = position+0.15
	ax.bar(positions, W_values, width=0.2, color = "mediumpurple", label="W")
	ax.set_xticklabels(V_keys)
	ax.legend(loc = "upper left")
	fig.tight_layout()
	plt.savefig(output_dir + "6_cluster_bar_plot.png", bbox_inches='tight')

	# plotting the values obtained for 7 clusters

	V_keys = list(metrics["V"]["7"].keys())
	V_keys = [float(x) for x in V_keys]
	V_values = list(metrics["V"]["7"].values())
	W_keys = list(metrics["W"]["7"].keys())
	W_keys = [float(x) for x in W_keys]
	W_values = list(metrics["W"]["7"].values())

	fig = plt.figure(figsize=(8,5))
	ax = fig.add_subplot(1,1,1)
	ax.set_ylim([0, 1])
	ax.set_title('Fitness (7 clusters)', fontsize="medium", weight='bold')
	ax.set_xlabel('True positive percentage', fontsize="medium")
	ax.set_ylabel('Score', fontsize="medium", labelpad=-20.0)
	ax.set_xticks(list(range(0,len(V_keys))))
	positions = list(range(0,len(V_keys)))
	for idx,position in enumerate(positions):
		positions[idx] = position-0.15
	ax.bar(positions, V_values, width=0.2, color = "seagreen", label="V")
	positions = list(range(0,len(V_keys)))
	for idx,position in enumerate(positions):
		positions[idx] = position+0.15
	ax.bar(positions, W_values, width=0.2, color = "mediumpurple", label="W")
	ax.set_xticklabels(V_keys)
	ax.legend(loc = "upper left")
	fig.tight_layout()
	plt.savefig(output_dir + "7_cluster_bar_plot.png", bbox_inches='tight')

fitness, precision = read_metrics()
plot(fitness, precision)

#metrics = read_metrics()