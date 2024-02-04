from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.conversion.process_tree import converter as pt_converter

import pm4py.algo.discovery as pd
import sys
import os

input_dir = "Input/BC/"
input_training_dir = input_dir + "Training/"
input_training_data_dir = input_training_dir + "Data/"
input_inference_dir = input_dir + "Inference/"
input_inference_data_dir = input_inference_dir + "Data/"

output_dir = "Output/BC/"
output_training_dir = output_dir + "Training/"
output_training_petrinets_dir = output_training_dir + "PetriNets/"
output_inference_dir = output_dir + "Inference/"
output_inference_petrinets_dir = output_inference_dir + "PetriNets/"

variant = ""
variant_type = ""
parameters = {}

def read_event_logs():
	event_logs = {}

	event_logs["Training"] = {}
	for observation in os.listdir(input_training_data_dir):
		event_logs["Training"][observation.split(".xes")[0]] = xes_importer.apply(input_training_data_dir + observation)

	event_logs["Inference"] = {}
	for observation in os.listdir(input_inference_data_dir):
		event_logs["Inference"][observation.split(".xes")[0]] = xes_importer.apply(input_inference_data_dir + observation)

	return event_logs

def process_discovery(event_log, variant, variant_type, parameters):

	petri_net = {}

	if variant == "inductive":
		if variant_type == "im":
			petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pd.inductive.algorithm.apply(event_log, variant=pd.inductive.algorithm.Variants.IM, parameters=parameters)
		
		elif variant_type == "imd":
			petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pd.inductive.algorithm.apply(event_log, variant=pd.inductive.algorithm.Variants.IMd, parameters=parameters)
		
		elif variant_type == "imf":
			petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pd.inductive.algorithm.apply(event_log, variant=pd.inductive.algorithm.Variants.IMf, parameters=parameters)

	return petri_net
	
def export_petri_net(petri_net, el_name, mode):
	if mode == "Training":
		output_petrinets_dir = output_training_petrinets_dir
	elif mode == "Inference":
		output_petrinets_dir = output_inference_petrinets_dir

	pnml_exporter.apply(petri_net["network"], petri_net["initial_marking"], output_petrinets_dir + el_name + ".pnml", final_marking = petri_net["final_marking"])

try:
	variant = sys.argv[1]
	variant_type = sys.argv[2]
	if variant == "inductive":
		if variant_type == "imf":
			parameters[pd.inductive.algorithm.Variants.IMf.value.Parameters.NOISE_THRESHOLD] = float(sys.argv[3])
except:
	print("Enter the right number of input arguments.")
	sys.exit()

event_logs = read_event_logs()

for event_log in event_logs["Training"]:
	try:
		petri_net = process_discovery(event_logs["Training"][event_log], variant, variant_type, parameters)
		export_petri_net(petri_net, event_log, "Training")
	except:
		print("Could not mine the training Petri net")
		pass
for event_log in event_logs["Inference"]:
	try:
		petri_net = process_discovery(event_logs["Inference"][event_log], variant, variant_type, parameters)
		export_petri_net(petri_net, event_log, "Inference")
	except:
		print("Could not mine the inference Petri nets")
		pass



