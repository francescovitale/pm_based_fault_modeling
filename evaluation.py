from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
import pm4py.algo.evaluation.replay_fitness as replay_fitness
import pm4py.algo.conformance.tokenreplay as tokenreplay
import pm4py.algo.conformance.alignments as alignments
from pm4py.algo.conformance.footprints import algorithm as fp_conformance
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.algo.conformance.footprints.util import evaluation

import pm4py
import sys
import os
import pandas as pd
import numpy as np

input_dir = "Input/E/"
input_eventlogs_dir = input_dir + "EventLogs/"
input_petrinets_dir = input_dir + "PetriNets/"

output_dir = "Output/E/"
output_metrics_dir = output_dir + "Metrics/"
output_tokenbased_metrics = output_metrics_dir + "TokenBased/"
output_alignmentbased_metrics = output_metrics_dir + "AlignmentBased/"
output_footprintbased_metrics = output_metrics_dir + "FootprintBased/"

parameters = {}

def read_event_logs():

	event_logs = {}

	for event_log in os.listdir(input_eventlogs_dir):
		event_logs[event_log.split(".xes")[0]] = xes_importer.apply(input_eventlogs_dir + event_log)

	return event_logs

def read_petri_nets():

	petri_nets = {}

	for petri_net in os.listdir(input_petrinets_dir):
		petri_nets[petri_net.split(".pnml")[0]] = {}
		petri_nets[petri_net.split(".pnml")[0]]["net"], petri_nets[petri_net.split(".pnml")[0]]["initial_marking"], petri_nets[petri_net.split(".pnml")[0]]["final_marking"] = pnml_importer.apply(input_petrinets_dir + petri_net)

	return petri_nets

def compute_fitness(event_log_1, event_log_2, petri_net, variant, variant_type, parameters):

	fitness = 0

	if variant == "token_based":
		if variant_type == "basic":
			replay_results = tokenreplay.algorithm.apply(log = event_log_1, net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.TOKEN_REPLAY)
		elif variant_type == "backwards":
			replay_results = tokenreplay.algorithm.apply(log = event_log_1, net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = tokenreplay.algorithm.Variants.BACKWARDS)
			
		fitness = replay_fitness.algorithm.evaluate(results = replay_results, variant = replay_fitness.algorithm.Variants.TOKEN_BASED)["log_fitness"]
	elif variant == "alignment_based":
		if variant_type == "dijkstra_less_memory":
			replay_results = alignments.petri_net.algorithm.apply_log(log = event_log_1, petri_net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_DIJKSTRA_LESS_MEMORY)
		elif variant_type == "dijkstra_no_heuristics":
			replay_results = alignments.petri_net.algorithm.apply_log(log = event_log_1, petri_net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_DIJKSTRA_NO_HEURISTICS)
		elif variant_type == "discounted_a_star":
			replay_results = alignments.petri_net.algorithm.apply_log(log = event_log_1, petri_net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_DISCOUNTED_A_STAR)
		elif variant_type == "state_equation_a_star":
			replay_results = alignments.petri_net.algorithm.apply_log(log = event_log_1, petri_net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_STATE_EQUATION_A_STAR)
		elif variant_type == "tweaked_state_equation_a_star":
			replay_results = alignments.petri_net.algorithm.apply_log(log = event_log_1, petri_net = petri_net["net"], initial_marking = petri_net["initial_marking"], final_marking = petri_net["final_marking"], parameters = parameters, variant = alignments.petri_net.algorithm.Variants.VERSION_TWEAKED_STATE_EQUATION_A_STAR)
		
		fitness = replay_fitness.algorithm.evaluate(results = replay_results, variant = replay_fitness.algorithm.Variants.ALIGNMENT_BASED)["log_fitness"]
	elif variant == "footprint_based":
		if variant_type == "entire_log":
			footprint_1 = footprints_discovery.apply(event_log_1, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
			footprint_2 = footprints_discovery.apply(event_log_2, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
			
		elif variant_type == "trace_by_trace":
			footprint_1 = footprints_discovery.apply(event_log_1, variant=footprints_discovery.Variants.TRACE_BY_TRACE)
			footprint_2 = footprints_discovery.apply(event_log_2, variant=footprints_discovery.Variants.TRACE_BY_TRACE)
			
		conf_result = fp_conformance.apply(footprint_1, footprint_2, variant=fp_conformance.Variants.LOG_EXTENSIVE)	
		fitness = evaluation.fp_fitness(footprint_1, footprint_2, conf_result)
	return fitness
	
def compute_precision(event_log, petri_net):

	precision = pm4py.precision_alignments(event_log, petri_net["net"], petri_net["initial_marking"], petri_net["final_marking"])

	return precision
	
def write_results(alignment_based_fitness, alignment_based_precision, event_logs, petri_nets):

	for event_log in alignment_based_fitness:
		temp = alignment_based_fitness[event_log].copy()
		file = open(output_alignmentbased_metrics + "Fitness_" + event_log + ".txt", "w")
		for idx,petri_net in enumerate(temp):
			if idx < len(petri_nets) - 1:
				file.write(petri_net + ":" + str(alignment_based_fitness[event_log][petri_net]) + "\n")
			else:
				file.write(petri_net + ":" + str(alignment_based_fitness[event_log][petri_net]))
		file.close()
	
	for event_log in alignment_based_precision:
		temp = alignment_based_precision[event_log].copy()
		file = open(output_alignmentbased_metrics + "Precision_" + event_log + ".txt", "w")
		for idx,petri_net in enumerate(temp):
			if idx < len(petri_nets) - 1:
				file.write(petri_net + ":" + str(alignment_based_precision[event_log][petri_net]) + "\n")
			else:
				file.write(petri_net + ":" + str(alignment_based_precision[event_log][petri_net]))
		file.close()
	return None

try:
	ab_cc_variant_type = sys.argv[1]
except:
	print("Enter the right number of input arguments.")
	sys.exit()


event_logs = read_event_logs()
petri_nets = read_petri_nets()

alignment_based_fitness = {}
alignment_based_precision = {}

		
for event_log in event_logs:
	alignment_based_fitness[event_log] = {}
	alignment_based_precision[event_log] = {}
	for petri_net in petri_nets:
		if petri_net == event_log or petri_net == "T":
		#	alignment_based_fitness[event_log][petri_net] = compute_fitness(event_logs[event_log], None, petri_nets[petri_net], "alignment_based", ab_cc_variant_type, parameters)			
			alignment_based_fitness[event_log][petri_net] = compute_fitness(event_logs[event_log], None, petri_nets[petri_net], "alignment_based", ab_cc_variant_type, parameters)
			alignment_based_precision[event_log][petri_net] = compute_precision(event_logs[event_log], petri_nets[petri_net])
	print(alignment_based_fitness)

write_results(alignment_based_fitness, alignment_based_precision, event_logs, petri_nets)



