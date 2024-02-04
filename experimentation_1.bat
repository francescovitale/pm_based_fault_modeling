set tp_percentage=1.0
set noise_thresholds=0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
set n_clusters=4 5 6

del /F /Q Results\*

for %%z in (%n_clusters%) do (
	call windows_extraction %%z

	for %%x in (%noise_thresholds%) do (
		call fault_modeling 1.0 %%x
		copy Output\E\Metrics\AlignmentBased\Fitness_V.txt Results
		copy Output\E\Metrics\AlignmentBased\Fitness_W.txt Results
		copy Output\E\Metrics\AlignmentBased\Precision_V.txt Results
		copy Output\E\Metrics\AlignmentBased\Precision_W.txt Results
		ren Results\Fitness_V.txt Fitness_V_IMf%%x_%%z.txt
		ren Results\Fitness_W.txt Fitness_W_IMf%%x_%%z.txt
		ren Results\Precision_V.txt Precision_V_IMf%%x_%%z.txt
		ren Results\Precision_W.txt Precision_W_IMf%%x_%%z.txt
	)
	
)