set n_reps=10
set tp_percentage=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

del /F /Q Results\*

call windows_extraction 5

for %%y in (%tp_percentage%) do (
	for /l %%x in (1, 1, %n_reps%) do (
		call fault_modeling %%y 0.2
		copy Output\E\Metrics\AlignmentBased\Fitness_V.txt Results
		copy Output\E\Metrics\AlignmentBased\Fitness_W.txt Results
		copy Output\E\Metrics\AlignmentBased\Precision_V.txt Results
		copy Output\E\Metrics\AlignmentBased\Precision_W.txt Results
		ren Results\Fitness_V.txt Fitness_V_%%y_%%x.txt
		ren Results\Fitness_W.txt Fitness_W_%%y_%%x.txt
		ren Results\Precision_V.txt Precision_V_%%y_%%x.txt
		ren Results\Precision_W.txt Precision_W_%%y_%%x.txt
	)
)
