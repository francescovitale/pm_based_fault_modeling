:: %1 tp_percentage, %2 noise threshold

set pd_variant=inductive
set pd_variant_type=imf
set tp_percentage=%1
set noise_threshold=%2
set ab_cc_variant_type=dijkstra_no_heuristics
set n_sensor_types=1
set sensor_types=J2

for /D %%p IN ("Output\AD\Inference\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Output\AD\Training\Model\*
del /F /Q Output\ELE\Training\EventLogs\*
del /F /Q Output\ELE\Inference\EventLogs\*
del /F /Q Output\E\Metrics\AlignmentBased\*
del /F /Q Output\E\Metrics\FootprintBased\*
del /F /Q Output\E\Metrics\TokenBased\*
del /F /Q Output\BC\Inference\PetriNets\*
del /F /Q Output\BC\Training\PetriNets\*

del /F /Q Input\AD\Inference\Model\*
del /F /Q Input\BC\Inference\Data\*
del /F /Q Input\BC\Training\Data\*
del /F /Q Input\E\EventLogs\*
del /F /Q Input\E\PetriNets\*
for /D %%p IN ("Input\AD\Inference\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Input\AD\Training\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /D %%p IN ("Input\ELE\Training\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Input\ELE\Inference\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

xcopy Output\WE\Training\Data Input\ELE\Training\Data /E
xcopy Output\WE\Inference\Data Input\ELE\Inference\Data /E
xcopy Output\WE\Training\Data Input\AD\Training\Data /E
xcopy Output\WE\Inference\Data Input\AD\Inference\Data /E

python event_logs_extraction.py %n_sensor_types% %sensor_types%

copy Output\ELE\Training\EventLogs\* Input\BC\Training\Data
copy Output\ELE\Inference\EventLogs\V.xes Input\E\EventLogs
copy Output\ELE\Inference\EventLogs\W.xes Input\E\EventLogs

python anomaly_detection.py %tp_percentage%

for /D %%p IN ("Input\ELE\Inference\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Output\ELE\Inference\EventLogs\*
xcopy Output\AD\Inference\Data Input\ELE\Inference\Data /E

python event_logs_extraction.py %n_sensor_types% %sensor_types%

copy Output\ELE\Inference\EventLogs\* Input\BC\Inference\Data 

python behavior_characterization.py %pd_variant% %pd_variant_type% %noise_threshold%

copy Output\BC\Inference\PetriNets\V.pnml Input\E\PetriNets
copy Output\BC\Inference\PetriNets\W.pnml Input\E\PetriNets
::copy Output\BC\Training\PetriNets\T.pnml Input\E\PetriNets

python evaluation.py %ab_cc_variant_type%
