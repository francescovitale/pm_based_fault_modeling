:: %1 n clusters
set n_clusters=%1
set window_length=1000
set n_sensor_types=1
set sensor_types=J2

for /D %%p IN ("Output\WE\Training\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Output\WE\Inference\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)


python windows_extraction.py %window_length% %n_clusters% %n_sensor_types% %sensor_types%

