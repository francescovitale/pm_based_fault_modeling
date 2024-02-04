del /F /Q Output\AD\Training\Model\*
del /F /Q Output\ELE\Training\EventLogs\*
del /F /Q Output\ELE\Inference\EventLogs\*
del /F /Q Output\E\Metrics\AlignmentBased\*
del /F /Q Output\E\Metrics\FootprintBased\*
del /F /Q Output\E\Metrics\TokenBased\*
del /F /Q Output\BC\Inference\PetriNets\*
del /F /Q Output\BC\Training\PetriNets\*
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
for /D %%p IN ("Output\AD\Inference\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Output\AD\Training\Data\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)



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
