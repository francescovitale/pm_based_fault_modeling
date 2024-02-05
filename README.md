# Requirements and instructions to run PM-based fault modeling against the RoAD dataset

Before delving into the details of the project files, please consider that this project has been executed on a Windows 10 machine with Python 3.11.5. There are a few libraries that have been used within Python modules. Among these, there are:

- scikit-learn 1.3.0
- scipy 1.11.2
- pm4py 2.2.20.1
- pandas 1.4.1

Please note that the list above is not comprehensive and there could be other requirements for running the project. Furthermore, please note that we are using the RoAD dataset (instructions for its installation in python can be found at https://gitlab.com/AlessioMascolini/roaddataset/).

The execution of the method pipeline involves chaining the execution of two batch scripts, namely windows_extraction.bat and fault_modeling.bat. The former collects several observations from the RoAD dataset, such as the T, V, and W observations, and splits them into time series windows; the script requires setting the cluster number with which individual samples are labeled. fault_modeling.bat collects the time series windows and executes the rest of the method pipeline, which involves: anomaly detection, i.e., a given percentage of time series windows related to each (faulty) observation is held out from the whole set (1.0 tp_percentage means that all time series windows are collected); event log extraction, i.e., event logs are extracted from the time series windows; and process discovery, i.e., a process model is obtained for each faulty observation. Please note that for evaluation purposes, event_log_extraction is run twice; the first time all time series windows from a faulty observation are collected in the resulting event log, whereas the second time only the fixed percentage of time series windows is taken into account. This way, one could compare the quality of the resulting process model while tuning the number of time series windows being included when building the process model. Finally, the fault_modeling.bat requires setting the tp_percentage and the noise_threshold parameters. The former refers to the aforementioned percentage of anomalous time series windows collected from a faulty observation, whereas the latter is related to process discovery.

The project also includes two scripts for the automatic chaining of windows_extraction.bat and fault_modeling.bat, focusing on the experimentation process followed in the paper. experimentation_1.bat assumes perfect Recall, thus only the cluster number and the noise threshold parameters are varied to assess how much the Fitness and Precision metrics are impacted. experimentation_2.bat drops the assumption of perfect Recall and varies the tp percentage; in this case, the cluster number and the noise threshold parameters are fixed. Anyhow, the results of the experimentation are collected in the "Results" folder.