#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_NAME = MLOps_MLflow_mnist_classification
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py -r data/raw/
	$(PYTHON_INTERPRETER) src/data/build_features.py -r data/raw/ -p data/processed/

