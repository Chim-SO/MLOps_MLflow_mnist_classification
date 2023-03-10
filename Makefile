#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_NAME = ml_project_structure
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py -r data/raw/
	$(PYTHON_INTERPRETER) src/data/build_features.py -r data/raw/ -p data/processed/

