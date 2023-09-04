# dataiku
Interview task for Dataiku

Navigating the repository:
- data:
  - `col_names.csv` is needed to read in data files correctly
  - pipeline presumes that learn/test data also in this folder - user should update with their path
- notebooks:
  - `eda.ipynb` - used for initial exploration of data and includes important visualisations for features vs target
  - `results.ipynb` - used for analysing optimal model
- src:
  - contains core code for fitting model and producing prections
  - `runner.py` - used to trigger `pipeline.py` with user's input
  - `pipeline.py` - logic for end-to-end running of model
  - `model_trainer.py` - class for training and evaluating models
- support:
  - `requirements.txt` - contains packages used in project, please install using `pip install -r requirements.txt`
 
