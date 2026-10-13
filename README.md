# MBDP Logistic Regression Model

This project trains a logistic regression model to predict the risk of metabolic bone disease of prematurity (MBDP) within a fixed horizon (e.g., 14 days). It is designed to run on Ubuntu and prioritizes high sensitivity (catching cases) over false positives.

## Setup on Ubuntu

1. Install Python 3 and pip:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv


## Commands for running locally
1. Go to folder
   cd ~/.../BioInf2070/repo/mbdp_model

2. Activate venv
   source venv/bin/activate

3. Install requirements (only need to do once)
   pip install -r requirements.txt

4. Run the pipeline
   python -m src.data_generator
   python -m src.train
   python -m src.predict --input data/train.csv --output data/predictions.csv

5. Exit venv
   deactivate