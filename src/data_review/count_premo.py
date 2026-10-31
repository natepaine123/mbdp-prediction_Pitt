import pandas as pd

# Make sure this path matches whatever you used to successfully load the discharge file!
file_path = 'data/raw/PreMo/redcap.Discharge & Outcomes data.parquet'
df = pd.read_parquet(file_path)

# These are the actual outcome columns identified from your terminal output
outcome_columns = [
    'died_yn',                  # Mortality
    'disch_dispo',              # Discharge disposition
    'ivh_grade',                # Intraventricular hemorrhage grade
    'bpd_36_weeks',             # Bronchopulmonary dysplasia
    'rop_laser_avastin',        # Retinopathy of prematurity treatment
    'bayley_yn'                 # Bayley developmental assessment completed
]

print("=== OUTCOME COUNTS ===")
for col in outcome_columns:
    if col in df.columns:
        print(f"\n--- {col} ---")
        # value_counts() counts the instances, dropna=False ensures we see missing data too
        print(df[col].value_counts(dropna=False))