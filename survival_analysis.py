import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter

if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/survival_dataframe.csv')
    retention_df['present_next_year'] = 1 - retention_df['present_next_year']
    retention_df['end_year'] = retention_df['end_year'] + 1

    # retention_df = retention_df.dropna(subset=['school_district_gifted_ratio'])
    print(retention_df.isna().sum())

    # Instantiate the Cox Proportional Hazards model
    ctv = CoxTimeVaryingFitter()

    # Fit the model
    ctv.fit(retention_df, id_col='teacher_id', event_col='present_next_year', start_col='start_year', stop_col='end_year',
            formula="assignments_count")

    # Print the summary
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    print(ctv.summary)

