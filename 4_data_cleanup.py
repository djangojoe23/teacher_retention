import pandas as pd
import time
import numpy as np


def impute_nearest(group):
    # Apply forward fill to propagate last valid observation forward
    forward_filled = group['title_i'].ffill()
    # Apply backward fill to propagate next valid observation backward
    backward_filled = group['title_i'].bfill()
    # Use backward filled values only where forward filled did not provide a value
    imputed_data = forward_filled.combine_first(backward_filled)
    return imputed_data


if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/custom_retention_dataframe.csv')
    print(retention_df.shape)

    # ------------------------------
    # make sure all the columns are either int or float
    # ------------------------------
    retention_df['present_next_year'] = retention_df['present_next_year'].replace(['True'], 1)
    retention_df['present_next_year'] = retention_df['present_next_year'].replace(['False'], 0)

    retention_df['title_i'] = retention_df['title_i'].replace(['Yes'], 1)
    retention_df['title_i'] = retention_df['title_i'].replace(['No'], 0)

    for col in ['present_next_year', 'title_i', 'rigorous_courses_all', 'regular_attendance_all',
                'math_test_participation_all', 'math_proficient_advanced_all', 'math_growth_all',
                'english_test_participation_all', 'english_proficient_advanced_all', 'english_growth_all',
                'science_test_participation_all', 'science_proficient_advanced_all', 'science_growth_all',
                'district_charter_enrollment']:
        retention_df[col] = pd.to_numeric(retention_df[col], errors='coerce')
        retention_df[col] = retention_df[col].astype(float)

    # ------------------------------
    # impute certain columns
    # ------------------------------
    null_counts = retention_df.isnull().sum()
    print(null_counts.to_string())
    print()

    retention_df.sort_values(by=['school_id', 'year'], inplace=True)
    retention_df['title_i'] = retention_df.groupby('school_id').apply(lambda x: impute_nearest(x)).reset_index(level=0, drop=True)
    retention_df['title_i'] = retention_df['title_i'].replace([np.nan], 0)

    print('finished imputing missing title_i values...')

    impute_average = ['turnover_rate', 'rigorous_courses_all', 'regular_attendance_all', 'math_test_participation_all',
                      'math_proficient_advanced_all', 'math_growth_all', 'english_test_participation_all',
                      'english_proficient_advanced_all', 'english_growth_all', 'science_test_participation_all',
                      'science_proficient_advanced_all', 'science_growth_all', 'district_enrollment',
                      'district_charter_enrollment', 'school_district_male_ratio', 'school_district_female_ratio',
                      'school_district_white_ratio', 'school_district_black_ratio', 'school_district_asian_ratio',
                      'school_district_hispanic_ratio', 'school_district_multiracial_ratio',
                      'school_district_sped_ratio', 'school_district_gifted_ratio', 'school_district_homeless_ratio',
                      'school_district_foster_care_ratio', 'school_district_english_learner_ratio',
                      'school_district_poor_ratio', 'district_school_count', 'district_area']

    # for each school, calculate the mean of all the unique year, value pairs excluding nans
    unique_school_year_pairs = retention_df.drop_duplicates(subset=['school_id', 'year'])
    print(unique_school_year_pairs.shape)

    total_schools = len(unique_school_year_pairs['school_id'].unique())
    count = 0
    for school in unique_school_year_pairs['school_id'].unique():
        for column in impute_average:
            column_mean = unique_school_year_pairs.loc[unique_school_year_pairs['school_id'] == school, column].mean()
            # for each row, replace nan in the data column with the average for the school
            retention_df.loc[retention_df['school_id'] == school, column] = retention_df.loc[retention_df['school_id'] == school, column].fillna(column_mean)
        count += 1
        if count % 100 == 0:
            print(f'{count} / {total_schools} = {count/total_schools}')


    retention_df.sort_values(by=['teacher_id', 'year'], inplace=True)
    # print(retention_df[['teacher_id', 'year', 'school_id', 'rigorous_courses_all', 'imputed_rigorous_courses_all']].head(100).to_string())
    null_counts = retention_df.isnull().sum()
    print()
    print(null_counts.to_string())

    print(retention_df.shape)
    retention_df.to_csv('dataframes/imputed_custom_retention_dataframe.csv', index=False)


