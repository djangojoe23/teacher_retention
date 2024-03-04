import pandas as pd
import time
import numpy as np


if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/retention_dataframe.csv')
    print(retention_df.shape)
    print(retention_df.head(100).to_string())
    quit()

    category_lists = {'assignments': [], 'district_types': [], 'degrees': []}
    for c in retention_df.columns:
        if c.startswith('assignments'):
            category_lists['assignments'].append(c)
        elif c.startswith('district_type'):
            category_lists['district_types'].append(c)
        elif 'degree' in c:
            category_lists['degrees'].append(c)

    total_time = time.time()

    # ------------------------------
    # school_count, district_count
    # ------------------------------
    # Sort the DataFrame for consistency
    retention_df.sort_values(by=['teacher_id', 'year'], inplace=True)
    # Add a new column for the count of different schools up to that year
    for col in ['school', 'district']:
        # Dropping duplicates for unique teacher-school combinations across all years
        unique_combos = retention_df.drop_duplicates(subset=['teacher_id', f'{col}_id']).copy()

        # Grouping by teacher_id and calculating the cumulative number of unique schools or districts up to each year
        unique_combos[f'{col}_count'] = unique_combos.groupby('teacher_id').cumcount() + 1

        # Merging this cumulative count back into the original retention_df on teacher_id and school_id or district_id
        # This step requires adjusting to ensure we merge based on the maximum count for each year
        # For that, we'll group by teacher_id and year to get the max cumulative value for each year
        max_cumulative_per_year = unique_combos.groupby(['teacher_id', 'year'])[f'{col}_count'].max().reset_index()

        retention_df = pd.merge(retention_df, max_cumulative_per_year, on=['teacher_id', 'year'], how='left')

        # Fill NaN values in district_count or school_count with the last valid value within each teacher's group
        retention_df[f'{col}_count'] = retention_df.groupby('teacher_id')[f'{col}_count'].ffill()

    print(f'school_count and district_count completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # years_at_school
    # ------------------------------
    # Correcting the calculation for years_at_current_school, specifically for T3 in 2021 at school 103
    retention_df['years_at_school'] = retention_df.groupby(['teacher_id', 'school_id'])['year'].rank(method='dense').astype(int)
    print(f'years_at_school completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # assignment_count, district_type_count, degree_count
    # ------------------------------
    for col_list in category_lists:
        # Step 1: Convert columns to boolean indicating whether the subject was taught
        col_bool = retention_df[category_lists[col_list]].gt(0)
        # Step 2: Group by teacher_id and calculate the cumulative sum of subjects taught over the years
        cumulative_count = col_bool.groupby(retention_df['teacher_id']).cumsum()
        # Step 3: For each row, calculate the cumulative count of unique subjects taught by checking for >0
        # (indicating the subject was taught at least once up to that year)
        retention_df[f'{col_list}_count'] = cumulative_count.gt(0).sum(axis=1)
    print(f'assignments_count, district_types_count, and degrees_count completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # raise_per_school_years
    # ------------------------------
    # Calculate the first year's salary for each teacher at each school as a variable
    first_year_salary = retention_df.groupby(['teacher_id', 'school_id'])['annual_salary'].transform('first')

    # Calculate the salary difference as a variable
    salary_difference = retention_df['annual_salary'] - first_year_salary

    # Now, calculate the average annual salary change using these variables
    retention_df['raise_per_school_years'] = salary_difference / retention_df['years_at_school']
    print(f'raise_per_school_years completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # raise_per_district_years, raise_per_education_years
    # ------------------------------
    for groupby_columns in [['teacher_id'], ['teacher_id', 'district_id']]:
        # 1. Identify the first year for each teacher and the max salary in that year
        first_year = retention_df.groupby(groupby_columns)['year'].transform('min')
        retention_df['first_year_max_salary'] = retention_df.loc[retention_df['year'] == first_year].groupby(groupby_columns)['annual_salary'].transform('max')
        # Fill NaN values that appear because the transform('max') was applied only to the first year rows
        retention_df['first_year_max_salary'] = retention_df.groupby(groupby_columns)['first_year_max_salary'].ffill()

        # 2. Calculate the max salary for each current year for each teacher
        groupby_columns.append('year')
        retention_df['current_year_max_salary'] = retention_df.groupby(groupby_columns)['annual_salary'].transform('max')

        # 3. Compute the salary difference from the first year's max to the current year's max salary
        retention_df['salary_diff_from_first_year'] = retention_df['current_year_max_salary'] - retention_df['first_year_max_salary']

        e = 'education'
        if 'district_id' in groupby_columns:
            e = 'district'
        # 4. Calculate the average annual salary change from the max salary in the first year
        retention_df[f'raise_per_{e}_years'] = retention_df.apply(
            lambda x: x['salary_diff_from_first_year'] / x[f'years_in_{e}'] if x[f'years_in_{e}'] > 1 else 0, axis=1
        )
    retention_df.drop(['first_year_max_salary', 'current_year_max_salary', 'salary_diff_from_first_year'], axis=1, inplace=True)

    print(f'raise_per_education_years and raise_per_district_years completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # turnover_rate
    # ------------------------------
    # Step 1: Count Teachers per School per Year
    teacher_counts = retention_df.groupby(['school_id', 'year'])['teacher_id'].nunique().reset_index(name='teacher_count')
    teacher_counts.sort_values(by=['school_id', 'year'], inplace=True)

    # Step 2: Calculate Year-Over-Year Difference in Teacher Counts for Each School
    teacher_counts['teacher_count_diff'] = teacher_counts.groupby('school_id')['teacher_count'].diff()

    # Step 3: Calculate Previous Year Teacher Count
    teacher_counts['previous_year_teacher_count'] = teacher_counts.groupby('school_id')['teacher_count'].shift(1)

    # Step 4: Compute Turnover Rate
    teacher_counts['turnover_rate'] = teacher_counts['teacher_count_diff'] / teacher_counts['previous_year_teacher_count']

    # Merge the Turnover Rate back to the original DataFrame
    retention_df = pd.merge(retention_df, teacher_counts[['school_id', 'year', 'turnover_rate']], on=['school_id', 'year'], how='left')
    print(f'staff_turnover_rate completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # school_info
    # ------------------------------
    all_school_data_df = pd.read_csv('dataframes/Datafile_df.csv', index_col=0)
    school_columns = ['school_id', 'year', 'rigorous_courses_all', 'title_i', 'regular_attendance_all',
                      'math_test_participation_all', 'math_proficient_advanced_all', 'math_growth_all',
                      'english_test_participation_all', 'english_proficient_advanced_all', 'english_growth_all',
                      'science_test_participation_all', 'science_proficient_advanced_all', 'science_growth_all']
    retention_df = pd.merge(retention_df, all_school_data_df[school_columns], on=['school_id', 'year'], how='left')
    print(f'school_info completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # district_info
    # ------------------------------
    all_district_data_df = pd.read_csv('dataframes/DistrictFastFacts_df.csv', index_col=0)
    district_columns = ['district_id', 'year', 'district_enrollment', 'district_charter_enrollment', 'district_school_count',
                        'district_area']
    retention_df = pd.merge(retention_df, all_district_data_df[district_columns], on=['district_id', 'year'], how='left')
    print(f'district_info completed in {time.time() - total_time:.2f}')

    # ------------------------------
    # school_district_ratios
    # ------------------------------
    all_district_data_df = pd.read_csv('dataframes/DistrictFastFacts_df.csv', index_col=0)
    all_school_data_df = pd.read_csv('dataframes/Datafile_df.csv', index_col=0)
    ratio_columns = ['id', 'male', 'female', 'white', 'black', 'asian', 'hispanic', 'multiracial', 'sped',
                     'gifted', 'homeless', 'foster_care', 'english_learner', 'poor']
    # Step 1: Merge teachers_df with all_school_data_df and all_district_data_df
    for p in ['school', 'district']:
        subset_columns = [f'{p}_' + s for s in ratio_columns]
        subset_columns.append('year')
        if p == 'school':
            subset = all_school_data_df[subset_columns]
        else:
            subset = all_district_data_df[subset_columns]
        retention_df = pd.merge(retention_df, subset, on=[f'{p}_id', 'year'], how='left')
        retention_df[subset_columns] = retention_df[subset_columns].replace(['none', '  -    -'], np.nan)
        subset_dict = {}
        for c in subset_columns:
            if c not in ['id', 'year']:
                subset_dict[c] = float
        retention_df = retention_df.astype(subset_dict)

    del all_school_data_df
    del all_district_data_df
    
    # Step 3: Calculate the ratios for each statistic
    ratio_columns.remove('id')
    # Pre-calculate all ratios in a dictionary
    ratios = {}
    for s in ratio_columns:
        ratio_name = f'school_district_{s}_ratio'  # Naming the new ratio columns
        ratios[ratio_name] = retention_df[f'school_{s}'] / retention_df[f'district_{s}']
    # Avoid division by zero - replace inf with np.nan
    ratios = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in ratios.items()}
    # Add the pre-calculated ratios to the DataFrame
    retention_df = retention_df.assign(**ratios)
    print(f'school_district_ratios completed in {time.time() - total_time:.2f}')

    unnecessary_columns = []
    for c in ratio_columns:
        unnecessary_columns.append(f'school_{c}')
        unnecessary_columns.append(f'district_{c}')
    retention_df.drop(unnecessary_columns, axis=1, inplace=True)

    print(f'Total time took {time.time() - total_time:.2f}')
    print("Writing dataframe to csv now...")
    retention_df.to_csv('dataframes/custom_retention_dataframe.csv', index=False)
    print("I'm done")