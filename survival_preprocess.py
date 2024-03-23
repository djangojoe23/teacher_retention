
import pandas as pd

if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/imputed_custom_retention_dataframe.csv')

    # Only look at certain teachers
    staff_filter = (retention_df['category_teacher'] == 1) & \
                   (retention_df['status_active'] == 1) & \
                   (retention_df['position_secondary'] == 1)
    retention_df = retention_df[staff_filter]

    # Due to filter above, drop these unnecessary columns
    retention_df = retention_df.drop(columns=[col for col in retention_df.columns if col.startswith('category')])
    retention_df = retention_df.drop(columns=[col for col in retention_df.columns if col.startswith('status')])
    retention_df = retention_df.drop(columns=[col for col in retention_df.columns if col.startswith('position')])

    # if i can't tell if the teacher left the school or not, then we'll say they didn't leave
    retention_df['present_next_year'].fillna(1, inplace=True)

    # Assimilate categorical variables that were one-hot encoded back to a single variable with a lot of categories
    # 74 teachers have a year/school where that have more than 1 assignment, we're choosing the first assignment with the max value of all assignments
    assignment_columns = [col for col in retention_df.columns if col.startswith('assignments')]
    assignment_columns.remove('assignments_count')
    retention_df['assignment'] = None

    district_type_columns = [col for col in retention_df.columns if col.startswith('district_type')]
    district_type_columns.remove('district_types_count')
    retention_df['district_type'] = None

    gender_columns = [col for col in retention_df.columns if col.startswith('gender')]
    retention_df['gender'] = None

    degree_columns = [col for col in retention_df.columns if col.startswith('highest_degree')]
    retention_df['highest_degree'] = None

    unique_teachers = retention_df['teacher_id'].unique()
    counter = len(unique_teachers)
    for t_id in unique_teachers:
        counter -= 1
        if counter % 1000 == 0:
            print(counter)

        t_df = retention_df.loc[retention_df['teacher_id'] == t_id]
        unique_years = t_df['year'].unique()
        for y in unique_years:
            y_df = t_df.loc[t_df['year'] == y]
            unique_schools = y_df['school_id'].unique()
            for s in unique_schools:
                s_df = y_df.loc[t_df['school_id'] == s]
                row_filter = (retention_df['teacher_id'] == t_id) & \
                             (retention_df['year'] == y) & \
                             (retention_df['school_id'] == s)

                # Assignments
                assignment_column = s_df[assignment_columns].idxmax(axis=1)
                assignment = '_'.join(assignment_column.values[0].split('_')[1:])
                retention_df.loc[row_filter, 'assignment'] = assignment

                # District Type
                district_type_column = s_df[district_type_columns].idxmax(axis=1)
                district_type = '_'.join(district_type_column.values[0].split('_')[2:])
                retention_df.loc[row_filter, 'district_type'] = district_type

                # Gender
                gender_column = s_df[gender_columns].idxmax(axis=1)
                gender = '_'.join(gender_column.values[0].split('_')[1:])
                retention_df.loc[row_filter, 'gender'] = gender

                # Highest Degree
                degree_column = s_df[degree_columns].idxmax(axis=1)
                degree = '_'.join(degree_column.values[0].split('_')[2:])
                retention_df.loc[row_filter, 'highest_degree'] = degree

    # Drop unnecessary columns due to the assimilation of these categorical variables
    retention_df = retention_df.drop(columns=assignment_columns)
    retention_df = retention_df.drop(columns=district_type_columns)
    retention_df = retention_df.drop(columns=gender_columns)
    retention_df = retention_df.drop(columns=degree_columns)

    # add columns for start and stop time for each teacher at each school
    # Calculate the earliest and latest year for each teacher at each school
    start_year = retention_df.groupby(['teacher_id', 'school_id'])['year'].min().reset_index(name='start_year')
    end_year = retention_df.groupby(['teacher_id', 'school_id'])['year'].max().reset_index(name='end_year')

    # Merge the earliest and latest year back into the original dataframe
    df_merged = retention_df.merge(start_year, on=['teacher_id', 'school_id'], how='left')
    df_merged = df_merged.merge(end_year, on=['teacher_id', 'school_id'], how='left')

    # Output the dataframe to a csv for use in survival analysis.py
    df_merged.to_csv('teacher_retention/dataframes/survival_dataframe.csv', index=False)
