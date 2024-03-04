"""
create single dataframe from datafile_df, districtfastfacts_df and individualstaffreport_df to be used for logistic regression

get a single row for each teacher in a single year from individualstaffreport
phase 1(complete) add columns based on individualstaffreport data including the predictor (are they still present next year?)
phase 2 add custom columns based on data from individualtaffreport
phase 3 focus in on specific types of teachers (ie high school math teachers...)
phase 4 add data from the school they are at to each row (need to deal with schools that are in the staff data frame but not in school dataframe...
phase 5 (incomplete because charter schools are not listed in districtfast facts) add data from the district they are in to each row
export to a single dataframe
"""
import csv
import pandas as pd


def print_dict_to_csv(my_dict, csv_file_name):
    with open(f'dictionaries/{csv_file_name}', 'w', newline='') as csv_file:
        print(f'Writing dict to {csv_file_name}...')
        writer = csv.DictWriter(csv_file, fieldnames=my_dict.keys())

        # Write header
        writer.writeheader()

        # Write data
        count = 0
        progress_printed = []
        for row in zip(*my_dict.values()):
            writer.writerow(dict(zip(my_dict.keys(), row)))
            count += 1
            progress = round(count / len(my_dict) * 100)
            if progress not in progress_printed:
                print(progress)
                progress_printed.append(progress)

    print(f"CSV file '{csv_file_name}' created.")


if __name__ == "__main__":
    print('Reading in IndividualStaffReport_df.csv...')
    staff_df = pd.read_csv('dataframes/IndividualStaffReport_df.csv', index_col=0)

    print('Cleaning up the data in the dataframe....')
    # eliminate rows with 'Salary Excluded for Fictitious/Contracted Staff' or 'Salary Excluded for Fictitious'
    staff_df = staff_df.drop(staff_df[
                                 (staff_df['annual_salary'] == 'Salary Excluded for Fictitious/Contracted Staff') | (
                                             staff_df['annual_salary'] == 'Salary Excluded for Fictitious')].index)

    staff_df.fillna('None Provided', inplace=True)

    # fix new lines in english second language assignments
    for v in staff_df['assignment_description'].unique():
        if '\n' in v:
            staff_df['assignment_description'].replace(v, v.replace('\n', ''), inplace=True)

    # this file contains a list of keys/columns that will represent each row in the final dataframe
    # note that assignments will include a dictionary itself of fte_sum and position
    # in the final dataframe each categorical column will be encoded
    df_columns = pd.read_csv('retention_df_columns.csv')

    val_con_folder = 'values_consolidations'
    # excluding job_class because there are too many missing values for job_class in staff_df
    values_consolidations = {'assignments': pd.read_csv(f'{val_con_folder}/assignment_descriptions.tsv', sep='\t'),
                             'category': pd.read_csv(f'{val_con_folder}/category_descriptions.csv'),
                             'district_type': pd.read_csv(f'{val_con_folder}/district_types.csv'),
                             'gender': pd.read_csv(f'{val_con_folder}/genders.csv'),
                             'highest_degree': pd.read_csv(f'{val_con_folder}/highest_degrees.csv'),
                             # 'job_class': pd.read_csv(f'{val_con_folder}/job_classes.csv'),
                             'position': pd.read_csv(f'{val_con_folder}/position_descriptions.csv'),
                             'status': pd.read_csv(f'{val_con_folder}/statuses.csv'), }

    print('Converting IndividualStaffReport_df.csv into a dictionary...')
    count = 0
    educator_count = len(staff_df['public_id'].unique())
    progress_printed = []
    all_educators_all_years = {}
    for educator_id in staff_df['public_id'].unique():
        educator_df = staff_df.loc[(staff_df['public_id'] == educator_id)]
        try:
            educator_id = float(educator_id)
        except ValueError:
            print(educator_df.to_string())
            quit()
        for year in educator_df['year'].unique():
            year_df = educator_df.loc[(educator_df['year'] == year)]
            for school_id in year_df['school_id'].unique():
                raw_school_id = school_id
                school_df = year_df.loc[(year_df['school_id'] == school_id)]
                try:
                    school_id = float(school_id)
                    if school_id == -1:
                        print(school_df.to_string())
                        quit()
                except ValueError:
                    if school_id != 'Off-Site':
                        print(f'school_id is not a float: {school_id}')
                    school_id = 9999
                educator_school_id = f'{educator_id}_{school_id}'
                if educator_school_id not in all_educators_all_years:
                    all_educators_all_years[educator_school_id] = {}
                if year not in all_educators_all_years[educator_school_id]:
                    all_educators_all_years[educator_school_id][year] = {}
                for c in df_columns['column_name'].unique():
                    if c not in all_educators_all_years[educator_school_id][year]:
                        all_educators_all_years[educator_school_id][year][c] = None
                    if c in school_df or c == 'category':
                        # for categorical columns get value from consolidated value csv file
                        if c == 'category':
                            raw_value = school_df[f'{c}_description'].values[0]
                        else:
                            raw_value = school_df[c].values[0]

                        consolidated_value = None
                        if c in values_consolidations:
                            vc_df = values_consolidations[c]
                            try:
                                consolidated_value = vc_df[vc_df['value'] == raw_value]['change to'].values[0]
                            except IndexError:
                                print('categorical value error (no consolidated value found):', c, raw_value)
                                quit()
                        else:
                            try:
                                if c == 'annual_salary':
                                    fte_sum = year_df["fte_percentage"].sum()
                                    raw_value = 0
                                    for i in year_df.index:
                                        try:
                                            float_salary = float(year_df.loc[i]['annual_salary'])
                                            if float_salary > raw_value:
                                                raw_value = float_salary # we always just take the biggest salary listed
                                        except ValueError:
                                            print('numerical value error A', c, raw_value)
                                            quit()
                                consolidated_value = float(raw_value)
                            except ValueError:
                                if c in ['annual_salary', 'years_in_education', 'years_in_district']:
                                    print('numerical value error B', c, raw_value)
                                    quit()
                        all_educators_all_years[educator_school_id][year][c] = consolidated_value
                    elif c == 'assignments':
                        # assignments - dictionary of assignments along with fte percentage and position description
                        all_educators_all_years[educator_school_id][year][c] = {}
                        for assignment in school_df['assignment_description'].unique():
                            if assignment not in all_educators_all_years[educator_school_id][year][c]:
                                assignment_dict = {'fte_sum': sum(school_df.loc[school_df['assignment_description']
                                                                                == assignment]['fte_percentage'])}
                                vc_df = values_consolidations['position']
                                position = school_df[school_df['assignment_description'] == assignment]['position_description'].values[0]
                                assignment_dict['position'] = vc_df[vc_df['value'] == position]['change to'].values[0]

                                vc_df = values_consolidations[c]
                                assignment_dict['change to'] = vc_df[vc_df['value'] == assignment]['change to'].values[0]

                                all_educators_all_years[educator_school_id][year][c][assignment] = assignment_dict.copy()
                            else:
                                print("this teacher has the same assignment listed twice at same school in same year")
                                print(school_df.to_string())

        count += 1
        progress = round(count / educator_count * 100)
        if progress not in progress_printed:
            print(progress)
            progress_printed.append(progress)
        # if count > 10:
        #     print(fte_sum_counts)
        #     # for e in all_educators_all_years:
        #     #     for y in all_educators_all_years[e]:
        #     #         print(e, y, all_educators_all_years[e][y])
        #     #     print()
        #     quit()

    # add predictor to dictionary
    print("Adding predictor to dictionary...")
    count = 0
    progress_printed = []
    for teacher_school_id in all_educators_all_years:
        for year in all_educators_all_years[teacher_school_id]:
            if year < 2022:
                if year+1 in all_educators_all_years[teacher_school_id]:
                    all_educators_all_years[teacher_school_id][year]['present_next_year'] = True
                else:
                    all_educators_all_years[teacher_school_id][year]['present_next_year'] = False
            else:
                all_educators_all_years[teacher_school_id][year]['present_next_year'] = None
        count += 1
        progress = round(count / len(all_educators_all_years) * 100)
        if progress not in progress_printed:
            print(progress)
            progress_printed.append(progress)

    print_dict_to_csv(all_educators_all_years, 'all_educators_all_years.csv')

    # export dictionary to dataframe and then to csv
    # this involves manually one hot encoding categorical features
    # and a custom one hot encoding for assignments and positions
    df_dict = {'teacher_id': [], 'school_id': [], 'year': []}
    for c in values_consolidations:
        for v in values_consolidations[c]['change to'].unique():
            df_dict[f'{c}_{v}'] = []
    print("Creating new dictionary for dataframe conversion...")
    count = 0
    progress_printed = []
    for ts_id in all_educators_all_years:
        for y in all_educators_all_years[ts_id]:
            ts_id_split = ts_id.split('_')
            df_dict['teacher_id'].append(ts_id_split[0])
            df_dict['school_id'].append(ts_id_split[1])
            df_dict['year'].append(y)
            for c in all_educators_all_years[ts_id][y]:
                if c in values_consolidations:
                    # categorical columns
                    if c == 'position':
                        pass
                    elif c == 'assignments':
                        for v in values_consolidations['position']['change to'].unique():
                            df_dict[f'position_{v}'].append(0)
                        # custom one hot encode assignments and positions
                        for v in values_consolidations[c]['change to'].unique():
                            possible_vals = values_consolidations[c].loc[values_consolidations[c]['change to'] == v]
                            val = None
                            for possibe_val in possible_vals['value'].unique():
                                if possibe_val in all_educators_all_years[ts_id][y][c]:
                                    val = possibe_val
                                    break
                            if val in all_educators_all_years[ts_id][y][c]:
                                fte_percentage = all_educators_all_years[ts_id][y][c][val]['fte_sum'] / 100
                                df_dict[f'{c}_{v}'].append(fte_percentage)
                                df_dict[f'position_{all_educators_all_years[ts_id][y][c][val]["position"]}'][-1] = fte_percentage
                            else:
                                df_dict[f'{c}_{v}'].append(0)

                    else:
                        for v in values_consolidations[c]['change to'].unique():
                            if all_educators_all_years[ts_id][y][c] == v:
                                df_dict[f'{c}_{v}'].append(1)
                            else:
                                df_dict[f'{c}_{v}'].append(0)
                else:
                    # NOT categorical columns
                    if c not in df_dict:
                        df_dict[c] = []
                    df_dict[c].append(all_educators_all_years[ts_id][y][c])
        count += 1
        progress = round(count / len(all_educators_all_years) * 100)
        if progress not in progress_printed:
            print(progress)
            progress_printed.append(progress)

    # for k in df_dict:
    #     print(k, len(df_dict[k]))

    print("Creating dataframe from new dictionary...")
    teachers_df = pd.DataFrame.from_dict(df_dict)
    # export that the dataframe to a csv

    print("Creating csv from new dataframe...")
    teachers_df.to_csv('dataframes/retention_dataframe.csv')
    print("All done!")
