"""
convert all files in excel_data to csv files then convert those csv files into 3 separate dataframes
1 Datafile All Years
2 DistrictFastFacts All Years
3 Individual Staff All Years

Export those dataframe to csv files so the create_dataframe script can turn them all into a single dataframe
"""
import pandas as pd


def convert_xlsx_to_csv(xlsx_filepath, csv_filepath):
    xls = pd.ExcelFile(xlsx_filepath)
    df_from_sheets = []
    for sheet in xls.sheet_names:
        df_from_sheets.append(pd.read_excel(xls, sheet_name=sheet))
    all_sheets_df = pd.concat(df_from_sheets)
    all_sheets_df.to_csv(csv_filepath)  # if a value has commas in it, pandas will enclose it in quotes

    return all_sheets_df


if __name__ == '__main__':
    for file_name in ['Datafile', 'DistrictFastFacts']:
        print(file_name)
        try:
            df = pd.read_csv(f'dataframes/{file_name}_df.csv')
        except FileNotFoundError:
            raw_df_dict = {}
            for year in range(2017, 2023):
                try:
                    df = pd.read_csv(f'csv_data/{file_name}_{year}{year + 1}.csv')
                except FileNotFoundError:
                    print('Converting', file_name, year)
                    df = convert_xlsx_to_csv(f'excel_data/{file_name}_{year}{year + 1}.xlsx',
                                             f'csv_data/{file_name}_{year}{year+1}.csv')

                df.rename(columns={'DistrictName': 'district_name', 'districtname': 'district_name',
                                   'name': 'school_name', 'Name': 'school_name', 'SchoolName': 'school_name',
                                   'AUN': 'district_id', 'aun': 'district_id',
                                   'schl': 'school_id', 'Schl': 'school_id',
                                   'DataElement': 'data_element', 'dataelement': 'data_element',
                                   'DisplayValue': 'display_value', 'displayvalue': 'display_value'}, inplace=True)
                raw_df_dict[year] = df

            all_years_dict = {}
            id_column = 'school_id'
            if file_name == 'DistrictFastFacts':
                id_column = 'district_id'
            for y in raw_df_dict:
                for uid in raw_df_dict[y][id_column].unique():
                    df = raw_df_dict[y].loc[raw_df_dict[y][id_column] == uid]
                    if uid not in all_years_dict:
                        all_years_dict[uid] = {y: {}}
                    elif y not in all_years_dict[uid]:
                        all_years_dict[uid][y] = {}
                    all_years_dict[uid][y]['district_name'] = df['district_name'].unique()[0]
                    all_years_dict[uid][y]['district_id'] = df['district_id'].unique()[0]
                    for d in df['data_element'].unique():
                        split_f = d.strip().split(' ')
                        cleaned_d = ' '.join([i for i in split_f if i])
                        try:
                            all_years_dict[uid][y][cleaned_d] = df.loc[df['data_element'] == d]['display_value'].values[0]
                        except IndexError:
                            print(y, uid, d)

            column_names_df = pd.read_csv(f'column_names_{file_name}.tsv', sep='\t')
            all_years_dict_for_df = {id_column: [], 'year': [], 'district_name': []}
            if file_name == "Datafile":
                all_years_dict_for_df['district_id'] = []
            for uid in all_years_dict:
                for y in all_years_dict[uid]:
                    all_years_dict_for_df[id_column].append(uid)
                    all_years_dict_for_df['year'].append(y)
                    if file_name == 'Datafile':
                        all_years_dict_for_df['district_name'].append(all_years_dict[uid][y]['district_name'])
                        all_years_dict_for_df['district_id'].append(all_years_dict[uid][y]['district_id'])
                    for c in column_names_df['Column']:
                        c_name_change = column_names_df.loc[column_names_df['Column'] == c]['Change to'].values[0]
                        if c_name_change != 'ignore':
                            data_val = 'none'
                            try:
                                data_val = all_years_dict[uid][y][c]
                                data_val = float(data_val)
                            except (KeyError, ValueError):
                                pass
                            if c_name_change not in all_years_dict_for_df:
                                all_years_dict_for_df[c_name_change] = []
                            all_years_dict_for_df[c_name_change].append(data_val)
            all_years_df = pd.DataFrame.from_dict(all_years_dict_for_df)
            all_years_df.to_csv(f'dataframes/{file_name}_df.csv')

    print('IndividualStaffReport')
    try:
        df = pd.read_csv(f'dataframes/IndividualStaffReport_df.csv')
    except FileNotFoundError:
        raw_df_dict = {}
        for year in range(2013, 2023):
            try:
                df = pd.read_csv(f'csv_data/IndividualStaffReport_{year}{year + 1}.csv', index_col=0)
            except FileNotFoundError:
                print('Converting', 'Individual Staff Report', year)
                file_ext = 'xlsb'
                if year > 2019:
                    file_ext = 'xlsx'
                df = convert_xlsx_to_csv(f'excel_data/{year}-{year + 1 - 2000} Professional Personnel Individual Staff '
                                         f'Report.{file_ext}', 'csv_data/IndividualStaffReport_{year}{year + 1}.csv')
            raw_df_dict[year] = df

        column_names_df = pd.read_csv(f'column_names_IndividualStaffReport.csv')
        all_years_dict_for_df = {'year': []}
        for y in raw_df_dict:
            print(y)
            i_count = 0
            for i in raw_df_dict[y].index:
                i_count += 1
                if i_count % 10000 == 0:
                    print(i)
                    i_count = 0
                row_df = raw_df_dict[y].iloc[i]
                all_years_dict_for_df['year'].append(y)
                for c in raw_df_dict[y].columns:
                    c_name_change = column_names_df.loc[column_names_df['Column'] == c]['Change to'].values[0]
                    if c_name_change != 'ignore':
                        if c_name_change not in all_years_dict_for_df:
                            all_years_dict_for_df[c_name_change] = []
                        all_years_dict_for_df[c_name_change].append(row_df[c])

        for c in all_years_dict_for_df:
            print(c, len(all_years_dict_for_df[c]))
        all_years_df = pd.DataFrame.from_dict(all_years_dict_for_df)
        all_years_df.to_csv(f'dataframes/IndividualStaffReport_df.csv')

