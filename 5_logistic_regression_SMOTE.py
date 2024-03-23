import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, summarize)
from ISLP import confusion_table
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/imputed_custom_retention_dataframe.csv')
    original_row_count = retention_df.shape[0]

    #Get rid of o null values in my predictor class
    retention_df = retention_df.dropna(subset=['present_next_year'])

    #Only look at high school teachers
    staff_filter = (retention_df['category_teacher'] == 1) & \
                   (retention_df['position_secondary'] == 1) & \
                   (retention_df['status_active'] == 1)
    retention_df = retention_df[staff_filter]
    possible_features = ['district_type_district', 'district_type_charter', 'district_type_IU', 'district_type_juvenile',
                         'gender_M', 'gender_F',
                         'highest_degree_bachelors', 'highest_degree_masters', 'highest_degree_doctoral',
                         'annual_salary', 'raise_per_education_years', 'raise_per_district_years', 'raise_per_school_years',
                         'school_count', 'district_count', 'assignments_count', 'district_types_count', 'degrees_count',
                         'turnover_rate', 'title_i',
                         'rigorous_courses_all', 'regular_attendance_all',
                         'math_test_participation_all', 'science_test_participation_all', 'english_test_participation_all',
                         'math_growth_all', 'science_growth_all', 'english_growth_all',
                         'math_proficient_advanced_all', 'science_proficient_advanced_all', 'english_proficient_advanced_all',
                         'district_enrollment', 'district_school_count', 'district_charter_enrollment',
                         'district_area',
                         'school_district_male_ratio', 'school_district_female_ratio', 'school_district_white_ratio',
                         'school_district_black_ratio', 'school_district_hispanic_ratio', 'school_district_asian_ratio',
                         'school_district_multiracial_ratio', 'school_district_gifted_ratio', 'school_district_sped_ratio',
                         'school_district_english_learner_ratio', 'school_district_poor_ratio', 'school_district_homeless_ratio',
                         'school_district_foster_care_ratio'
                         ]



    results_headers = ['z_score', 'p_value', 'accuracy', 'true_leaver_predictions',
                       'false_leaver_predictions', 'leaver_prediction_accuracy', 'true_stayer_predictions',
                       'false_stayer_predictions', 'stayer_prediction_accuracy']
    reg_results = {}

    model_features = ['title_i', 'raise_per_school_years']

    for f in possible_features:
        print(f)
        if f not in model_features:
            feature_list = model_features.copy()
            feature_list.append(f)
            reg_results[f] = {}
            for h in results_headers:
                if h in ['z_score', 'p_value']:
                    for x in feature_list:
                        reg_results[f][f'{x}_{h}'] = []
                else:
                    reg_results[f][h] = []

            retention_df = retention_df.dropna(subset=feature_list)

            features_included = retention_df[feature_list]

            y = retention_df['present_next_year']

            design = MS(features_included)
            X = design.fit_transform(retention_df)
            for i in range(0, 10):
                r_state = random.randint(0, 9999)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=r_state)

                train_false_count = y_train.value_counts()[0]

                # Applying SMOTE to oversample the minority class in the training set
                smote = SMOTE(sampling_strategy=1, random_state=r_state)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                glm_train = sm.GLM(y_train_resampled, X_train_resampled, family=sm.families.Binomial())
                results = glm_train.fit()
                for x in feature_list:
                    reg_results[f][f'{x}_p_value'].append(results.pvalues[x])
                    reg_results[f][f'{x}_z_score'].append(results.tvalues[x])
                probs = results.predict(exog=X_test)

                labels = np.array([0]*X_test.shape[0])
                labels[probs > 0.5] = 1

                ct = confusion_table(labels, y_test)
                reg_results[f]['true_leaver_predictions'].append(ct[0][0])
                reg_results[f]['false_leaver_predictions'].append(ct[1][0])
                reg_results[f]['leaver_prediction_accuracy'].append(ct[0][0]/(ct[0][0] + ct[1][0]))
                reg_results[f]['true_stayer_predictions'].append(ct[1][1])
                reg_results[f]['false_stayer_predictions'].append(ct[0][1])
                reg_results[f]['stayer_prediction_accuracy'].append(ct[1][1]/(ct[1][1] + ct[0][1]))
                reg_results[f]['accuracy'].append(np.mean(labels == y_test))

            for h in results_headers:
                if h in ['p_value', 'z_score']:
                    for x in feature_list:
                        reg_results[f][f'avg_{x}_{h}'] = sum(reg_results[f][f'{x}_{h}']) / len(reg_results[f][f'{x}_{h}'])
                else:
                    reg_results[f][f'avg_{h}'] = sum(reg_results[f][h]) / len(reg_results[f][h])

            header_str = ''
            for c in reg_results:
                header_str = 'coefficient'
                for v in reg_results[c]:
                    if v.startswith('avg'):
                        header_str += f',{v}'
                break
            print(header_str)

            for c in reg_results:
                row = f'{c}'
                for v in reg_results[c]:
                    if v.startswith('avg'):
                        row += f',{reg_results[c][f"{v}"]}'
                print(row)
            print()
