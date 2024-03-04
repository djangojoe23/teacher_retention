import pandas as pd
import numpy as np
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, summarize)
from ISLP import confusion_table

if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/retention_dataframe.csv')
    # corr_df = retention_df.corr()
    # for c in corr_df.columns:
    #     for i in range(0, corr_df[c].count()):
    #         print(f'{c}, {corr_df.columns[i]}, {corr_df[c].iloc[i]}')

    #CLEAN UP THE DATA A BIT
    retention_df.dropna(subset=['present_next_year', 'annual_salary'], inplace=True)
    retention_df['present_next_year'].replace(True, 1, inplace=True)
    retention_df['present_next_year'].replace(False, 0, inplace=True)

    #UNDERSAMPLE THE MAJORITY CLASS
    true_count = retention_df['present_next_year'].value_counts()[1]
    false_count = retention_df['present_next_year'].value_counts()[0]
    true_training_indices = retention_df[retention_df['present_next_year'] == 1].sample(n=round(false_count*.75), random_state=123).index
    false_training_indices = retention_df[retention_df['present_next_year'] == 0].sample(n=round(false_count*.75), random_state=123).index
    combined_training_indices = true_training_indices.union(false_training_indices)
    print(len(combined_training_indices))

    # all_features = retention_df[['annual_salary', 'gender_M', 'years_in_education']]
    all_features = retention_df[['annual_salary',
                                 'year',
                                 'gender_M',
                                 'years_in_education',
                                 'years_in_district',
                                 'assignments_elementary',
                                 'assignments_learning_support',
                                 'assignments_kindergarten',
                                 'assignments_math',
                                 'assignments_english',
                                 'assignments_music',
                                 'assignments_health',
                                 'assignments_history',
                                 'assignments_science',
                                 'assignments_art',
                                 'assignments_foreign_language',
                                 'assignments_business',
                                 'assignments_home_ec',
                                 'assignments_esl',
                                 'assignments_technology',
                                 'assignments_pre_k',
                                 'assignments_social_science',
                                 'assignments_drama',
                                 'assignments_shop_class',
                                 'district_type_district',
                                 'district_type_charter',
                                  # 'district_type_IU',
                                  # 'district_type_CTC',
                                  # 'district_type_juvenile'
                                 'highest_degree_bachelors',
                                 'highest_degree_masters',
                                 'highest_degree_doctoral',
                                 'status_active',
                                 'status_sabbatical',
                                 'status_workers_comp',
                                 'status_military_leave',
                                 'status_suspension',
                                 'position_secondary',
                                 'position_elementary',
                                 'position_learning_support',
                                 'position_ungraded teacher',
                                 ]]

    y = retention_df['present_next_year']

    design = MS(all_features)
    X = design.fit_transform(retention_df)
    # for training data, need to combine the row_indices sampled with a sample where present_next_year == 0
    X_train = X.loc[combined_training_indices]
    X_test = X[~X.index.isin(combined_training_indices)]
    test_length = X_test.shape[0]
    print(test_length)

    y_train = y.loc[combined_training_indices]
    y_test = y[~y.index.isin(combined_training_indices)]
    glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial())
    results = glm_train.fit()
    print(summarize(results))
    probs = results.predict(exog=X_test)

    labels = np.array([0]*test_length)
    labels[probs > 0.5] = 1
    print(confusion_table(labels, y_test))
    print(np.mean(labels == y_test), np.mean(labels != y_test))
