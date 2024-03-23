import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, summarize)
from ISLP import confusion_table
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/imputed_custom_retention_dataframe.csv')

    feature_list = ['raise_per_school_years',
                    'english_proficient_advanced_all',
                    'math_test_participation_all',
                    ]

    print(retention_df[(retention_df['teacher_id'] == -189109114)][['year', 'title_i', 'annual_salary', 'years_at_school']])

    # Get rid of o null values in my predictor class
    retention_df = retention_df.dropna(subset=['present_next_year'])

    # Only look at high school teachers
    staff_filter = (retention_df['category_teacher'] == 1) & \
                   (retention_df['position_secondary'] == 1) & \
                   (retention_df['status_active'] == 1)
    retention_df = retention_df[staff_filter]

    features_included = retention_df[feature_list]

    y = retention_df['present_next_year']

    design = MS(features_included)
    X = design.fit_transform(retention_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    train_false_count = y_train.value_counts()[0]
    print(f'{train_false_count}/{len(y_train)} = {train_false_count/len(y_train)}')
    # Applying SMOTE to oversample the minority class in the training set
    smote = SMOTE(sampling_strategy=1, random_state=23)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    train_false_count = y_train_resampled.value_counts()[0]
    print(f'{train_false_count}/{len(y_train_resampled)} = {train_false_count / len(y_train_resampled)}')

    glm_train = sm.GLM(y_train_resampled, X_train_resampled, family=sm.families.Binomial())
    results = glm_train.fit()

    observation = {'title_i': 1,
                   'raise_per_school_years': 1901.0,
                   'highest_degree_masters': 0,
                   'highest_degree_bachelors': 1,
                   'district_type_district': 0,
                   'district_type_charter:': 1}
    observation_df = pd.DataFrame([observation])
    print(observation_df.to_string())
    observation_with_const = sm.add_constant(observation_df, has_constant='add')
    predicted_outcome = results.predict(observation_with_const)
    predicted_class = (predicted_outcome >= 0.5).astype(int)

    print("Predicted probability:", predicted_outcome)
    print("Predicted class:", predicted_class)

    print(summarize(results))
    probs = results.predict(exog=X_test)

    labels = np.array([0]*X_test.shape[0])
    labels[probs > 0.5] = 1
    print(confusion_table(labels, y_test))
    print(np.mean(labels == y_test), np.mean(labels != y_test))
