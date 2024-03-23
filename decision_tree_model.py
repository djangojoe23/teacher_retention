import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/imputed_custom_retention_dataframe.csv')

    # Get rid of o null values in my predictor class
    retention_df = retention_df.dropna(subset=['present_next_year'])

    # Only look at certain teachers
    staff_filter = (retention_df['category_teacher'] == 1) & \
                   (retention_df['status_active'] == 1) & \
                   (retention_df['position_secondary'] == 1)
    retention_df = retention_df[staff_filter]

    feature_list = ['gender_M', 'highest_degree_bachelors', 'highest_degree_masters', 'highest_degree_doctoral',
                    'district_type_district', 'district_type_charter', 'district_type_IU', 'district_type_juvenile',
                    'annual_salary', 'years_in_education', 'years_in_district', 'years_at_school', 'school_count',
                    'district_count', 'assignments_count', 'district_types_count', 'degrees_count',
                    'raise_per_school_years', 'raise_per_education_years', 'raise_per_district_years', 'turnover_rate',
                    'rigorous_courses_all', 'title_i', 'regular_attendance_all', 'math_test_participation_all',
                    'math_proficient_advanced_all', 'math_growth_all', 'english_test_participation_all',
                    'english_proficient_advanced_all', 'english_growth_all', 'science_test_participation_all',
                    'science_proficient_advanced_all', 'science_growth_all', 'district_enrollment',
                    'district_charter_enrollment', 'district_school_count', 'district_area',
                    'school_district_male_ratio', 'school_district_female_ratio', 'school_district_white_ratio',
                    'school_district_black_ratio', 'school_district_asian_ratio', 'school_district_hispanic_ratio',
                    'school_district_multiracial_ratio', 'school_district_sped_ratio', 'school_district_gifted_ratio',
                    'school_district_homeless_ratio', 'school_district_foster_care_ratio',
                    'school_district_english_learner_ratio', 'school_district_poor_ratio'
                    ]

    feature_list = ['years_in_education', 'district_charter_enrollment', 'years_in_district', 'annual_salary',
                    'math_proficient_advanced_all', 'raise_per_school_years', 'turnover_rate']

    retention_df = retention_df.dropna(subset=feature_list)

    features_included = retention_df[feature_list]

    y = retention_df['present_next_year']

    # leave out test set (stratified)
    X_train, X_test, y_train, y_test = train_test_split(features_included, y, test_size=0.2, random_state=42, stratify=y)

    # DEFAULT BALANCED CLASS WEIGHTS: {0: 6.361263074970458, 1: 0.5426529225530898}
    # NO CLASS WEIGHTS
    # Model accuracy: 0.9212934296525628
    # Confusion Matrix:
    # [[143  4427]
    #  [149 53421]]
    # Precision: [0.48972603 0.92347186]
    # Recall: [0.03129103 0.99721859]
    custom_class_weights = {0: 6.1,
                            1: 0.5426529225530898}
    # Initialize the Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=23, max_depth=10, min_samples_split=10, criterion='entropy', class_weight=custom_class_weights)

    num_class0 = np.sum(y == 0)
    num_class1 = np.sum(y == 1)
    total_samples = len(y)
    weight_class_0 = total_samples / (2 * num_class0)
    weight_class_1 = total_samples / (2 * num_class1)
    class_weights = {0: weight_class_0, 1: weight_class_1}
    print(class_weights)

    # Train the model
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    # Rearrange feature names so they match the sorted feature importances
    # for feature, importance in zip(feature_list, feature_importances):
    #     print(f"{feature},{importance}")
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # For binary classification or if specifying a specific positive class in a multi-class setting
    precision = precision_score(y_test, y_pred, average=None)  # Adjust pos_label for your positive class
    recall = recall_score(y_test, y_pred, average=None)  # Adjust pos_label accordingly
    print("Precision:", precision)
    print("Recall:", recall)

    # Plot the decision tree
    # Create a figure and axes with tight layout
    # fig, ax = plt.subplots(figsize=(200, 20), dpi=100)
    # plot_tree(model, filled=True, fontsize=5, feature_names=features_included.columns, class_names=['leaver', 'stayer'])
    # plt.subplots_adjust(left=0.0, right=0.99, top=1.0, bottom=0.0, hspace=0, wspace=0)
    # ax.axis('off')
    # plt.show()