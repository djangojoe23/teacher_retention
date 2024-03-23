import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


if __name__ == '__main__':
    retention_df = pd.read_csv('dataframes/catboost_dataframe.csv')

    retention_df = retention_df.dropna()

    feature_list = ['gender', 'highest_degree',
                    'district_type', 'assignment',
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

    X = retention_df[feature_list]
    y = retention_df['present_next_year']
    # X and y are your features and target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Identify categorical features for CatBoost
    cat_features = ['gender', 'assignment', 'district_type', 'highest_degree']

    # Initialize the CatBoostClassifier
    model = CatBoostClassifier(iterations=1000,
                               auto_class_weights='Balanced',
                               learning_rate=0.1,
                               depth=6,
                               eval_metric='Logloss',
                               use_best_model=True,
                               random_seed=3,
                               verbose=100)

    # Convert the test dataset to Pool to speed up the evaluation
    eval_dataset = Pool(X_test, y_test, cat_features=cat_features)

    # Train the model
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=eval_dataset)

    # Extract evaluation results
    results = model.get_evals_result()

    # Make predictions
    y_pred = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # For binary classification or if specifying a specific positive class in a multi-class setting
    precision = precision_score(y_test, y_pred, average=None)  # Adjust pos_label for your positive class
    recall = recall_score(y_test, y_pred, average=None)  # Adjust pos_label accordingly
    print("Precision:", precision)
    print("Recall:", recall)

    # accuracy = results['validation']['Accuracy']
    # error = [1 - acc for acc in accuracy]
    # # Number of trees used (x-axis)
    # iterations = range(1, len(error) + 1)

    # Logloss values
    logloss = results['validation']['Logloss']
    # Number of trees used (x-axis)
    iterations = range(1, len(logloss) + 1)

    plt.figure(figsize=(10, 6))
    # plt.plot(iterations, error, marker='o', linestyle='-', color='blue')
    plt.plot(iterations, logloss, marker='o', linestyle='-', color='red')
    plt.title('Test Classification Error (blue) and Logloss (red) vs. Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Classification Error (blue) and Logloss (red)')
    plt.grid(True)
    plt.show()

    feature_importances = model.get_feature_importance()
    feature_names = X_train.columns  # Assuming X_train is a DataFrame
    importance_scores = dict(zip(feature_names, feature_importances))

    # Sort the features by importance
    sorted_importance_scores = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    # Display the sorted feature importance scores
    for feature, score in sorted_importance_scores:
        print(f"{feature}: {score}")

    # Create the explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(Pool(X_test, cat_features=cat_features))

    # Summarize the SHAP values in a plot
    shap.summary_plot(shap_values, X_test)