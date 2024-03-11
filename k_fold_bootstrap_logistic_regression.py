import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

retention_df = pd.read_csv('dataframes/imputed_custom_retention_dataframe.csv')
# Get rid of o null values in my predictor class
retention_df = retention_df.dropna(subset=['present_next_year'])

# Only look at certain teachers
staff_filter = (retention_df['category_teacher'] == 1) & (retention_df['status_active'] == 1) & (retention_df['position_secondary'] == 1)
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

feature_list = ['raise_per_school_years',
                'english_proficient_advanced_all',
                'raise_per_district_years',
                'math_test_participation_all',
                ]

retention_df = retention_df.dropna(subset=feature_list)

features_included = retention_df[feature_list]

y = retention_df['present_next_year']

# leave out test set (stratified)
X_train, X_test, y_train, y_test = train_test_split(features_included, y, test_size=0.2, random_state=23, stratify=y)

get_baseline = True
if get_baseline:
    # Standardize features
    scaler = StandardScaler()
    # Fit the scaler to the training data only, then transform both the training and test sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit model to the entire dataset using balanced class weights
    # Initialize the logistic regression model with balanced class weights
    model = LogisticRegression(solver='saga', penalty='l1', C=1.0, random_state=42, class_weight='balanced', max_iter=5000)

    # Fit the model to the scaled training data
    model.fit(X_train_scaled, y_train)
    # Predict outcomes on the scaled test set
    y_pred = model.predict(X_test_scaled)
    # Evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print()
    print(model.coef_)
# precision is of all the teachers predicted to stay/leave, how many were correct predictions
# recall is of all the teachers that stayed/left, how many were correctly predicted to do that

quit()
# Number of bootstrap samples
n_bootstrap_samples = 100

# K-Fold setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Placeholders for coefficients across all bootstrap samples and folds
coefficients_all = np.zeros((n_bootstrap_samples * n_splits, X_train.shape[1]))

for i in range(n_bootstrap_samples):
    boot_start_time = time.time()
    # Generate a bootstrap sample
    X_boot, y_boot = resample(X_train, y_train, replace=True, n_samples=len(y_train), random_state=i)
    # Convert to NumPy arrays
    X_boot_np = X_boot.values
    y_boot_np = y_boot.values

    # Initialize a manual counter for the folds
    fold = 0
    # Loop through the splits provided by skf.split()
    for train_index, test_index in skf.split(X_boot, y_boot):
        fold_start_time = time.time()
        X_train_fold, y_train_fold = X_boot_np[train_index], y_boot_np[train_index]
        X_test_fold, y_test_fold = X_boot_np[test_index], y_boot_np[test_index]

        # Fit the scaler to the training data only, then transform both the training and test sets
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler.transform(X_test_fold)

        # Initialize and train the Lasso regression model on the scaled training fold data
        # Note: Adjust the 'C' parameter as needed for your specific dataset and goals
        model = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, class_weight='balanced', random_state=42,
                                   C=1.0)
        model.fit(X_train_fold_scaled, y_train_fold)

        # Store the model's coefficients for this fold and bootstrap sample
        idx = i * n_splits + fold
        coefficients_all[idx, :] = model.coef_[0]

        # Increment the fold counter manually
        fold += 1
        print(f'Fold #{fold} finished in {time.time() - fold_start_time:.2f}')
    print(f'Bootstrap sample #{i} finished in {time.time()-boot_start_time:.2f}')

# Average coefficients over all bootstrap samples
coefficients_means = np.mean(coefficients_all, axis=0)
coefficients_stds = np.std(coefficients_all, axis=0)
features = X_train.columns
print()
for feature, coef_mean, coef_std in zip(features, coefficients_means, coefficients_stds):
    print(f"{feature}, {coef_mean:.4f}, {coef_std:.4f}")

