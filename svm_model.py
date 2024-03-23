from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

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

    feature_list = ['years_in_education']

    retention_df = retention_df.dropna(subset=feature_list)

    X = retention_df[feature_list]
    y = retention_df['present_next_year']

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a SVM Classifier
    clf = SVC(kernel='rbf', C=1, gamma=1)  # Example parameters

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))
