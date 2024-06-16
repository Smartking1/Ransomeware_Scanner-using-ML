# Ransomware Scanner Using Machine Learning

## Overview
This project aims to develop a ransomware scanner using machine learning techniques. The primary objective is to classify software as either Encryptor Ransomware, Locker Ransomware, or Goodware using various machine learning models. The project involves data loading, preprocessing, feature engineering, and model training and evaluation.

## Table of Contents
- [Installation](#installation)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Support Vector Machine](#support-vector-machine)
- [Results](#results)
- [Conclusion](#conclusion)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/ransomware-scanner.git
    cd ransomware-scanner
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure `Annex.csv` is in the root directory.

## Data Description

The dataset `Annex.csv` contains features extracted from various software samples. The target variable is `family`, which indicates the class of the software:
- `E`: Encryptor Ransomware
- `G`: Goodware
- `L`: Locker Ransomware

### Sample Data

| family | proc_pid | file | urls | type | name | ... |
|--------|----------|------|------|------|------|-----|
| E      | 0        | 0    | 0    | 0    | 0    | ... |
| G      | 0        | 0    | 0    | 0    | 0    | ... |
| L      | 0        | 0    | 0    | 0    | 0    | ... |

### Data Statistics

- Total samples: 1750
- Classes: 3 (`E`, `G`, `L`)

## Data Preprocessing

1. **Handling Missing Values**:
    ```python
    data.isnull().sum()
    ```

    The dataset has no missing values.

2. **Class Distribution**:
    ```python
    data['family'].value_counts()
    ```

    - Goodware (`G`): 820 samples
    - Encryptor Ransomware (`E`): 767 samples
    - Locker Ransomware (`L`): 163 samples

3. **Mapping Classes**:
    ```python
    family_mapping = {'G': 0, 'E': 1, 'L': 2}
    data['family'] = data['family'].map(family_mapping)
    ```

## Modeling

### Logistic Regression

1. **Data Splitting**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Feature Scaling**:
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

3. **Training**:
    ```python
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    ```

4. **Evaluation**:
    ```python
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    ```

### Random Forest Classifier

1. **Training**:
    ```python
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    ```

2. **Evaluation**:
    ```python
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    ```

### Support Vector Machine

1. **Training**:
    ```python
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    ```

2. **Evaluation**:
    ```python
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    ```

## Results

- **Logistic Regression**:
    - Accuracy: 0.89
    - Detailed classification report provided in the script.

- **Random Forest Classifier**:
    - Accuracy: 1.00
    - Detailed classification report provided in the script.

- **Support Vector Machine**:
    - Accuracy: 0.96
    - Detailed classification report provided in the script.

## Conclusion

This project successfully implemented a ransomware scanner using machine learning techniques. Among the models tested, the Random Forest Classifier achieved perfect accuracy, making it the most effective model for this classification task.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the contributors and the open-source community for providing valuable resources and tools for this project.


