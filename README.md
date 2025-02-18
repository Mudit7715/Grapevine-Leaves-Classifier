# Grapevine-Leaves-Image-Classification

## README.md

This project focuses on classifying grapevine leaves using image analysis and machine learning techniques. It utilizes a dataset of grapevine leaf images to train and evaluate different classification models, including Naive Bayes, Support Vector Machines (SVM), and Random Forest.

### Overview

The goal of this project is to accurately classify grapevine leaves into different categories based on their visual features. The project involves the following steps:

1.  **Data Loading and Preprocessing:** Importing the necessary libraries and loading the grapevine leaf image dataset.
2.  **Feature Extraction:** Defining a function to preprocess images and extract relevant features, such as color histograms from the HSV color space.
3.  **Model Training and Evaluation:** Training and evaluating several machine learning models, including:

    *   Gaussian Naive Bayes
    *   Support Vector Machine (SVM) with hyperparameter tuning using GridSearchCV
    *   Random Forest Classifier
4.  **Performance Analysis:** Comparing the accuracy and classification reports of the different models to determine the best performing one. Confusion matrices are also visualized to analyze the classification results.

### Dependencies

The following Python libraries are required to run this project:

*   `cv2` (OpenCV): For image processing.
*   `os`: For interacting with the operating system, such as listing directories and joining paths.
*   `numpy`: For numerical computations and array manipulation.
*   `seaborn`: For data visualization, specifically for creating heatmaps of confusion matrices.
*   `sklearn` (scikit-learn): For machine learning tasks, including:

    *   `SVC` (Support Vector Classifier)
    *   `GaussianNB` (Gaussian Naive Bayes)
    *   `RandomForestClassifier`
    *   `train_test_split`: For splitting the dataset into training and testing sets.
    *   `GridSearchCV`: For hyperparameter tuning of the SVM model.
    *   `accuracy_score`, `classification_report`, `confusion_matrix`: For evaluating model performance.
    *   `StandardScaler`: For feature scaling (though not explicitly used in the provided code).
*   `matplotlib`: For plotting graphs and visualizations.

### Data

The project uses the "Grapevine Leaves Image Dataset". The dataset should be organized into directories, where each directory represents a different category of grapevine leaf.

*   `Grapevine_Leaves_Image_Dataset`: This is the root directory.

    *   `Category_1`: Contains images of grapevine leaves belonging to the first category.
    *   `Category_2`: Contains images of grapevine leaves belonging to the second category.
    *   `Category_n`: more categories...

### Usage

1.  **Clone the repository:**

    ```
    git clone [repository link]
    cd [repository name]
    ```

2.  **Install the required libraries:**

    ```
    pip install opencv-python numpy scikit-learn seaborn matplotlib
    ```

3.  **Organize the dataset:**

    *   Download the Grapevine Leaves Image Dataset and place it in a directory.
    *   Update the `dataset_dir` variable in the ipynb file.
        ```
        dataset_dir = "/path/to/your/dataset" # Replace with the actual path
        ```

4.  **Run the ipynb file:**

    *   Open the `ml-project-1.ipynb` file using Jupyter Notebook or JupyterLab.
    *   Execute the cells sequentially to perform the analysis and train the models.

### Results

The project evaluates the performance of three different classification models: Gaussian Naive Bayes, Support Vector Machines (SVM), and Random Forest. The results are presented in terms of accuracy and classification reports.

*   **Naive Bayes:** Achieved an accuracy of 62.0%.
*   **Support Vector Machine (SVM):** Achieved an accuracy of 80.0% after hyperparameter tuning.
*   **Random Forest:** Achieved the highest accuracy of 84.0%.
The confusion matrices are visualized for each model to provide insights into the classification accuracy for each category.

### Conclusion

The Random Forest classifier outperformed the other models in terms of accuracy for the grapevine leaf classification task. The SVM model also showed promising results after hyperparameter tuning.
