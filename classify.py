"""
Heart Disease Prediction
"""

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

np.set_printoptions(suppress=True)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to scale the data
def scaling(df):
    # Standardize the data
    cols = ['HeartDisease', 'Sex', 'ChestPainType', 'FastingBS',
            'RestingECG', 'ExerciseAngina', 'ST_Slope']
    X = df.drop(cols, axis=1)
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X), columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'])
    # concatenate scaled data with other categorical variables
    std_data = pd.concat([df[cols], X_std], axis=1)
    return std_data, X_std


# Function for creating dummy variables
def dummy_features(df):
    # Get list of object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    # Create dummy variables for each object column
    for col in obj_cols:
        dummy_vars = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy_vars], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df



# Function to create correlation plot
def corr_plot(df):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


# Function for Liear regression
def linear_regression(df, dependent_var, baseline=True):
    if baseline:
        # Dropping columns with only 2 dummy variables to avoid perfect multicollinearity
        df = df.drop(columns=['Sex_F', 'ExerciseAngina_N'])
    else:
        df = df
    # Define X (predictor) and y (target) variables
    X = df.drop([dependent_var], axis=1)
    y = df[dependent_var]
    # Add constant to X for intercept term
    X = sm.add_constant(X)
    # Fit linear regression model
    model = sm.OLS(y, X).fit()
    # Print summary of regression results
    print(model.summary())
    # Get p-values for each feature
    p_values = model.summary2().tables[1]['P>|t|']

    # Extract features with p-value less than 0.05
    selected_features = p_values[p_values < 0.05].index
    # Calculate the VIF values for each column
    vif = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    # Print the VIF values
    for i in range(len(X.columns)):
        print(f"{X.columns[i]} VIF: {vif[i]:.2f}")
    # Print selected features
    print(selected_features)
    final_X = X[selected_features]
    final_data = pd.concat([final_X, y], axis=1)
    return final_data


# Function to perform PCA on data
def pca_info(df):
    # Perform PCA for finding the variable explaining most variance
    pca = PCA()
    pca.fit(df)

    # Print explained variance for each principal component
    print('Explained variance ratio:', pca.explained_variance_ratio_)

    # Print top contributing features for first 3 principal components
    n_top_features = 2
    for i, component in enumerate(pca.components_[:3]):
        top_features_idx = abs(component).argsort()[::-1][:n_top_features]
        top_features = df.columns[top_features_idx]
        print(f'Top {n_top_features} features for principal component {i + 1}:')
        print(top_features)
        print('\n')


# Creating polynomial features
def polynomial_features(df):
    # Not including categorical dummy variables
    df1 = df.drop(
        columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS', 'HeartDisease'],
        axis=1)
    # Create a new PolynomialFeatures object with degree=3
    poly = PolynomialFeatures(degree=3, include_bias=False)
    # Fit and transform the original features to polynomial features
    df_poly = pd.DataFrame(poly.fit_transform(df1), columns=poly.get_feature_names_out(df1.columns))
    df_poly = pd.concat([df_poly, df['HeartDisease']], axis=1)
    return df_poly


# Function to find optimal K
def optimal_k(df):
    # Determine optimal k using elbow method
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    # Plot the elbow curve
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()


# Function to create clusters
def create_clusters(df, k):
    # Fit k-means clustering for optimal K value
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)

    # Add cluster labels to the data
    df['Cluster'] = kmeans.labels_
    # Get the mean values for each cluster
    cluster_means = df.groupby('Cluster')['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'].mean()

    # Print the mean values for each cluster
    print(cluster_means)


# function to pre-process the train and test data
def pre_process(df, final_input, select_features = True):
    # Changing FastingBP to categorical
    df['FastingBS'] = df['FastingBS'].astype(object)
    # Creating polynomial Features
    poly_data = polynomial_features(df)
    # Create dummy variables
    dummy_data = dummy_features(df)
    if select_features:
        merged_data = pd.concat([poly_data, dummy_data], axis=1).drop_duplicates()[final_input]
    else:
        merged_data = pd.concat([poly_data, dummy_data], axis=1).drop_duplicates()
    X = merged_data.iloc[:, :-1]
    y = merged_data.iloc[:, -1]

    return X, y


# Function for creating dummy variables
def label_encoding(train_df, test_df):
    # Get list of object columns
    obj_cols = train_df.select_dtypes(include=['object']).columns
    # print("label encoding: ", obj_cols)
    # Create dummy variables for each object column
    for col in obj_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    return train_df, test_df


def preprocess2(train_df, test_df, poly_features = True, encoding = 'onehot', scale=True):
    # Changing FastingBP to categorical
    train_df['FastingBS'] = train_df['FastingBS'].astype(object)
    test_df['FastingBS'] = test_df['FastingBS'].astype(object)

    if scale:
        train_df, scaled_numeric_data = scaling(train_df)
        test_df, scaled_numeric_data = scaling(test_df)

    if encoding == 'onehot':
        # Create dummy variables
        dummy_data_train = dummy_features(train_df.drop(columns=['HeartDisease']))
        dummy_data_test = dummy_features(test_df.drop(columns=['HeartDisease']))
    elif encoding == 'label':
        dummy_data_train, dummy_data_test = label_encoding(train_df.drop(columns=['HeartDisease']),
                                                           test_df.drop(columns=['HeartDisease']))
    if poly_features:
        # Creating polynomial Features
        # print(df.columns)

        poly_data_train = polynomial_features(train_df)
        poly_data_test = polynomial_features(test_df)

        merged_data_train = pd.concat([poly_data_train, dummy_data_train], axis=1).drop_duplicates()
        merged_data_test = pd.concat([poly_data_test, dummy_data_test], axis=1).drop_duplicates()

    else:
        merged_data_train = dummy_data_train
        merged_data_train['HeartDisease'] = train_df['HeartDisease']
        merged_data_test = dummy_data_test
        merged_data_test['HeartDisease'] = test_df['HeartDisease']

    # print(merged_data.columns)
    X_train = merged_data_train.drop(columns=['HeartDisease']).to_numpy()
    y_train = train_df['HeartDisease'].to_numpy()
    X_test = merged_data_test.drop(columns=['HeartDisease']).to_numpy()
    y_test = test_df['HeartDisease'].to_numpy()
    # print(X, y)

    return X_train, y_train, X_test, y_test,  merged_data_train.drop(columns=['HeartDisease']).columns


def metrics(model, y_train, y_train_pred, X_test, y_test, y_pred):
    # Calculate and print the confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    # Calculate various performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    train_error = 1 - accuracy_train
    test_error = 1 - accuracy
    variance = test_error - train_error

    # Print the performance metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    print('Bias:', train_error)
    print('Variance:', variance)

    # Generate a plot of the ROC curve and calculate the AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    print('Area under the curve: ', auc)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUC = {:.3f})'.format(auc))
    plt.show()

    # Calculate and plot the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


# Function for logistic regression
def logistic_reg(X_train, y_train, X_test, y_test):
    # Train a logistic regression model on the training data
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)


# Function for Decision tree
def decision_tree(X_train, y_train, X_test, y_test, X_cols):
    # Train a decision tree model on the training data
    model = DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_split=3)
    model.fit(X_train, y_train)

    fig = plt.figure(figsize=(15, 8))
    _ = plot_tree(model,
                       feature_names=X_cols,
                       class_names=['No Disease', "Disease"],
                       filled=True,
                        max_depth=2)
    plt.show()
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)



# Function for SVM
def svm(X_train, y_train, X_test, y_test):
    # Train an SVM model on the training data
    model = SVC(random_state=0)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)


# Function for Naive bayes
def naive_bayes(X_train, y_train, X_test, y_test):
    # Train an naive bayes model on the training data
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)


# Function for adaboost
def adaboost(X_train, y_train, X_test, y_test):
    # Train an Adaboost model on the training data
    model = AdaBoostClassifier(n_estimators=5)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)


# Function for Random Forest
def random_forest(X_train, y_train, X_test, y_test):
    # Train an SVM model on the training data
    model = RandomForestClassifier(n_estimators=40, random_state=100)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)


# Function for Random Forest tuning
def random_forest_tuned(X_train, y_train, X_test, y_test):
    n_estimators = [10, 20, 40, 100]
    max_features = ['auto', 'sqrt']
    max_depth = [2, 4, 5, 8, None]
    min_samples_split = [2, 3, 5]
    min_samples_leaf = [1, 2, 4]

    params_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }
    # Train a random forest model on the training data
    model = RandomForestClassifier(n_estimators=40, random_state=42)
    model_cv = GridSearchCV(model, params_grid, scoring="recall", cv=5, verbose=1, n_jobs=-1)
    model_cv.fit(X_train, y_train)

    best_params = model_cv.best_params_
    print(f"Best parameters: {best_params}")

    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)


# Function for Adaboost tuning
def adaboost_tuned(X_train, y_train, X_test, y_test):
    n_estimators = [10, 20, 40, 100]
    algorithm = ['SAMME', 'SAMME.R']
    lr = [0.01, 0.1, 0.5, 0.9, 1.0]
    params_grid = {
        'n_estimators': n_estimators,
        'algorithm': algorithm,
        'learning_rate': lr
    }
    # Train a adaboost model on the training data
    model = AdaBoostClassifier(n_estimators=5, random_state=100)
    model_cv = GridSearchCV(model, params_grid, scoring="recall", cv=5, verbose=1, n_jobs=-1)
    model_cv.fit(X_train, y_train)

    best_params = model_cv.best_params_
    print(f"Best parameters: {best_params}")

    model = AdaBoostClassifier(**best_params, random_state=100)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)

# Function for decision tree tuning
def decision_tree_tuned(X_train, y_train, X_test, y_test):
    criterion = ['gini', 'entropy']
    max_depth = [2, 4, 6, 8, 10, 12]
    max_features = ['sqrt', 'log2', None]
    min_samples_leaf = [1, 2]
    params_grid = {
        'criterion': criterion,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf
    }
    # Train a decision tree model on the training data
    model = DecisionTreeClassifier(random_state=100)
    model_cv = GridSearchCV(model, params_grid, scoring="recall", cv=5, verbose=1, n_jobs=-1)
    model_cv.fit(X_train, y_train)

    best_params = model_cv.best_params_
    print(f"Best parameters: {best_params}")

    model = DecisionTreeClassifier(**best_params, random_state=100)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    metrics(model, y_train, y_train_pred, X_test, y_test, y_pred)

# Main function
def main():
    # Read data
    data = pd.read_csv('data/heart_train_718.csv')
    train = pd.read_csv('data/heart_train_718.csv')
    test = pd.read_csv('data/heart_test_200.csv')
    k = 3
    # Changing FastingBP to categorical
    data['FastingBS'] = data['FastingBS'].astype(object)
    # Creating polynomial Features
    poly_data = polynomial_features(data)
    # Create dummy variables
    dummy_data = dummy_features(data)
    # Fitting baseline Linear Regression to find the relationship
    baseline_input = linear_regression(dummy_data, 'HeartDisease')
    # fitting linear regression on polynomial features
    poly_input = linear_regression(poly_data, 'HeartDisease', False)
    merged_data = pd.concat([baseline_input, poly_input], axis=1)
    merged_data = merged_data.iloc[:, :-1]
    # merged_data=merged_data.drop(['RestingBP Oldpeak','MaxHR Oldpeak^2','Age MaxHR^2','Age RestingBP MaxHR','RestingECG_LVH'],axis=1)
    final_data = linear_regression(merged_data, 'HeartDisease', False)
    # Show correlation Plot
    # corr_plot(dummy_data)
    # Scale the data
    data, scaled_numeric_data = scaling(data)
    # Optimal k using elbow method
    # optimal_k(scaled_numeric_data)
    # # Creating clusters
    # create_clusters(scaled_numeric_data, k)
    # # PCA
    # pca_info(scaled_numeric_data)
    # Final inputs after doing linear regression, k-means, PCA
    final_input = final_data.columns.drop('const')
    # Index(['const', 'Cholesterol', 'Oldpeak', 'Sex_M', 'ChestPainType_ASY',
    #        'FastingBS_1', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up',
    #        'Age Cholesterol^2'],
    #       dtype='object')

    # print(final_input)
    # Preprocessing and creating new attributes in train and test data
    # X_train, y_train = pre_process(train, final_input)
    # X_test, y_test = pre_process(test, final_input)
    # Models tried: without polynomial features, with one hot encoding, with label encoding
    X_train, y_train, X_test, y_test, X_train_cols = preprocess2(train, test, poly_features=False, encoding='label'
                                                                 , scale=False)
    # Running Logistic regression
    # logistic_reg(X_train, y_train, X_test, y_test)
    # Takes too long
    # svm(X_train, y_train, X_test, y_test)


    # decision_tree(X_train, y_train, X_test, y_test, X_train_cols) # try without poly features
    decision_tree_tuned(X_train, y_train, X_test, y_test)

    # naive_bayes(X_train, y_train, X_test, y_test)
    #
    # adaboost(X_train, y_train, X_test, y_test)
    # adaboost_tuned(X_train, y_train, X_test, y_test)


    #best: label encoding, no polynomial
    # random_forest(X_train, y_train, X_test, y_test)
    # random_forest_tuned(X_train, y_train, X_test, y_test)







if __name__ == '__main__':
    main()
