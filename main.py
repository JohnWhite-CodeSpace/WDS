import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def load_csv(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError as ex:
        print(f"There is no such file as{filepath}!: {ex}")
        return None


def normalize_and_standardize_dataset(data, label):
    x = data.drop(columns=[label])
    y = data[label]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x, y, x_scaled


def perform_PCA(x_train, x_test, n_components=2):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    pca_evr = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {pca_evr}")
    return x_train_pca, x_test_pca, pca_evr


def split_dataset(x, y, test_size=0.3, random_state=40, stratify=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        stratify=stratify)
    return x_train, y_train, x_test, y_test


# def balance_dataset(x_train, y_train):
#     smote = SMOTE(random_state=42)
#     x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
#     return x_train_balanced, y_train_balanced


def perform_LDA(x_train, y_train, x_test, y_test):
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(x_train, y_train)
    y_pred = LDA.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification report: \n{classification_report(y_test, y_pred, zero_division=1)}")
    return accuracy, y_pred


def perform_naive_bayes(x_train, y_train, x_test, y_test):
    naive_bayes = GaussianNB()
    naive_bayes.fit(x_train, y_train)
    y_pred = naive_bayes.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification report: \n{classification_report(y_test, y_pred, zero_division=1)}")
    return accuracy, y_pred


def perform_svm(x_train, y_train, x_test, y_test, kernel, gamma):
    svm = SVC(kernel=kernel, gamma=gamma)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification report: \n{classification_report(y_test, y_pred, zero_division=1)}")
    return accuracy, y_pred


def perform_knn(x_train, y_train, x_test, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification report: \n{classification_report(y_test, y_pred, zero_division=1)}")
    return accuracy, y_pred

def plot_pca_variance(explained_variance_ratio):
    cumulative_variance = np.cumsum(explained_variance_ratio)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6,
            label="Explained Variance per Component", color="blue")
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Explained Variance Ratio", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color="red",
             label="Cumulative Explained Variance")
    ax2.set_ylabel("Cumulative Explained Variance", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    plt.title("PCA Explained Variance & Component Histogram")
    fig.tight_layout()
    plt.show()

def plot_class_distribution(y):
    plt.figure(figsize=(6, 4))
    print(y.value_counts())
    sns.countplot(x=y)
    plt.xlabel("Class Labels")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.show()


def plot_correlation_matrix(data):
    data_numeric = data.copy()
    if data_numeric.dtypes[-1] == 'object':
        data_numeric[data_numeric.columns[-1]] = data_numeric[data_numeric.columns[-1]].astype('category').cat.codes

    plt.figure(figsize=(10, 8))
    sns.heatmap(data_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation Matrix")
    plt.show()


def init_whole_process(filename, label, n_components=None):
    data = load_csv(filename)
    x, y, x_scaled = normalize_and_standardize_dataset(data, label=label)
    x_train, y_train, x_test, y_test = split_dataset(x_scaled, y, test_size=0.2, random_state=40, stratify=y)
    x_train_pca, x_test_pca, pca_evr = perform_PCA(x_train, x_test, n_components)
    accuracyLDA, pred_LDA_y = perform_LDA(x_train_pca, y_train, x_test_pca, y_test)
    accuracyNB, pred_NB_y = perform_naive_bayes(x_train_pca, y_train, x_test_pca, y_test)
    accuracySVM, pred_svm_y = perform_svm(x_train_pca, y_train, x_test_pca, y_test, kernel="rbf", gamma="scale")
    accuracyKNN, pred_knn_y = perform_knn(x_train_pca, y_train, x_test_pca, y_test, n_neighbors=5)
    plot_pca_variance(pca_evr)
    plot_class_distribution(y)
    #plot_correlation_matrix(data)
    print(f"Summary:\n"
          f"AccuracyLDA: {accuracyLDA}\n"
          f"AccuracyNB: {accuracyNB}\n"
          f"AccuracySVM: {accuracySVM}\n"
          f"AccuracyKNN: {accuracyKNN}\n")

if __name__ == "__main__":
    print("Init process for Iris dataset...")
    init_whole_process("Iris.csv", "Species")
    print("############################################################################################################")
    print("Init process for credit card fraud dataset...")
    init_whole_process("creditcard.csv", "Class")
    print("############################################################################################################")
