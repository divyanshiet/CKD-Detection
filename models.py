import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class Model:

    def __init__(self):
        self.name = ''
        path = 'dataset/kidney_disease.csv'

        data = pd.read_csv(path)
        data = data[['age', 'bp', 'su', 'pc', 'pcc', 'sod', 'hemo', 'htn', 'dm', 'classification']]
        data.dropna(inplace=True)

        data['dm'] = data['dm'].str.replace(" ", "")
        data['dm'] = data['dm'].str.replace("\t", "")
        data['classification'] = data['classification'].str.replace("\t", "")

        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)
        labelencoder = LabelEncoder()
        data['pc'] = labelencoder.fit_transform(data['pc'])
        data['pcc'] = labelencoder.fit_transform(data['pcc'])
        data['htn'] = labelencoder.fit_transform(data['htn'])
        data['dm'] = labelencoder.fit_transform(data['dm'])
        data['classification'] = labelencoder.fit_transform(data['classification'])

        self.df = data
        self.split_data(self.df)

    def split_data(self, df):
        X = df.drop('classification', axis=1)
        y = df['classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def svm_classifier(self):
        self.name = 'Svm Classifier'
        classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        return classifier.fit(self.x_train, self.y_train)

    def decisionTree_classifier(self):
        self.name = 'Decision tree Classifier'
        classifier = DecisionTreeClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def randomforest_classifier(self):
        self.name = 'Random Forest Classifier'
        classifier = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
        return classifier.fit(self.x_train, self.y_train)

    def naiveBayes_classifier(self):
        self.name = 'Naive Bayes Classifier'
        classifier = GaussianNB()
        return classifier.fit(self.x_train, self.y_train)

    def knn_classifier(self):
        self.name = 'Knn Classifier'
        classifier = KNeighborsClassifier()
        return classifier.fit(self.x_train, self.y_train)

    def accuracy(self, model):
        predictions = model.predict(self.x_test)
        y_true = self.y_test
        cm = confusion_matrix(y_true, predictions)
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        acc = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)

        print(f"\n{self.name}")
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_comparison(metrics_dict):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'RMSE']
    classifiers = list(metrics_dict.keys())
    values = {metric: [metrics_dict[clf][metric] for clf in classifiers] for metric in metrics}

    plt.figure(figsize=(10, 6))
    x = range(len(classifiers))
    width = 0.15

    for i, metric in enumerate(metrics):
        plt.bar([p + i*width for p in x], values[metric], width=width, label=metric)

    plt.xticks([p + 2*width for p in x], classifiers, rotation=15)
    plt.ylabel('Score')
    plt.title('Classifier Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = Model()
    metrics_dict = {}

    for clf_func in [
        model.svm_classifier,
        model.decisionTree_classifier,
        model.randomforest_classifier,
        model.naiveBayes_classifier,
        model.knn_classifier
    ]:
        clf = clf_func()
        predictions = clf.predict(model.x_test)
        y_true = model.y_test
        cm = confusion_matrix(y_true, predictions)
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        acc = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)

        # Store metrics for comparison
        metrics_dict[model.name] = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'RMSE': rmse
        }

        # Plot confusion matrix for each classifier
        plot_confusion_matrix(cm, model.name)

        # Optionally print metrics
        print(f"\n{model.name}")
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")

    # Plot comparison bar chart
    plot_comparison(metrics_dict)