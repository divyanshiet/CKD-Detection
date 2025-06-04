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
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


        data.reset_index( inplace=True)
        data.drop('index', axis = 1, inplace=True)
        labelencoder = LabelEncoder()
        data['pc'] = labelencoder.fit_transform(data['pc'])
        data['pcc'] = labelencoder.fit_transform(data['pcc'])
        data['htn'] = labelencoder.fit_transform(data['htn'])
        data['dm'] = labelencoder.fit_transform(data['dm'])
        data['classification'] = labelencoder.fit_transform(data['classification'])

        self.df = data
        self.split_data(self.df)

        # --- CNN Data Preprocessing (Separate Copy) ---
        self.cnn_df = self.preprocess_for_cnn(pd.read_csv(path))

    def preprocess_for_cnn(self, data):
        # 1. Select relevant columns
        data = data[['age', 'bp', 'su', 'pc', 'pcc', 'sod', 'hemo', 'htn', 'dm', 'classification']]

        # 2. Handle missing values
        # Numerical: mean imputation
        for col in ['age', 'bp', 'su', 'sod', 'hemo']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            mean_val = data[col].mean()
            data[col].fillna(mean_val, inplace=True)
        # Categorical: mode imputation
        for col in ['pc', 'pcc', 'htn', 'dm', 'classification']:
            data[col] = data[col].astype(str).str.strip().str.replace("\t", "").str.replace(" ", "")
            mode_val = data[col].mode()[0]
            data[col].replace('', np.nan, inplace=True)
            data[col].fillna(mode_val, inplace=True)

        # 3. Outlier Detection and Removal (IQR method for numerical columns)
        num_cols = ['age', 'bp', 'su', 'sod', 'hemo']
        for col in num_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower) & (data[col] <= upper)]

        # 4. Categorical Data Encoding (binary for yes/no, label encoding for others)
        labelencoder = LabelEncoder()
        for col in ['pc', 'pcc', 'htn', 'dm', 'classification']:
            data[col] = labelencoder.fit_transform(data[col])

        # 5. Data Normalization (Standardization: mean=0, std=1)
        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])

        data.reset_index(drop=True, inplace=True)
        return data

    def split_data(self, df):
        X = df.drop('classification', axis=1)
        y = df['classification']

        x = df.iloc[:, : 9].values
        y = df.iloc[:, 9].values
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def svm_classifier(self):
        self.name = 'Svm Classifier'
        classifier = SVC(kernel= 'rbf', C=1.0, gamma='scale', random_state = 42)
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

    def cnn_classifier(self, epochs=30, batch_size=16, lr=0.001):
        # Prepare data
        df = self.cnn_df.copy()
        X = df.drop('classification', axis=1).values.astype(np.float32)
        y = df['classification'].values.astype(np.int64)

        num_classes = len(np.unique(y))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        # Reshape for Conv1d: (samples, channels=1, features)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train)
        y_train_tensor = torch.tensor(y_train)
        X_test_tensor = torch.tensor(X_test)
        y_test_tensor = torch.tensor(y_test)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define CNN model for tabular data
        class TabularCNN(nn.Module):
            def __init__(self, num_features, num_classes):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
                self.bn1 = nn.BatchNorm1d(32)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
                self.bn2 = nn.BatchNorm1d(64)
                self.dropout = nn.Dropout(0.2)
                self.flatten = nn.Flatten()
                # Dynamically compute the flattened size
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, num_features)
                    out = self.conv1(dummy)
                    out = self.bn1(out)
                    out = torch.relu(out)
                    out = self.dropout(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    out = torch.relu(out)
                    out = self.dropout(out)
                    out = self.flatten(out)
                    flattened_size = out.shape[1]
                self.fc1 = nn.Linear(flattened_size, 32)
                self.fc2 = nn.Linear(32, num_classes)

            def forward(self, x):
                x = torch.relu(self.bn1(self.conv1(x)))
                x = self.dropout(x)
                x = torch.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TabularCNN(X_train.shape[2], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor.to(device))
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()
            y_true = y_test_tensor.numpy()

        # Metrics
        cm = confusion_matrix(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print("\nCNN Classifier (PyTorch)")
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # For comparison plot
        return {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'RMSE': rmse
        }, cm

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

    # CNN Classifier
    cnn_metrics, cnn_cm = model.cnn_classifier()
    metrics_dict['CNN Classifier'] = cnn_metrics
    plot_confusion_matrix(cnn_cm, "CNN Classifier")
