import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# CNN-specific imports
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.utils import to_categorical

class Model:

    def __init__(self):
        self.name = ''
        path = 'dataset/kidney_disease.csv'

        data = pd.read_csv
        data.drop("id", axis=1, inplace=True)
        data.dropna(inplace=True)

        data.reset_index( inplace=True)
        data.drop('index', axis = 1, inplace=True)
        data['classification'] = data['classification'].map({'ckd': 1, 'notckd': 0})
        data['rbc'] = data['rbc'].map({'abnormal': 1, 'normal': 0})
        data['pc'] = data['pc'].map({'abnormal': 1, 'normal': 0})
        data['pcc'] = data['pcc'].map({'present': 1, 'notpresent': 0})
        data['ba'] = data['ba'].map({'present': 1, 'notpresent': 0})
        data['htn'] = data['htn'].map({'yes': 1, 'no': 0})
        data['dm'] = data['dm'].map({'yes': 1, 'no': 0})
        data['cad'] = data['cad'].map({'yes': 1, 'no': 0})
        data['pe'] = data['pe'].map({'yes': 1, 'no': 0})
        data['ane'] = data['ane'].map({'yes': 1, 'no': 0})
        data['appet'] = data['appet'].map({'good': 1, 'poor': 0})

        self.df = data
        self.split_data(self.df)

        # df = pd.read_csv(path)
        # df = df[['age', 'bp', 'su', 'pc', 'pcc', 'sod', 'hemo', 'htn', 'dm', 'classification']]

        # df['age'] = df['age'].fillna(df['age'].mean())
        # df['bp'] = df['bp'].fillna(df['bp'].mean())
        # df['su'] = df['su'].fillna(df['su'].mode()[0])
        # df['pc'] = df['pc'].fillna(df['pc'].mode()[0])
        # df['pcc'] = df['pcc'].fillna(df['pcc'].mode()[0])
        # df['sod'] = df['sod'].fillna(df['sod'].mode()[0])
        # df['hemo'] = df['hemo'].fillna(df['hemo'].mode()[0])
        # df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
        # df['dm'] = df['dm'].str.replace(" ", "")
        # df['dm'] = df['dm'].str.replace("\t", "")
        # df['dm'] = df['dm'].fillna(df['dm'].mode()[0])
        # df['classification'] = df['classification'].str.replace("\t", "")
        # df['classification'] = df['classification'].fillna(df['classification'].mode()[0])

        # labelencoder = LabelEncoder()
        # df['pc'] = labelencoder.fit_transform(df['pc'])
        # df['pcc'] = labelencoder.fit_transform(df['pcc'])
        # df['htn'] = labelencoder.fit_transform(df['htn'])
        # df['dm'] = labelencoder.fit_transform(df['dm'])
        # df['classification'] = labelencoder.fit_transform(df['classification'])
        # self.df = df  # store cleaned df
        # self.split_data(df)

    def split_data(self, df):
        X = df.drop('classification', axis=1)
        y = df['classification']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values
        # y = df.iloc[:, 9].values
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=24)

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

    def cnn_classifier(self):
        self.name = 'CNN Classifier'
        x = self.df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values
        y = self.df.iloc[:, 9].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=24)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        y_train_cat = to_categorical(y_train)

        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(y_train_cat.shape[1], activation='softmax'))

        model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train_cat, epochs=20, batch_size=16, verbose=0)

        self.cnn_x_test = x_test
        self.cnn_y_test = y_test
        return model

    def accuracy(self, model, is_cnn=False):
        if is_cnn:
            predictions = model.predict(self.cnn_x_test)
            predictions = predictions.argmax(axis=1)
            cm = confusion_matrix(self.cnn_y_test, predictions)
        else:
            predictions = model.predict(self.x_test)
            cm = confusion_matrix(self.y_test, predictions)

        accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
        print(f"{self.name} has accuracy of {accuracy * 100:.2f} %")


if __name__ == '__main__':
    model = Model()
    model.accuracy(model.svm_classifier())
    model.accuracy(model.decisionTree_classifier())
    model.accuracy(model.randomforest_classifier())
    model.accuracy(model.naiveBayes_classifier())
    model.accuracy(model.knn_classifier())
    model.accuracy(model.cnn_classifier(), is_cnn=True)
