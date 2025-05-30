import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
<<<<<<< HEAD
# import tensorflow as tf
# CNN-specific imports
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, Flatten, Dropout
# from keras.optimizers import adam_v2
# from keras.utils import to_categorical
=======
import tensorflow as tf
# CNN-specific imports
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout
from keras.optimizers import adam_v2
from keras.utils import to_categorical
>>>>>>> 921e75ec0c27a209fd30a304e3b6d76666bdc767

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

<<<<<<< HEAD
=======
    def cnn_classifier(self):
        self.name = 'CNN Classifier'
        # Use the same data splitting approach as other classifiers
        x_train = self.x_train.values
        x_test = self.x_test.values
        y_train = self.y_train.values
        y_test = self.y_test.values
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
>>>>>>> 921e75ec0c27a209fd30a304e3b6d76666bdc767

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
<<<<<<< HEAD
    # model.accuracy(model.cnn_classifier(), is_cnn=True)
=======
    model.accuracy(model.cnn_classifier(), is_cnn=True)
>>>>>>> 921e75ec0c27a209fd30a304e3b6d76666bdc767
