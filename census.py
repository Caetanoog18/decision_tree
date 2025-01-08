import os
import pickle
from sklearn import tree
from pre_processing import Census
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report


class CensusDataBase:

    def __init__(self):
        self.census = Census().pre_processing()
        self.X_census = None
        self.y_census = None
        self.X_census_training = None
        self.y_census_training = None
        self.X_census_test = None

    def decision_tree_census(self):
        file_path = 'database/census.pkl'

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.X_census_training, self.y_census_training, self.X_census_test, self.y_census_test = pickle.load(file)
            print('Data was successfully loaded. \n')
        else:
            raise FileNotFoundError(f'The file {file_path} was not found.')

        decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
        decision_tree.fit(self.X_census_training, self.y_census_training)

        predictions = decision_tree.predict(self.X_census_test)
        # print(predictions)
        # print(self.y_census_test)

        accuracy = accuracy_score(self.y_census_test, predictions)
        print(f'The decision tree accuracy is {accuracy}.')

        confusion_matrix = ConfusionMatrix(decision_tree)
        confusion_matrix.fit(self.X_census_training, self.y_census_training)
        confusion_matrix.score(self.X_census_test, self.y_census_test)
        confusion_matrix.show()

        print(classification_report(self.y_census_test, predictions))



