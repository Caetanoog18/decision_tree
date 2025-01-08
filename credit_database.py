import os
import pickle
from sklearn import tree
from pre_processing import Credit
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report

class CreditDatabase:

    def __init__(self):
        self.credit = Credit().pre_processing()
        self.X_credit = None
        self.y_credit = None
        self.X_credit_training = None
        self.y_credit_training = None
        self.X_credit_test = None
        self.y_credit_test = None

    def decision_tree_credit(self):
        file_path = 'database/credit.pkl'

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.X_credit_training, self.y_credit_training, self.X_credit_test, self.y_credit_test = pickle.load(file)
            print('Data was successfully loaded. \n')
        else:
            raise FileNotFoundError(f'The file {file_path} was not found.')

        # print(self.X_credit_training.shape, self.y_credit_training.shape)
        # print(self.X_credit_test.shape, self.y_credit_test.shape)


        decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
        decision_tree.fit(self.X_credit_training, self.y_credit_training)

        prediction = decision_tree.predict(self.X_credit_test)
        # print(prediction)
        # print(self.y_credit_test)

        accuracy = accuracy_score(self.y_credit_test, prediction)
        print(f'The decision tree accuracy is {accuracy}.')

        confusion_matrix = ConfusionMatrix(decision_tree)
        confusion_matrix.fit(self.X_credit_training, self.y_credit_training)
        confusion_matrix.score(self.X_credit_test, self.y_credit_test)
        confusion_matrix.show()

        print(classification_report(self.y_credit_test, prediction))

        predictors = ['income', 'age', 'loan']
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        tree.plot_tree(decision_tree, feature_names=predictors, class_names=['0', '1'], filled=True)
        fig.savefig('images/credit_tree.png')
        fig.show()

