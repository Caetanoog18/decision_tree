import os
import pickle
from sklearn import tree
from matplotlib import pyplot as plt
from pre_processing import CreditRisk
from sklearn.tree import DecisionTreeClassifier


class CreditRiskClassifier:

    def __init__(self):
        self.pre_processing = CreditRisk().pre_processing()
        self.X_credit_risk = None
        self.y_credit_risk = None

    def credit_risk(self):
        file_path = 'database/credit_risk.pkl'
        if os.path.exists(file_path):
            with open('database/credit_risk.pkl', 'rb') as file:
                self.X_credit_risk,self.y_credit_risk = pickle.load(file)
            print('Data was successfully loaded. \n')
        else:
            raise FileNotFoundError(f'The file {file_path} was not found.')

        # print(self.X_credit_risk)

        decision_tree = DecisionTreeClassifier(criterion='entropy')
        decision_tree.fit(self.X_credit_risk, self.y_credit_risk)

        # print(decision_tree.classes_)
        # print(decision_tree.feature_importances_)

        predictiors = ['History', 'Debt', 'Guarantee', 'Income']
        plt.figure(figsize = (12,8))
        tree.plot_tree(decision_tree, feature_names=predictiors, class_names=decision_tree.classes_, filled=True)
        plt.title('Decision Tree for credit risk database')

        image_path = 'images/credit_risk.png'
        plt.savefig(image_path)
        plt.show()

        # Good history, high debt, no guarantee, income > 35
        # Bad history, high debt, no guarantee, income < 15

        prediction = decision_tree.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
        print(f'Prediction result: {prediction}')


