import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CreditRisk:

    def __init__(self):
        self.credit_risk = pd.read_csv('database/credit_risk.csv')
        self.X_credit_risk = None
        self.y_credit_risk = None

    def pre_processing(self):

        self.X_credit_risk = self.credit_risk.iloc[:, 0:4].values
        self.y_credit_risk = self.credit_risk.iloc[:, 4].values

        label_encoder_history = LabelEncoder()
        label_encoder_debt = LabelEncoder()
        label_encoder_guarantee = LabelEncoder()
        label_encoder_income = LabelEncoder()

        self.X_credit_risk[:, 0] = label_encoder_history.fit_transform(self.X_credit_risk[:, 0])
        self.X_credit_risk[:, 1] = label_encoder_debt.fit_transform(self.X_credit_risk[:, 1])
        self.X_credit_risk[:, 2] = label_encoder_guarantee.fit_transform(self.X_credit_risk[:, 2])
        self.X_credit_risk[:, 3] = label_encoder_income.fit_transform(self.X_credit_risk[:, 3])

        with open(r'database/credit_risk.pkl', 'wb') as file:
            pickle.dump([self.X_credit_risk, self.y_credit_risk], file)

