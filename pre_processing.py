import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



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


class Credit:
    def __init__(self):
        self.credit = pd.read_csv('database/credit.csv')
        self.X_credit = None
        self.y_credit = None
        self.X_credit_training = None
        self.X_credit_test = None
        self.y_credit_training = None
        self.y_credit_test = None

        # Verifying inconsistent values
        print(self.credit.loc[self.credit['age']<0])

        # Mean without inconsistent values
        mean = self.credit['age'][self.credit['age']>0].mean()

        self.credit.loc[self.credit['age']<0, 'age'] = mean

        #Missing values
        print(self.credit.loc[pd.isnull(self.credit['age'])])
        self.credit.fillna(self.credit['age'].mean(), inplace=True)

        self.X_credit = self.credit.iloc[:, 1:4].values
        self.y_credit = self.credit.iloc[:, 4].values

        # Printing the maximum and the minimum values
        print(self.X_credit[:, 0].min(), self.X_credit[:, 1].min(), self.X_credit[:, 2].min())
        print(self.X_credit[:, 0].max(), self.X_credit[:, 1].max(), self.X_credit[:, 2].max())

        credit_scaler = StandardScaler()
        self.X_credit = credit_scaler.fit_transform(self.X_credit)

        # Printing the maximum and the minimum values
        print(self.X_credit[:, 0].min(), self.X_credit[:, 1].min(), self.X_credit[:, 2].min())
        print(self.X_credit[:, 0].max(), self.X_credit[:, 1].max(), self.X_credit[:, 2].max())

        self.X_credit_training, self.y_credit_training, self.X_credit_test, self.y_credit_test = train_test_split(self.X_credit, self.y_credit, test_size=0.25, random_state=42)

        with open('database/credit.pkl', 'wb') as file:
            pickle.dump([self.X_credit_training, self.y_credit_training], file)