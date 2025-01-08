from credit_risk import CreditRiskClassifier
from credit_database import CreditDatabase
from census import CensusDataBase

credit_risk = CreditRiskClassifier()
credit_risk.credit_risk()

credit = CreditDatabase()
credit.decision_tree_credit()

census = CensusDataBase()
census.decision_tree_census()
