import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def detect_shift(X_P, X_Q, y_P, y_Q, print_logloss = True):
    
    # Pre step : Define Models
    model_f = RandomForestClassifier(max_depth=2, random_state=0) #LogisticRegression(solver = 'liblinear') # need check
    pi_estimator = LogisticRegression(solver = 'liblinear') # this model is a better fit for the TREE model.

    # Step 1: Estimation of E_P[R_P(X)] and E_Q[R_Q(X)]
    model_f.fit(X_P, y_P)

    # Compute losses for P and Q
    # We use Log Loss for evaluation!
    E_P_RP = log_loss(y_P, model_f.predict_proba(X_P))
    E_Q_RQ = log_loss(y_Q, model_f.predict_proba(X_Q))
    
    if print_logloss:
        print("===Estimiate log loss===")
        print("E_P_RP: ", E_P_RP)
        print("E_Q_RP: ", E_Q_RQ)
    else:
        pass

    # Step 2: Estimate alpha hat
    n_P = len(y_P)
    n_Q = len(y_Q)
    alpha_hat = n_Q / (n_P + n_Q)

    # Step 3: Estimate pi_hat(x)
    # Compare this paper Stable Learning via Sample Reweighting https://arxiv.org/abs/1911.12580
    X_combined = np.vstack((X_P, X_Q))
    y_combined = np.hstack((np.zeros(len(y_P)), np.ones(len(y_Q))))
    pi_estimator.fit(X_combined, y_combined)

    # Step 4: Calculate importance weights
    pi_P = pi_estimator.predict_proba(X_P)[:, 1]
    pi_Q = pi_estimator.predict_proba(X_Q)[:, 1]
    w_P = pi_P / ((1 - alpha_hat) * pi_P + alpha_hat * (1 - pi_P))
    w_Q = (1 - pi_Q) / ((1 - alpha_hat) * pi_Q + alpha_hat * (1 - pi_Q))
    
    # Step 5: Estimate E_S[R_P(X)] and E_S[R_Q(X)] using these importance weights:
    proba_P = model_f.predict_proba(X_P)
    proba_Q = model_f.predict_proba(X_Q)

    logloss_P = np.array([-np.log(proba_P[i][y_P[i]]) for i in range(len(y_P))])
    logloss_Q = np.array([-np.log(proba_Q[i][y_Q[i]]) for i in range(len(y_Q))])

    E_S_RP = np.sum(w_P * logloss_P) / np.sum(w_P)
    E_S_RQ = np.sum(w_Q * logloss_Q) / np.sum(w_Q)


    # Step 6: Differences between consecutive pairs of estimates
    P_to_S    = E_P_RP - E_S_RP # P to S
    Y_X_Shift = E_S_RP - E_S_RQ # Y|X shfit  
    S_to_Q    = E_S_RQ - E_Q_RQ # S to Q

    
    return P_to_S, Y_X_Shift, S_to_Q