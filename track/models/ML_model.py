from lightgbm import LGBMClassifier
import optuna
from catboost import CatBoostClassifier


'''
import optuna


def get_callback(trial, model_str):
    if model_str == 'lgbm':
            return optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
'''    

def get_callback(trial, model_str):
    if model_str == 'lgbm':
            return optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
        

def get_ml_model(model_str:str,paramenter:dict):
    if model_str == 'lgbm':
        model = LGBMClassifier(**paramenter)
        return model
    if model_str == 'cat':
        model = CatBoostClassifier(**paramenter)
        return model
