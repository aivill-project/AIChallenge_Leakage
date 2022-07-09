import os
import sys
import joblib
import pickle
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

from models.ML_model import get_ml_model
from modules.utils import load_yaml, load_pkl, make_directory, save_pkl, save_yaml

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA

from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold

from lightgbm.callback import record_evaluation


######################################################################
### VARIABLES ###

PROJECT_DIR = os.path.dirname(__file__)

PREPROCESS_CONFIG_PATH = os.path.join(
    PROJECT_DIR, 'config/preprocess_config.yaml')
preprocess_config = load_yaml(PREPROCESS_CONFIG_PATH)

SRC_DIR = preprocess_config['DIRECTORY']['srcdir']
DST_DIR = preprocess_config['DIRECTORY']['dstdir']

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)

# DATA
DATA_DIR = config['DIRECTORY']['data']

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# MODEL
MODEL_STR = config['MODEL']
PARAMETER = config['PARAMETER'][MODEL_STR]
# LABEL_ENCODE
LABEL_ENCODING = config['LABEL_ENCODING']

# TRAIN
EARLY_STOPPING_ROUND = config['TRAIN']['early_stopping_round']

# time offset set
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S")
TRAIN_SERIAL = MODEL_STR + '_' + TRAIN_TIMESTAMP

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(
    PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)


######################################################################
### FUNCTIONS ###

def scale_data(X, test, cols):
    """스케일 전처리"""
    scaler = StandardScaler()
    scaler.fit(X[cols])
    X.loc[:, cols] = scaler.transform(X[cols])
    test.loc[:, cols] = scaler.transform(test[cols])

    return X, test


def add_stats(X, test, cols):
    """통계 파생변수 추가"""
    for df in [X, test]:
        df['mean'] = df[cols].mean(axis=1, numeric_only=True)
        df['std'] = df[cols].std(axis=1, numeric_only=True)
        df['skew'] = df[cols].skew(axis=1, numeric_only=True)
        df['kurt'] = df[cols].kurt(axis=1, numeric_only=True)
        df['entropy'] = entropy(df.iloc[:, 1:].T)

    return X, test


def get_pca(X, test, cols, n_components):
    """n_components에 따라 PCA"""
    pca = PCA(n_components=n_components)
    pca.fit(X[cols])
    X = pd.DataFrame(pca.transform(X[cols]))
    id_col = test['id']
    test = pd.DataFrame(pca.transform(test[cols]))
    test['id'] = id_col

    return X, test


def oversample_data(X, y, sampling_strategy, k_neighbors):

    oversamplers = [
        SMOTE(random_state=RANDOM_SEED, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors),
        # ADASYN(sampling_strategy='auto', random_state=RANDOM_SEED, n_neighbors=k_neighbors),
        # BorderlineSMOTE(sampling_strategy='auto', random_state=RANDOM_SEED, k_neighbors=k_neighbors),
        # KMeansSMOTE(sampling_strategy='auto', random_state=RANDOM_SEED, k_neighbors=k_neighbors),
    ]

    result = []
    for oversampler in oversamplers:
        X_res, y_res = oversampler.fit_resample(X, y)
        res = pd.concat([X_res, y_res], axis=1)
        result.append(res)
    total = pd.concat(result, axis=0)
    total.drop_duplicates(subset=total.columns, keep='first', inplace=True)

    total.reset_index(drop=True, inplace=True)

    X_oversampled = total[cols]
    y_oversampled = total['leaktype']

    return X_oversampled, y_oversampled


def get_pca_sliced2(X_train, X_valid, test, cols, idx_sliced, n_components):
    leave_cols = cols[:idx_sliced]
    pca_cols = cols[idx_sliced:]
    id_col = test['id']

    pca = PCA(n_components=n_components)
    pca.fit(X_train[pca_cols])

    X_train_pca = pd.DataFrame(pca.transform(X_train[pca_cols]))
    X_valid_pca = pd.DataFrame(pca.transform(X_valid[pca_cols]))
    test_pca = pd.DataFrame(pca.transform(test[pca_cols]))

    X_train = pd.concat([X_train[leave_cols], X_train_pca], axis=1)

    temp = X_valid[leave_cols]
    temp.reset_index(drop=True, inplace=True)

    X_valid = pd.concat([temp, X_valid_pca], axis=1)
    test = pd.concat([test[leave_cols], test_pca], axis=1)

    test['id'] = id_col

    print(f'--- Feature extracted: X_valid {X_valid.shape}, test {test.shape}')

    return X_train, X_valid, test


def df_cutter(df, cut):
    cut = 10*cut
    df_cut = df.iloc[:, cut:cut+10]
    return df_cut


"""
feature_cols = ['mean', 'std', 'skew', 'kurt', 'entropy']

X_cut_list = []
test_cut_list = []
test_ids = test['id']

for i in range(math.floor(len(cols)/10)):
    X_cut_list.append(df_cutter(X,i))
    test_cut_list.append(df_cutter(test[cols],i))

def segment_calc(cutlist):
    ret = []
    for i in range(len(cutlist)):
        if i<10: 
            ret.append(cutlist[i])
        else: 
            calc = cutlist[i].mean(axis='columns', numeric_only=True)
            #calc = cutlist[i+1].sum(axis='columns', numeric_only=True)
            ret.append(calc)
    return ret
   
X_cut_list = segment_calc(X_cut_list)
test_cut_list = segment_calc(test_cut_list)

X = pd.concat(X_cut_list,axis=1)
test = pd.concat(test_cut_list,axis=1)
test = pd.concat([test_ids, test],axis=1)

X.rename(columns = lambda x: str(x)+"HZ", inplace = True)
X.rename(columns={'leaktypeHZ':'leaktype'}, inplace=True)

test.rename(columns = lambda x: str(x)+"HZ", inplace = True)
test.rename(columns={'leaktypeHZ':'leaktype', 'idHZ':'id'}, inplace=True)
"""


def evaluate_macroF1_lgb(truth, predictions):
    pred_labels = predictions.reshape(len(np.unique(truth)), -1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True)


######################################################################
### PREPROCESS ###

# Prepare data
DATA_PATH = os.path.join(SRC_DIR, 'train.csv')
TEST_PATH = os.path.join(SRC_DIR, 'test.csv')

data = pd.read_csv(DATA_PATH)
test = pd.read_csv(TEST_PATH)

cols = [x for x in data.columns if 'HZ' in x]
X, y = data[cols], data['leaktype']

print(f'--- Prepared data: X {X.shape}, y {y.shape}, test {test.shape}')


# Split data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=RANDOM_SEED, stratify=y)

# Preprocess data

X_train, y_train = oversample_data(X_train, y_train, 'auto', 5)
X_train, X_valid, test = get_pca_sliced2(
    X_train, X_valid, test, cols, idx_sliced=100, n_components=40)
X_valid.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)

print('### Preprocess done ###')

# Save preprocessed data
traindf = pd.concat([X_train, y_train], axis=1)
validdf = pd.concat([X_valid, y_valid], axis=1)
testdf = test

trainpath = os.path.join(DST_DIR, 'train.csv')
validpath = os.path.join(DST_DIR, 'valid.csv')
testpath = os.path.join(DST_DIR, 'test.csv')

traindf.to_csv(trainpath, index=False)
validdf.to_csv(validpath, index=False)
testdf.to_csv(testpath, index=False)

print('### Saved preprocessed data ###')


if __name__ == '__main__':

    # Encode target
    y_train = y_train.replace(LABEL_ENCODING)
    y_valid = y_valid.replace(LABEL_ENCODING)

    # MODEL
    model = get_ml_model(MODEL_STR, PARAMETER)

    history = dict()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric=evaluate_macroF1_lgb,
        early_stopping_rounds=EARLY_STOPPING_ROUND,
        verbose=1,
        callbacks=[record_evaluation(history)])

    # GET F1 SCORE and CONFUSION MATRIX
    y_valid_np = np.array(y_valid)
    y_valid_pred = model.predict(X_valid)

    f1 = history['valid_1']['macroF1'][-1]
    cf = confusion_matrix(y_valid_np, y_valid_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=[
                                  'other', 'noise', 'normal', 'in', 'out'])

    # SAVE MODEL
    PERFORMANCE_RECORD_DIR = f'{PERFORMANCE_RECORD_DIR}_f1_{f1}'
    make_directory(PERFORMANCE_RECORD_DIR)
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)

    joblib.dump(model, os.path.join(PERFORMANCE_RECORD_DIR, 'model.pkl'))
    save_pkl(os.path.join(PERFORMANCE_RECORD_DIR, 'loss_history.pkl'), history)

    print('')
    print(f'### VALID F1 SCORE: {f1}')
    
    disp.plot()
    plt.savefig(os.path.join(PERFORMANCE_RECORD_DIR, f'cf_f1_{f1}.jpg'))
    plt.close()

    print(f'### Model saved')
