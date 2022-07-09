import os
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.utils import load_yaml, load_pkl, make_directory, save_pkl, save_yaml

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from scipy.stats import entropy

from imblearn.over_sampling import SMOTE

PROJECT_DIR = os.path.dirname(__file__)

PREPROCESS_CONFIG_PATH = os.path.join(
    PROJECT_DIR, 'config/preprocess_config.yaml')
config = load_yaml(PREPROCESS_CONFIG_PATH)

SRC_DIR = config['DIRECTORY']['srcdir']
DST_DIR = config['DIRECTORY']['dstdir']

# Read in train data
DATA_PATH = os.path.join(SRC_DIR, 'train.csv')
TEST_PATH = os.path.join(SRC_DIR, 'test.csv')
data = pd.read_csv(DATA_PATH)
test = pd.read_csv(TEST_PATH)


# Split dataset
cols = list(data.columns)
cols = [x for x in cols if 'HZ' in x]

X = data[cols]
y = data['leaktype']

# preprcessing functions


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
# n_components=32: f1=0.45


def oversample_data(X, y, sampling_strategy, k_neighbors):
    smote = SMOTE(random_state=0,
                  sampling_strategy=sampling_strategy,
                  k_neighbors=k_neighbors)
    X_oversampled, y_oversampled = smote.fit_resample(X, y)

    return X_oversampled, y_oversampled
# sampling_strategy='auto', k_neighbors=5


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


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

X_train, y_train = oversample_data(X_train, y_train, 'auto', 5)
X_train, X_valid, test = get_pca_sliced2(
    X_train, X_valid, test, cols, idx_sliced=100, n_components=40)


X_valid.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)

traindf = pd.concat([X_train, y_train], axis=1)
validdf = pd.concat([X_valid, y_valid], axis=1)
testdf = test

# Save split dataset
trainpath = os.path.join(DST_DIR, 'train.csv')
validpath = os.path.join(DST_DIR, 'valid.csv')
testpath = os.path.join(DST_DIR, 'test.csv')

traindf.to_csv(trainpath, index=False)
validdf.to_csv(validpath, index=False)
testdf.to_csv(testpath, index=False)

print('preprocess done')
