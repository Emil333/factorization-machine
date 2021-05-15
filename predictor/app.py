import pandas as pd
import xlearn as xl
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from predictor.converter import *

warnings.filterwarnings('ignore')


def preprocess_dataset(train_dataframe):
    train_dataframe['Credit_History'].fillna(0, inplace=True)
    dict_ls = {'Y': 1, 'N': 0}
    self_employed_dict = {'Yes': 1, 'No': 1}
    train_dataframe['Self_Employed'].replace(self_employed_dict, inplace=True)
    train_dataframe['Self_Employed'].fillna(0, inplace=True)

    train_dataframe['Loan_Status'].replace(dict_ls, inplace=True)

    categorical_feature_mask = train_dataframe.dtypes == object
    categorical_columns = train_dataframe.columns[categorical_feature_mask].to_list()
    print(train_dataframe.head(10))
    label_encoder = LabelEncoder()
    train_dataframe[categorical_columns] = train_dataframe[categorical_columns].apply(
        lambda col: label_encoder.fit_transform(col))
    print(train_dataframe.head(10))
    return train_dataframe


def train_dataset():
    train = pd.read_csv('./datasets/train_ctrUa4K.csv')
    cols = ['Education', 'ApplicantIncome', 'Credit_History', 'CoapplicantIncome', 'Self_Employed', 'Loan_Status']
    train_sub = train[cols]
    train_sub = preprocess_dataset(train_sub)

    X_train, X_test = train_test_split(train_sub, test_size=0.3, random_state=5)
    convert_to_ffm(X_train, 'train', X_train, None,
                   X_train.filter(['Education', 'ApplicantIncome', 'Credit_History']))
    convert_to_ffm(X_test, 'test', X_test, None,
                   X_test.filter(['Education', 'ApplicantIncome', 'Credit_History']))
    print("completed")

    ffm_model = xl.create_ffm()
    ffm_model.setTrain("./generated/train_ffm.txt")
    param = {'task': 'binary',
             'lr': 0.2,
             'lambda': 0.002,
             'metric': 'acc'}

    # Start to train
    # The trained model will be stored in model.out
    ffm_model.fit(param, './generated/model.out')
    ffm_model.cv(param)

    # Prediction task
    ffm_model.setTest("./generated/test_ffm.txt")  # Test data
    ffm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    ffm_model.predict("./generated/model.out", "./generated/output.txt")


if __name__ == '__main__':
    train_dataset()
