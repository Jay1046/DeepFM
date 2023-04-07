import pandas as pd
from Data import load
from Preprocess.features import GetFeature
from Preprocess.preprocess import Preprocess
from Preprocess.postpreprocess import post_preprocess
import Settings
from deepctr.models.deepfm import DeepFM
from Model.model import DeepFMModel
# from Model.model import train_model
import warnings
warnings.filterwarnings(action='ignore')

if __name__=='__main__':
    path = Settings.path
    
    print('data load')
    df = load.load_data(path)
    
    print(df.head())
    
    print('===========================================================================================')
    print("Preprocess starts")
    input_data, train_model_input, test_model_input, y_train = Preprocess.get_input_data(df)
    feature_columns, linear_feature_columns, dnn_feature_columns = GetFeature().feature_names(input_data)
    
    print("feature_columns : ", feature_columns)
    
    
    print("Train starts")
    model = DeepFMModel(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns)
    history = model.train(train_model_input, y_train, 100)
    y_pred = model.predict(test_model_input)
    
    print('===========================================================================================')
    result = post_preprocess(df, y_pred)
    print(result)
    
    
    
    
    
    