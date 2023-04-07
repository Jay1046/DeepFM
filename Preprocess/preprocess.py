from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from Settings import settings
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from Preprocess.features import GetFeature



class Preprocess:
    
    inputs = settings['input_columns']
    target = settings['target_column']
    sparse_features = settings['sparse_features']
    dense_features = settings['dense_features']
    
    
    def _preprocess(df):
        input_df = df[Preprocess.inputs]
        
        encoder = LabelEncoder()
        scaler = MinMaxScaler(feature_range=(0,1))
        
        # Labeling for sparse features
        for col in Preprocess.sparse_features:
            input_df[col] = encoder.fit_transform(input_df[col])
            
        # Scaling for dense features
        input_df[Preprocess.dense_features] = scaler.fit_transform(input_df[Preprocess.dense_features])
        
        
        target_idx = df[df[Preprocess.target[0]].isnull() == True].index
        
        
        train = input_df.drop(target_idx)
        y_train = df.drop(target_idx)[Preprocess.target]
        
        test = input_df.loc[target_idx]
        
        return input_df, train, y_train, test
        
    
    def get_input_data(df):
        
        input_df, train, y_train, test = Preprocess._preprocess(df)
        
        feature_names, linear_feature_columns, dnn_feature_columns = GetFeature().feature_names(input_df)
        
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}
        
        return input_df, train_model_input, test_model_input, y_train
        

    
        