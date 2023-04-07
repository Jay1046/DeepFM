from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from Settings import settings


class GetFeature:
    
    def __init__(self):
        self.target = settings['target_column']
        self.input = settings['input_columns']
        self.embed = settings['embedding_dim']
        
        
    def fixlen_feature_columns(self, df):
                             
        sparse_feature = [SparseFeat(feat, df[feat].max() + 1, embedding_dim=self.embed) for feat in settings['sparse_features']]
        dense_feature = [DenseFeat(feat, 1, ) for feat in settings['dense_features']]
        
        return sparse_feature + dense_feature
    
    
    def feature_names(self, df):
        dnn_feature_columns = self.fixlen_feature_columns(df)
        linear_feature_columns = self.fixlen_feature_columns(df)
        
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        
        return feature_names, linear_feature_columns, dnn_feature_columns
    
    
    