import pandas as pd
from Data import load
from Preprocess.features import GetFeature
from Preprocess.preprocess import Preprocess
import Settings
from deepctr.models import DeepFM
from sklearn.metrics import r2_score





class DeepFMModel:
    def __init__(self, linear_feature_columns, dnn_feature_columns, task='regression', optimizer='adam', loss='mean_squared_error', metrics_list=['mse']):
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.task = task
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics_list
        self.model = None
        
    def _build(self):
        model = DeepFM(self.linear_feature_columns, self.dnn_feature_columns, task=self.task)
        model.compile(optimizer=self.optimizer,
                      loss = self.loss,
                      metrics=self.metrics)
        return model
    
    def train(self, input_data, target_data, epochs):
        self.model = self._build()
        hist = self.model.fit(input_data, target_data, epochs=epochs)
        return hist
    
    def predict(self, data):
        return self.model.predict(data)
    
    def evaluate(self, train_input, train_target):
        predict = self.predict(train_input)
        print("R2 score: ", round(r2_score(predict, train_target), 4))
        
    def _save(self, path):
        self.model.save(path)
        
    
    