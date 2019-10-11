import pandas as pd

class CrossValidationProvider():
    def __init__(self, path, no_folds, amount_data, ignored_features):
        self.path = path
        self.data = None
        self.no_folds = no_folds
        self.amount_data = amount_data
        self.ignored_features = ignored_features
        
    def load_data(self):
        # Read hole dataset und drop features
        dataset = pd.read_csv(self.path)
        data = dataset.drop(labels=self.ignored_features, axis=1)
        self.data = data.iloc[:self.amount_data,:]
        
    def split_dataset_in_folds(self):
        amount_each_fold = round(len(self.data) / self.no_folds)
        
        folds = []
        for fold_number in range(1,self.no_folds+1):
            fold = self.data.iloc[(fold_number-1)*amount_each_fold:fold_number*amount_each_fold,:]
            folds.append(fold)
        return folds
        
    def provide_data(self):
        self.load_data()
        folds = self.split_dataset_in_folds()
        return folds
    