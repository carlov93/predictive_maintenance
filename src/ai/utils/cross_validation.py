import pandas as pd

class CrossValidationProvider():
    def __init__(self, path, no_folds, amount_data, stake_training_data, ignored_features):
        self.path = path
        self.train_data = None
        self.test_data = None
        self.no_folds = no_folds
        self.amount_data = amount_data
        self.stake_training_data = stake_training_data
        self.ignored_features = ignored_features
        
    def split_training_test(self):
        # Read hole dataset und drop features
        dataset = pd.read_csv(self.path)
        dataset = dataset.drop(labels=self.ignored_features, axis=1)
        
        # Select training and test dataset
        sub_dataset = dataset.iloc[:self.amount_data,:]
        amount_training_data = round(len(sub_dataset)*self.stake_training_data)
        self.train_data = sub_dataset.iloc[:amount_training_data,:]
        self.test_data = sub_dataset.iloc[amount_training_data:,:]
        
    def split_dataset_in_folds(self):
        amount_each_fold = round(len(self.train_data) / self.no_folds)
        
        folds = []
        for fold_number in range(1,self.no_folds+1):
            fold = self.train_data.iloc[(fold_number-1)*amount_each_fold:fold_number*amount_each_fold,:]
            folds.append(fold)
        return folds
        
    def provide_data(self):
        self.split_training_test()
        folds = self.split_dataset_in_folds()
        return self.test_data, folds 