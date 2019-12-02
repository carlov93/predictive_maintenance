import pandas as pd

class CrossValidationProvider():
    def __init__(self, path, no_folds, amount_data, ignored_features, stake):
        self.path = path
        self.no_folds = no_folds
        self.amount_data = amount_data
        self.ignored_features = ignored_features
        self.stake_training_data = stake
        
    def load_data(self):
        # Read hole dataset und drop features
        dataset = pd.read_csv(self.path)
        data = dataset.drop(labels=self.ignored_features, axis=1)
        return data.iloc[:self.amount_data,:]
        
    def split_trainingset_into_folds(self, training_set):
        amount_each_fold = round(len(training_set) / self.no_folds)
        
        folds = []
        for fold_number in range(1,self.no_folds+1):
            fold = training_set.iloc[(fold_number-1)*amount_each_fold:fold_number*amount_each_fold,:]
            folds.append(fold)
        return folds
    
    def split_data_into_tain_and_test(self, all_data):
        amount_train_data = round(all_data.shape[0] * self.stake_training_data)
        train_set = all_data.iloc[:amount_train_data,:]
        test_set = all_data.iloc[amount_train_data:,:]
        return train_set, test_set
        
    def provide_data(self):
        all_data = self.load_data()
        training_set, test_set = self.split_data_into_tain_and_test(all_data)
        folds_training = self.split_trainingset_into_folds(training_set)
        return folds_training, test_set
    