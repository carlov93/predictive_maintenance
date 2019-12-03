class Logger():
    def __init__(self, base_path, ID):
        self.path = base_path + "id" + ID + ".txt"
        
    def log_current_statistics(self, epoch, mean_epoch_training_loss):
        with open(self.path, 'a+') as file:
            file.write("-------- epoch_no. {} finished with training loss {}--------".format(epoch, mean_epoch_training_loss))
            file.write("\n")
            
    def log_message(self, message):
        with open(self.path, 'a+') as file:
            file.write(message)
            file.write("\n")
