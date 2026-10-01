import numpy as np

class Loss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    @staticmethod
    def BCELoss(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def MSELoss(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def MAELoss(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def HuberLoss(y_true, y_pred, delta=1.0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if(np.abs(y_true - y_pred) <= delta):
            return 0.5 * np.mean((y_true - y_pred) ** 2)
        else:
            return delta * np.mean(np.abs(y_true - y_pred)) - 0.5 * delta ** 2


    @staticmethod
    def CCELoss(y_true, y_pred, num_classes):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        loss = 0.0
        for i in range(num_classes):
            loss += -np.mean(y_true[:, i] * np.log(y_pred[:, i]))
        return loss
    

    def compute_loss(self, y_true, y_pred):
        return self.loss_function(y_true, y_pred)
