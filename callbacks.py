from keras import backend as K
from keras import callbacks
from metrics import rmse, mae, mse, gini, xent
import time

class BaseCallback(callbacks.Callback):
    def __init__(self, config, training_data, validation_data=None, metrics=None):
        self.config = config

        self.X = training_data["X"]

        if validation_data is None:
            self.X_val = None
        else:
            self.X_val = validation_data["X"]
        
        self.metrics = metrics

        self.metric_fns = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "xent": xent,
            "gini": gini
        }

    def on_train_begin(self, logs={}):
        self.__set_lr(self.config["lr"])
        self.epoch_times = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.__epoch_begin_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.__epoch_end_time = time.time()
        epoch_duration = self.__epoch_end_time - self.__epoch_begin_time
        self.epoch_times.append(epoch_duration)
        self.__decay_lr()

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def __decay_lr(self):
        self.__set_lr(self.lr * self.config["lr_decay"])
    
    def __set_lr(self, lr):
        self.lr = lr
        K.set_value(self.model.optimizer.lr, lr)

    def print_scores(self, epoch, train_scores, valid_scores, logs):
        epoch_str = "{:04d}/{:04d}".format(epoch+1, self.params["epochs"])
        lr_str = "lr:{:.6f}".format(self.lr)
        train_str = self.__scores_to_str(train_scores)
        train_str = "loss:{:6.4f} {}".format(logs["loss"], train_str)
        print_str = "EPOCH {} | {} | TRAIN {}".format(epoch_str, lr_str, train_str)

        valid_str = self.__scores_to_str(valid_scores)
        valid_str = "loss:{:6.4f} {}".format(logs["val_loss"], valid_str)
        if valid_str is not None:
            print_str += " | VALID {}".format(valid_str)

        duration_str = "{:4.1f}s".format(self.epoch_times[-1])
        print_str += " | {}".format(duration_str)
        
        print(print_str)

    def __scores_to_str(self, scores):
        if scores is None:
            return None
        else:
            return " ".join(["{:s}:{:6.4f}".format(m, scores[m]) for m in self.metrics])

    def score(self, y_true, y_pred):
        if y_true is None or y_pred is None:
            return None
        else:
            return dict([(m, self.metric_fns[m](y_true, y_pred)) for m in self.metrics])


class AutoEncoderCallback(BaseCallback):
    def __init__(self, config, training_data, validation_data=None, metrics=["mse", "rmse", "mae"]):
        super().__init__(config, training_data, validation_data, metrics)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        self.__score_epoch(epoch, logs)

    def __score_epoch(self, epoch, logs):
        X_pred = self.model.predict(self.X)
        train_scores = super().score(self.X, X_pred)

        if self.X_val is None:
            X_val_pred = None
        else:
            X_val_pred = self.model.predict(self.X_val)
        valid_scores = super().score(self.X_val, X_val_pred)

        super().print_scores(epoch, train_scores, valid_scores, logs)

class MultiLayerPerceptronCallback(BaseCallback):
    def __init__(self, config, training_data, validation_data=None, metrics=["xent", "gini"]):
        super().__init__(config, training_data, validation_data, metrics)

        self.y = training_data["y"]

        if validation_data is None:
            self.y_val = None
        else:
            self.y_val = validation_data["y"]

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        self.__score_epoch(epoch, logs)

    def __score_epoch(self, epoch, logs):
        y_pred = self.model.predict(self.X)
        train_scores = super().score(self.y, y_pred)

        if self.X_val is None:
            y_val_pred = None
        else:
            y_val_pred = self.model.predict(self.X_val)
        valid_scores = super().score(self.y_val, y_val_pred)

        super().print_scores(epoch, train_scores, valid_scores, logs)