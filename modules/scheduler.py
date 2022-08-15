import torch


class EarlyStopping:
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.percentage = percentage
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.best_epoch = None
        self.num_bad_epochs = 0
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics, epoch):
        if self.patience == 0:
            return False, self.best, self.best_epoch, self.num_bad_epochs

        if self.best is None:
            self.best = metrics
            self.best_epoch = epoch
            return False, self.best, self.best_epoch, 0

        if torch.isnan(metrics):
            return True, self.best, self.best_epoch, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.best_epoch, self.num_bad_epochs

        return False, self.best, self.best_epoch, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)

    def state_dict(self):
        state_dict = {
            'best': self.best,
            'best_epoch': self.best_epoch,
            'num_bad_epochs': self.num_bad_epochs,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_epoch = state_dict['best_epoch']
        self.num_bad_epochs = state_dict['num_bad_epochs']

    def reset(self):
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.best_epoch = None
        self._init_is_better(self.mode, self.min_delta, self.percentage)

    def __str__(self):
        return f'EarlyStopping({self.mode},{self.min_delta},{self.patience},{self.percentage})'
