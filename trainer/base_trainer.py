class BaseTrainer:
    def __init__(self, trainer, dataset, loss, resume, device='cpu'):
        self.trainer = trainer
        self.dataset = dataset
        self.loss = loss
        self.resume = resume
        self.device = device

        self.mode = None
        self._training = False
        self._validate = False
        self._evaluate = False

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def _set_mode(self):
        raise NotImplementedError()

    def training(self):
        self.mode = 'train'
        self._training = True
        self._validate = False
        self._evaluate = False
        self._set_mode()

    def validate(self):
        self.mode = 'val'
        self._training = False
        self._validate = True
        self._evaluate = False
        self._set_mode()

    def evaluate(self):
        self.mode = 'eval'
        self._training = False
        self._validate = False
        self._evaluate = True
        self._set_mode()

    def check_mode_set(self):
        assert self.mode is not None

    def reset(self):
        self.mode = None
        self._training = False
        self._validate = False
        self._evaluate = False
