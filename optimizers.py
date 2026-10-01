import numpy as np
import pandas as pd
import activations as act
import loss

class optimizers:

    class SGD:
        def __init__(self, lr=0.01):
            self.lr = lr

        def update(self, params, grads):
            for param, grad in zip(params, grads):
                param -= self.lr*grad


    class NAG:
        def __init__(self, lr=0.01, momentum=0.9):
                    self.lr = lr
                    self.momentum = momentum
                    self.velocity = None
        
        def update(self, params, grads):
            if self.velocity is None:
                 self.velocity = [np.zeros_like(p) for p in params]

            pass

    class Adagrad:
        pass  

    class RMSProp:
        pass
    
    class Adam:
        pass

    