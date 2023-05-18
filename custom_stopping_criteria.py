from transformers.generation.stopping_criteria import *


class UserStoppCriteria(StoppingCriteria):

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores):
        return self.event.is_set()
    




    
