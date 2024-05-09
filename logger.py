# logger.py

import os
import wandb


class Logger:

    def __init__(self, experiment_name, project, entity):
        self.experiment_name = experiment_name
        self.project = project
        self.entity = entity      
        self.logger = None

    def login(self):
        self.api_key = os.getenv('WANDB_API_KEY')
        if self.api_key is None:
            print('API key not found')
        else:
            wandb.login(key=self.api_key)
        
    def start(self, settings):
        self.logger = wandb.init(project=self.project, entity=self.entity, name=self.experiment_name,
                                  config = settings, settings=wandb.Settings(start_method="thread"))    

    def log(self, data, step=None):
        if self.logger:  
            self.logger.log(data, step=step)
        else:
            print("wandb logger is not initialized.")
   
