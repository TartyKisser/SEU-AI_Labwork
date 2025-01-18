import argparse
import numpy as np
import time
import sys

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(os.path.dirname(currentdir),'src')
sys.path.insert(0,parentdir)
from tasks.task_template import TaskTemplate


class DemoFSLTask(TaskTemplate):
    
    @staticmethod
    def get_parser(parser=argparse.ArgumentParser()):
        parser.add_argument('--support_idx', type=list, default=[], 
                            help="List of idxs for supports of a task")
        parser.add_argument('--support_lbls', type=list, default=[], 
                            help="List of labels for the supports of a task")
        parser.add_argument('--target_idx', type=list, default=[],
                            help="List of idxs for targets of a task")
        parser.add_argument('--target_lbls', type=list, default=[], 
                            help="List of labels for the targets of a task")
        return parser
    
    @staticmethod
    def get_output_dim(args, dataset):
        return dataset.get_num_classes()
    
    def __init__(self, dataset, args, class_seed, sample_seed):
        """
        Few Shot Learning Task sampler for creating a single episode for a few-shot learning task
        """
        super().__init__(dataset, args, class_seed, sample_seed)
        self.support_idx = args.support_idx
        self.target_idx = args.target_idx
        self.support_lbls = args.support_lbls
        self.target_lbls = args.target_lbls
        
    def set_targarts(self, target_idx, target_lbls):
        self.target_idx = target_idx
        self.target_lbls = target_lbls
        
    def set_supports(self, support_idx, support_lbls):
        self.support_idx = support_idx
        self.support_lbls = support_lbls
    
    def __len__(self):
        return 1
    
    def __iter__(self):
        supports_x = self.support_idx
        supports_y = self.support_lbls
        targets_x = self.target_idx
        targets_y = self.target_lbls
        
        rng = np.random.RandomState(self.sample_seed)
        support_seeds = rng.randint(0, 999999999, len(supports_y))
        target_seeds = rng.randint(0, 999999999, len(targets_y))
        supports_y = zip(supports_y, support_seeds)
        targets_y = zip(targets_y, target_seeds)
        
        support_set = (supports_x, supports_y)
        target_set = (targets_x, targets_y)
        
        yield (support_set, target_set)