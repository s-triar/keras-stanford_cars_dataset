import cv2
import os
import argparse
from enum import Enum
import logging

class CLI(Enum):
    DATA = 'path_data'
    CHECKPOINT = 'path_model_checkpoint'
    CHECKPOINT_FREQUENCY = 'checkpoint_frequency'
    EPOCHS = 'epochs'
    VALIDATION_FREQUENCY = 'validation_frequency'
    NUM_CLASSES = 'number_of_classes'
    DATA_SUBSET = 'data_subset'
    FREEZE_WEIGHTS = 'freeze_weights'
    IMAGES = 'path_images'
    LABELS = 'path_labels'
    MODEL = 'model'


class Hyperparameters(Enum):
    LEARNING_RATE_SCHEDULER = 'learning_rate_scheduler'
    BATCH_SIZE = 'batch_size'
    NESTEROV = 'nesterov'
    WEIGHT_DECAY = 'weight_decay'
    MOMENTUM = 'momentum'
    LEARNING_RATE = 'learning_rate'
    SCEDULER_RATE = 'scheduler_rate'

class CropImage():

    def __init__(self, *args, **kwargs):

        return super().__init__(*args, **kwargs)

    def arg_parse(self):
        """CLI interface"""
        parser = argparse.ArgumentParser(description='CLI for tuning ResNet for Stanford Cars dataset.')
        parser.add_argument("--" + CLI.DATA.value, dest=CLI.DATA.value, type=str,
                            help="mat file with training and test data", required=True)
        parser.add_argument("--" + CLI.IMAGES.value, dest=CLI.IMAGES.value, type=str,
                            help="path to directory with images", required=True)
        parser.add_argument("--" + CLI.LABELS.value, dest=CLI.LABELS.value, type=str,
                            help="file with id and human readable label name", required=True)

        parser.add_argument("--" + CLI.CHECKPOINT.value, dest=CLI.CHECKPOINT.value, type=str,
                            help="directory to save model checkpoints", required=False, default=None)
        parser.add_argument("--" + CLI.MODEL.value, dest=CLI.MODEL.value, type=str,
                            help="model to use. options: ResNet18, ResNet50", required=True)
        parser.add_argument("--" + CLI.CHECKPOINT_FREQUENCY.value, dest=CLI.CHECKPOINT_FREQUENCY.value, type=int,
                            help="frequency to save model", required=False, default=None)
        parser.add_argument("--" + CLI.NUM_CLASSES.value, dest=CLI.NUM_CLASSES.value, type=int,
                            help='number of unique classes in labels', required=True)

        parser.add_argument("--" + CLI.EPOCHS.value, dest=CLI.EPOCHS.value, type=int,
                            help="total number of training epochs", required=True)
        parser.add_argument("--" + CLI.VALIDATION_FREQUENCY.value, dest=CLI.VALIDATION_FREQUENCY.value, type=int,
                            help="frequency to run validation", required=True)

        parser.add_argument("--" + CLI.DATA_SUBSET.value, dest=CLI.DATA_SUBSET.value, type=float,
                            help="subset of training data to use", required=True)

        parser.add_argument("--" + CLI.FREEZE_WEIGHTS.value, dest=CLI.FREEZE_WEIGHTS.value, action='store_true',
                            help="whether or not to freeze weights on pretrained model")

        parser.add_argument("--" + "no-" + CLI.FREEZE_WEIGHTS.value, dest=CLI.FREEZE_WEIGHTS.value,
                            action='store_false',
                            help="whether or not to freeze weights on pretrained model")

        return parser

dt = CropImage()
parsed_cli = dt.arg_parse()
e = parsed_cli.parse_args()
parsed_cli_dict = e.__dict__
print(parsed_cli)
print(e)
print(parsed_cli_dict)
logging.debug("command line arguments: %s", parsed_cli_dict)
