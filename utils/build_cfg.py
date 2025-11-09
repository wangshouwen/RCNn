from yacs.config import CfgNode as CN

_C = CN()

# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"
_C.no_sim = False
_C.no_sim_weights = None
_C.cutimage = False
_C.count_list = [2, 3]
_C.RECORD_PATH = ""
# Path to a directory where the files were saved previously
_C.RESUME = ""
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.USE_CUDA = True

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
_C.INPUT.TRAIN_SIZE = (224, 224)
_C.INPUT.TEST_SIZE = (224, 224)
# Cutout
_C.INPUT.CUTOUT_FACTOR = 0.5
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# RandomAugment
_C.INPUT.RANDAUGMENT_N = 2
_C.INPUT.RANDAUGMENT_M = 10

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ""
_C.DATASET.NAME = ""
_C.DATASET.TRAIN_SPLIT = "train"
_C.DATASET.VAL_SPLIT = "val"
_C.DATASET.TEST_SPLIT = "test"
_C.DATASET.IMG_PATH = ""
_C.DATASET.ANNO_PATH = ""
_C.DATASET.TEXT_PATH = ""

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.TRAIN_X = CN()
_C.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
_C.DATALOADER.TRAIN_X.BATCH_SIZE = 32
_C.DATALOADER.TRAIN_X.SHUFFLE = True

# Setting for the test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = "SequentialSampler"
_C.DATALOADER.TEST.BATCH_SIZE = 32
_C.DATALOADER.TEST.SHUFFLE = False

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.INIT_WEIGHTS = ""
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ""
_C.MODEL.BACKBONE.PRETRAINED = True

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "adam"
_C.OPTIM.LR = 0.0003
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
# The following also apply to other
# adaptive optimizers like adamw
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "single_step"
# -1 or 0 means the stepsize is equal to max_epoch
_C.OPTIM.STEPSIZE = (-1, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 10
# Set WARMUP_EPOCH larger than 0 to activate warmup training
_C.OPTIM.WARMUP_EPOCH = -1
# Either linear or constant
_C.OPTIM.WARMUP_TYPE = "linear"
# Constant learning rate when type=constant
_C.OPTIM.WARMUP_CONS_LR = 1e-5
# Minimum learning rate when type=linear
_C.OPTIM.WARMUP_MIN_LR = 1e-5
# Recount epoch for the next scheduler (last_epoch=-1)
# Otherwise last_epoch=warmup_epoch
_C.OPTIM.WARMUP_RECOUNT = True

###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to only save the last one
_C.TRAIN.CHECKPOINT_FREQ = 0
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
# Use 'train_x', 'train_u' or 'smaller_one' to count
# the number of iterations in an epoch (for DA and SSL)
_C.TRAIN.COUNT_ITER = "train_x"

_C.TRAIN.EVAL_PERIOD = 1

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = "Classification"
_C.TEST.COMPUTE_CMAT = False
_C.TEST.SPLIT = "test"
# Which model to test after training (last_step or best_val)
# If best_val, evaluation is done every epoch (if val data
# is unavailable, test data will be used)
_C.TEST.FINAL_MODEL = "last_step"

_C.TRAINER = CN()

_C.TRAINER.TEXT = CN()
_C.TRAINER.TEXT.N_CLSN = 3
_C.TRAINER.TEXT.CTX_INIT = "a photo of a"


def get_cfg_default():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

def reset_cfg(cfg, args):
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir


def setup_cfg(args):
    cfg = get_cfg_default()

    # From the dataset config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    cfg.freeze()

    return cfg