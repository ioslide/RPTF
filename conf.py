from yacs.config import CfgNode as CN
import getpass
username = getpass.getuser()

_C = CN()
cfg = _C

# ----------------------------- loader options -------------------------- #
_C.LOADER = CN()

_C.LOADER.SAMPLER = CN()
_C.LOADER.SAMPLER.TYPE = "sequence"
_C.LOADER.SAMPLER.GAMMA = 0.1

_C.LOADER.NUM_WORKS = 4

# --------------------------------- Contrastive options --------------------- #
_C.CONTRAST = CN()

_C.CONTRAST.TEMPERATURE = 0.1

_C.CONTRAST.PROJECTION_DIM = 128

_C.CONTRAST.MODE = 'all'
_C.MODEL = CN()

_C.MODEL.ARCH = 'Standard'
_C.MODEL.FEATURE_DIM = 128

_C.MODEL.EPISODIC = False

_C.MODEL.PROJECTION = CN()

_C.MODEL.PROJECTION.HEAD = "linear"
_C.MODEL.PROJECTION.EMB_DIM = 2048
_C.MODEL.PROJECTION.FEA_DIM = 128

_C.CORRUPTION = CN()

_C.CORRUPTION.DATASET = 'cifar10'
_C.CORRUPTION.SOURCE = ''
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5]
_C.CORRUPTION.GRADUAL_SEVERITY = [1,2,3,4,5,4,3,2,1]
_C.CORRUPTION.NUM_EX = 10000
_C.CORRUPTION.NUM_CLASS = -1
_C.CORRUPTION.ORDER_NUM = 11

_C.MODEL.ADAPTATION = 'source'
_C.INPUT = CN()

_C.INPUT.SIZE = (32, 32)

_C.OPTIM = CN()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3
_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.WD = 0.0
_C.OPTIM.EPS = 1e-5
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64

_C.SEED = -1
_C.OUTPUT_DIR = "./output"
_C.DATA_DIR = f"/home/{username}/datasets/"
_C.NOTE = 'none'
_C.CKPT_DIR = "./models"
_C.CKPT_PATH = "./models"
_C.LOG_DIR = "./log"
_C.bash_file_name = 'run.sh'
_C.DEBUG = 0
_C.ENABLE_PROGRESS_BAR = False
_C.SETTING = ''

_C.ADAPTER = CN()
_C.ADAPTER.NAME = "OURS"

_C.ADAPTER.BN = CN()
_C.ADAPTER.BN.RESET_STATS = False
_C.ADAPTER.BN.NO_STATS = False
_C.ADAPTER.BN.EPS = 1e-5

_C.ADAPTER.CoTTA = CN()
_C.ADAPTER.CoTTA.RST = 0.01
_C.ADAPTER.CoTTA.AP = 0.92
_C.ADAPTER.CoTTA.MT = 0.999

_C.ADAPTER.RoTTA = CN()
_C.ADAPTER.RoTTA.MEMORY_SIZE = 64
_C.ADAPTER.RoTTA.UPDATE_FREQUENCY = 64
_C.ADAPTER.RoTTA.NU = 0.001
_C.ADAPTER.RoTTA.ALPHA = 0.05
_C.ADAPTER.RoTTA.LAMBDA_T = 1.0
_C.ADAPTER.RoTTA.LAMBDA_U = 1.0

_C.ADAPTER.TEA = CN()
_C.ADAPTER.TEA.UNCOND = 'uncond'
_C.ADAPTER.TEA.STEPS = 20
_C.ADAPTER.TEA.SGLD_LR = 0.1
_C.ADAPTER.TEA.SGLD_STD = 0.01
_C.ADAPTER.TEA.BUFFER_SIZE = 10000
_C.ADAPTER.TEA.REINIT_FREQ = 0.05
_C.ADAPTER.TEA.IMG_SIZE = 32
_C.ADAPTER.TEA.NUM_CHANNEL = 3

_C.ADAPTER.TRIBE = CN()
_C.ADAPTER.TRIBE.ETA = 0.005
_C.ADAPTER.TRIBE.GAMMA = 0.0
_C.ADAPTER.TRIBE.LAMBDA = 0.5
_C.ADAPTER.TRIBE.H0 = 0.05

_C.ADAPTER.LAW = CN()
_C.ADAPTER.LAW.TAU = 1.


_C.ADAPTER.SAR = CN()

_C.ADAPTER.Source = CN()

_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

