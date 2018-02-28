# Attentional Pooling for Action Recognition

If this code helps with your work/research, please consider citing

Rohit Girdhar and Deva Ramanan. **Attentional Pooling for Action Recognition**. Advances in Neural Information Processing Systems (NIPS), 2017.

```txt
@inproceedings{Girdhar_17b_AttentionalPoolingAction,
    title = {Attentional Pooling for Action Recognition},
    author = {Girdhar, Rohit and Ramanan, Deva},
    booktitle = {NIPS},
    year = 2017
}
```

## Pre-requisites

This code was trained and tested with

1. CentOS 6.5
2. Python 2.7
3. TensorFlow 1.1.0-rc2 ([6a1825e2](https://github.com/tensorflow/tensorflow/tree/6a1825e2369d2537e15dc585705c53c4b763f3f6))

## Getting started

Clone the code and create some directories for outputs

```bash
$ git clone --recursive https://github.com/rohitgirdhar/AttentionalPoolingAction.git
$ export ROOT=`pwd`/AttentionalPoolingAction
$ cd $ROOT/src/
$ mkdir -p expt_outputs data
$ # compile some custom ops
$ cd custom_ops; make; cd ..
```

## Data setup

You can download the `tfrecord` files for MPII I used from
[here](https://cmu.box.com/shared/static/xb7esevyl6uzmra2eehnkbt2ud7awld9.tar)
and uncompress on to a fast local disk.
If you want to create your own tfrecords, you can use the following steps, which is
what I used to create the linked tfrecord files

Convert the MPII data into tfrecords. The system also can read from individual JPEG files,
but that needs a slightly different intial setup.

First download the MPII [images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)
and [annotations](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip),
and un-compress the files.

```bash
$ cd $ROOT/utils/dataset_utils
$ # Set the paths for MPII images and annotations file in gen_tfrecord_mpii.py
$ python gen_tfrecord_mpii.py  # Will generate the tfrecord files
```

## Testing pre-trained models

First download and unzip the
[pretrained models](https://cmu.box.com/shared/static/s72scgtjj3lm60hsufi25rfjs2dk3a7i.zip)
to a `$ROOT/src/pretrained_models/`.
The models can be run by

```bash
# Baseline model (no attention)
$ python eval.py --cfg ../experiments/001_MPII_ResNet_pretrained.yaml
# With attention
$ python eval.py --cfg ../experiments/002_MPII_ResNet_pretrained.yaml
# With pose regularized attention
$ python eval.py --cfg ../experiments/003_MPII_ResNet_withPoseAttention_pretrained.yaml
```

### Expected performance on MPII Validation set

| Method  | mAP | Accuracy |
|--------|-----|------|
| Baseline (no attention) | 26.2 | 33.5 |
| With attention | 30.3 | 37.2 |
| With pose regularized attention | 30.6 | 37.8 |

## Training

Train a attentional pooled model on MPII dataset, using `python train.py --cfg <path to YAML file>`.

```bash
$ cd $ROOT/src
$ python train.py --cfg ../experiments/002_MPII_ResNet_withAttention.yaml
# To train the model with pose regularized attention, use the following config
$ python train.py --cfg ../experiments/003_MPII_ResNet_withPoseAttention.yaml
# To train the baseline without attention, use the following config
$ python train.py --cfg ../experiments/001_MPII_ResNet.yaml
```

## Testing and evaluation

Test the model trained above on the validation set, using `python eval.py --cfg <path to YAML file>`.

```bash
$ python eval.py --cfg ../experiments/002_MPII_ResNet_withAttention.yaml
# To evaluate the model with pose regularized attention
$ python eval.py --cfg ../experiments/003_MPII_ResNet_withPoseAttention.yaml
# To evaluate the model without attention
$ python train.py --cfg ../experiments/001_MPII_ResNet.yaml
```

The performance of these models should be similar to the above
released pre-trained models.

## Train + test on the final test set

This is for getting the final number on MPII test set.

```bash
# Train on the train + val set
$ python train.py --cfg ../experiments/002_MPII_ResNet_withAttention_train+val.yaml
# Test on the test set
$ python eval.py --cfg ../experiments/002_MPII_ResNet_withAttention_train+val.yaml --save
# Convert the output into the MAT files as expected by MPII authors (requires matlab/octave)
$ cd ../utils;
$ bash convert_mpii_result_for_eval.sh ../src/expt_outputs/002_MPII_ResNet_withAttention_train+val.yaml/<filename.h5>
# Now the generated mat file can be emailed to MPII authors for test evaluation
```


# 代码解读
## 参数
cfg_file文件，可选

```
__C的结构

INPUT:
    INPUT_IMAGE_FORMAT:'normal'
    INPUT_IMAGE_FORMAT_POSE_RENDER_TYPE = 'rgb'
    POSE_GLIMPSE_CONTEXT_RATIO = 0.0
    POSE_GLIMPSE_RESIZE = False
    POSE_GLIMPSE_PARTS_KEEP = []
    SPLIT_ID = 1
    VIDEO:
        MODALITY = 'rgb'
TRAIN:
    BATCH_SIZE = 10
    WEIGHT_DECAY = 0.0005
    CLIP_GRADIENTS = -1.0
    IMAGE_SIZE = 450
    RESIZE_SIDE = 480
    FINAL_POSE_HMAP_SIDE = 15
    LABEL_SMOOTHING = False
    MOVING_AVERAGE_VARIABLES = None
    FINAL_POSE_HMAP_SIDE = 15
    LABEL_SMOOTHING = False
    MOVING_AVERAGE_VARIABLES = None
    LEARNING_RATE = 0.01
    LEARNING_RATE_DECAY_RATE = 0.33
    END_LEARNING_RATE = 0.00001
    NUM_STEPS_PER_DECAY = 0 
    NUM_EPOCHS_PER_DECAY = 40.0
    LEARNING_RATE_DECAY_TYPE = 'exponential'
    OPTIMIZER = 'momentum'
    MOMENTUM = 0.9
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    OPT_EPSILON = 1.0
    TRAINABLE_SCOPES = ''
    MAX_NUMBER_OF_STEPS = 100000
    LOG_EVERY_N_STEPS = 10
    SAVE_SUMMARIES_SECS = 300
    SAVE_INTERVAL_SECS = 1800
    IGNORE_MISSING_VARS = True
    CHECKPOINT_PATH = 'data/pretrained_models/inception_v3.ckpt'
    CHECKPOINT_EXCLUDE_SCOPES = ''
    DATASET_SPLIT_NAME = 'trainval_train'

    LOSS_FN_POSE = 'l2'  # can be 'l2'/'log-loss'/'sigmoid-log-loss'/'cosine-loss'
    LOSS_FN_POSE_WT = 1.0
    LOSS_FN_POSE_SAMPLED = False  # Harder loss, sample the negatives
    LOSS_FN_ACTION = 'softmax-xentropy'  # can be 'softmax-xentropy'
    LOSS_FN_ACTION_WT = 1.0
    VAR_NAME_MAPPER = ''  # to be used when loading from npy checkpoints
    VIDEO_FRAMES_PER_VIDEO = 1
    READ_SEGMENT_STYLE = False
    ITER_SIZE = 1  # accumulate gradients over this many iterations
    OTHER_IMG_SUMMARIES_TO_ADD = ['PosePrelogitsBasedAttention']

TEST:
    BATCH_SIZE = 10
    DATASET_SPLIT_NAME = 'trainval_val'
    MAX_NUM_BATCHES = None 
    CHECKPOINT_PATH = b''
    MOVING_AVERAGE_DECAY = None
    VIDEO_FRAMES_PER_VIDEO = 1  # single image dataset. Set 25 for hmdb
    EVAL_METRIC = ''  # normal eval. Set ='mAP' to compute that.    

NET:
    USE_POSE_ATTENTION_LOGITS = False
    USE_POSE_ATTENTION_LOGITS_DIMS = [-1]  # by default use all parts
    USE_POSE_ATTENTION_LOGITS_AVGED_HMAP = False
    USE_POSE_LOGITS_DIRECTLY = False
    USE_POSE_LOGITS_DIRECTLY_PLUS_LOGITS = False
    USE_POSE_LOGITS_DIRECTLY_v2 = False
    USE_POSE_LOGITS_DIRECTLY_v2_EXTRA_LAYER = False
    USE_POSE_PRELOGITS_BASED_ATTENTION = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_DEBUG = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_SOFTMAX_ATT = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_RELU_ATT = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_PER_CLASS = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_SINGLE_LAYER_ATT = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_WITH_POSE_FEAT = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_WITH_POSE_FEAT_2LAYER = False
    USE_POSE_PRELOGITS_BASED_ATTENTION_RANK = 1
    USE_TEMPORAL_ATT = False
    USE_COMPACT_BILINEAR_POOLING = False
    LAST_CONV_MAP_FOR_POSE:
        inception_v2_tsn = 'InceptionV2_TSN/inception_5a'
        inception_v3 = 'Mixed_7c'
        resnet_v1_101 = 'resnet_v1_101/block4'
        vgg_16 = 'vgg_16/conv5'
    TRAIN_TOP_BN = False
    DROPOUT = -1.0

RNG_SEED = 42
EPS = 1e-14
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data')) 
EXP_DIR = 'expt_outputs/' 
DATASET_NAME = 'mpii' 
DATASET_DIR = 'data/mpii/mpii_tfrecords'
DATASET_LIST_DIR = ''
MODEL_NAME = 'inception_v3'
NUM_READERS = 4
NUM_PREPROCESSING_THREADS = 4
GPUS = '2'
HEATMAP_MARKER_WD_RATIO = 0.1
MAX_INPUT_IMAGE_SIZE = 512  # to avoid arbitrarily huge input images
INPUT_FILE_STYLE_LABEL = ''

```
说白了，以上edict是默认配置，要根据不同阶段需要修改部分配置，所以加了个cfg_file，也就是.yaml文件，用yaml文件更新原默认配置即可。
(更新的方法感兴趣的可以看一看，就是递归地用yaml中的项替换原默认配置中的项，yaml存在而默认中不存在的或是类型不匹配的统统报错)
























