import tensorflow as tf
import argparse
import os
import pickle
import io
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transformer_02_02 import transformer, CustomSchedule, loss_function
from util import load_Dataset, Batcher

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_points",
                    type=int,
                    default=5,
                    help="the number of points of dataset.")
parser.add_argument("--data_path",
                    type=str,
                    default="./data/delaunay_5_100man.txt")
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default="./training_checkpoints/pn")
parser.add_argument("--suffix",
                    type=str,
                    default="")

parser.add_argument("-b", "--batch_size",
                    type=int,
                    default=256)
parser.add_argument("-e", "--epochs",
                    type=int,
                    default=50)
parser.add_argument("--val_step",
                    type=int,
                    default=10)
parser.add_argument("--test_step",
                    type=int,
                    default=20)
parser.add_argument("-u", "--units",
                    type=int,
                    default=256)
parser.add_argument("-l", "--learning_rate",
                    type=float,
                    default=0.002)
parser.add_argument("-B", "--beam_width",
                    type=int,
                    default=4)
parser.add_argument("--output_std",
                    type=float,
                    default=None)
parser.add_argument("--tri_penalty",
                    action="store_true")
parser.add_argument("--del_penalty",
                    action="store_true")

parser.add_argument("--use_sa",
                    action="store_true")
parser.add_argument("--dropout_rate",
                    type=float,
                    default=0.0)
parser.add_argument("--clipping_value",
                    type=float,
                    default=None)
parser.add_argument("-F", "--forward_only",
                    action="store_true")

args = parser.parse_args()

ENC_MAX_STEP = delaunay = args.n_points
suffix = str(delaunay) + args.suffix
# 2n - 5
DATA_MAX_OUTPUT = (2 * delaunay - 5) * 3

DATA_PATH = args.data_path

# DATA_PATH = "./data/delaunay_5_100man.txt"

DEC_MAX_STEP = DATA_MAX_OUTPUT + 2  # max output lengths
START_ID = 0
PAD_ID = 1
END_ID = 2
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
VAL_STEP = args.val_step
TEST_STEP = args.test_step
UNITS = args.units
LEARNING_RATE = args.learning_rate
BEAM_WIDTH = args.beam_width
OUTPUT_STD = args.output_std
FORCED_TO_MAKE_TRIANGLE = args.tri_penalty
USE_CIRCUMCIRCLE = args.del_penalty
USE_SELF_ATTENTION = args.use_sa
DROPOUT_RATE = args.dropout_rate
CLIPPING_VALUE = args.clipping_value
FORWARD_ONLY = args.forward_only

inputs, labels = load_Dataset(DATA_PATH, ENC_MAX_STEP, DEC_MAX_STEP)

train_data_ratio = 0.9
cut = int(inputs.shape[0] * train_data_ratio)

train_inputs = inputs[:cut]
train_labels = labels[:cut]

val_inputs = inputs[cut:]
val_labels = labels[cut:]

STEPS_PER_EPOCH = cut // BATCH_SIZE

print(inputs.shape)
print(labels.shape)
print(labels[0])

BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': train_inputs,
        'dec_inputs': train_labels[:, :-1] # 디코더의 입력, 마지막 패딩 토큰이 제거
    },
    {
        'outputs': train_labels[:, 1:]     # 시작 토근 제거
    }
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()

# Hyper-parameters
vocab_size = DEC_MAX_STEP
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

model = transformer(
    vocab_size=vocab_size,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
    MAX_LENGTH = DEC_MAX_STEP
    # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

EPOCHS = 1
model.fit(dataset, epochs=EPOCHS)   

def evaluate():
    
    END_TOKEN = ENC_MAX_STEP + 2
    START_TOKEN = ENC_MAX_STEP + 1
    
    eval_inputs = tf.expand_dims(val_inputs[0], 0)
    eval_outputs = tf.ones_like(eval_inputs) * START_TOKEN
    eval_labels = val_labels[0]
    
    # 디코더의 예측 시작
    for i in range(DEC_MAX_STEP):
        predictions = model(inputs=[eval_inputs, eval_outputs], training=False)
        
        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN):
            break
        
        # 마지막 시점의 예측 단어를 출력
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정
        output = tf.concat([output, predicted_id], axis=-1)
        
    print("eval_inputs: ", eval_inputs)
    print("eval_outputs: ", output)
    print("eval_labels: ", eval_labels)
        
    return tf.squeeze(output, axis=0)
    
evaluate()
