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

from transformer_02_01 import transformer
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

DEC_MAX_STEP = DATA_MAX_OUTPUT + 1  # max output lengths
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

train_batcher = Batcher(train_inputs, train_labels, randomize=True)
val_batcher = Batcher(val_inputs, val_labels, randomize=False)

STEPS_PER_EPOCH = cut // BATCH_SIZE

TF = transformer(
    vocab_size=9000,
    num_layers=4,
    dff=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="small_transfomer")

checkpoint_dir = args.checkpoint_dir + suffix
checkpoint_prefix = os.path.join(checkpoint_dir, "tf")
checkpoint = tf.train.Checkpoint(**TF.get_model())

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

statistic_path = os.path.join(checkpoint_dir, "statistic.pck")
if os.path.exists(statistic_path):
    with open(statistic_path, "rb") as file:
        st = pickle.load(file)
        st_accs, st_step_accs, st_vaccs, st_tcs, st_gt_rates = st
else:
    st_accs = []
    st_step_accs = []
    st_vaccs = []
    st_tcs = []
    st_gt_rates = []


def validation_print(N=10,
                     use_beam=True,
                     beam_width=4,
                     FORCED_TO_MAKE_TRIANGLE=False,
                     USE_CIRCUMCIRCLE=False):
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    val_batcher.reset_random()
    acc = np.array([0, 0])
    step_acc = np.array([0, 0])
    vacc = np.array([0, 0])
    tc_acc = np.array([0, 0])
    gt_rate = np.array([0, 0])
    count_acc = np.array([0, 0])
    MSE = np.array([0, 0])
    MAE = np.array([0, 0])
    kwargs = {'use_beam_search': False,
              'return_historys': True,
              'FORCED_TO_MAKE_TRIANGLE': FORCED_TO_MAKE_TRIANGLE,
              'USE_CIRCUMCIRCLE': USE_CIRCUMCIRCLE
              }
    for _ in range(N):
        inp, lab = val_batcher.get_batch(BATCH_SIZE)

        result, tmp_accs = PN.eval(inp, lab, **kwargs)
        tmp_acc, tmp_step_acc, tmp_vacc, tmp_tc_acc, tmp_gt_rate, tmp_count_acc, tmp_MSE, tmp_MAE = tmp_accs
        acc += tmp_acc
        step_acc += tmp_step_acc
        vacc += tmp_vacc
        tc_acc += tmp_tc_acc
        gt_rate += tmp_gt_rate
        count_acc += tmp_count_acc
        MSE += tmp_MSE
        MAE += tmp_MAE
    acc = acc[0] / acc[1]
    step_acc = step_acc[0] / step_acc[1]
    vacc = vacc[0] / vacc[1]
    tc_acc = tc_acc[0] / tc_acc[1]
    gt_rate = gt_rate[0] / gt_rate[1]
    count_acc = count_acc[0] / count_acc[1]
    MSE = MSE[0] / MSE[1]
    RMSE = np.sqrt(MSE)
    MAE = MAE[0] / MAE[1]

    print(result.shape)

    print("Evaluation: ")
    print("strictly_acc: ", acc)
    print("step_acc: ", step_acc)
    print("virtual_acc: ", vacc)
    print("triangle coverage: ", tc_acc)
    print("Delaunay triangle rate: ", gt_rate)
    print("Count Accuracy: ", count_acc)
    print("MSE of the number of triangles:", MSE)
    print("RMSE of the number of triangles:", RMSE)
    print("MAE of the number of triangles:", MAE)
    for i in range(min(5, BATCH_SIZE)):
        print("=====================step {0}=====================".format(i + 1))
        r_str = "result: ["
        t_str = "target: ["
        for j in range(result.shape[1]):
            r_str += "{:^3}".format(result[i, j, 0])
        for j in range(DEC_MAX_STEP):
            t_str += "{:^3}".format(lab[i, j])
        r_str += "]"
        t_str += "]"
        print(r_str)
        print(t_str)
        # print("attention: ", atten[i])
    print("----------------------------------")

    if (use_beam):
        beam_size = BEAM_WIDTH

        val_batcher.reset_random()
        acc = np.array([0, 0])
        step_acc = np.array([0, 0])
        vacc = np.array([0, 0])
        tc_acc = np.array([0, 0])
        gt_rate = np.array([0, 0])
        count_acc = np.array([0, 0])
        MSE = np.array([0, 0])
        MAE = np.array([0, 0])
        kwargs = {'use_beam_search': True,
                  'beam_size': beam_width,
                  'return_historys': True,
                  'FORCED_TO_MAKE_TRIANGLE': FORCED_TO_MAKE_TRIANGLE,
                  'USE_CIRCUMCIRCLE': USE_CIRCUMCIRCLE
                  }
        for _ in range(N):
            inp, lab = val_batcher.get_batch(BATCH_SIZE)
            result, tmp_accs, history, history_score = PN.eval(inp, lab, **kwargs)
            tmp_acc, tmp_step_acc, tmp_vacc, tmp_tc_acc, tmp_gt_rate, tmp_count_acc, tmp_MSE, tmp_MAE = tmp_accs
            acc += tmp_acc
            step_acc += tmp_step_acc
            vacc += tmp_vacc
            tc_acc += tmp_tc_acc
            gt_rate += tmp_gt_rate
            count_acc += tmp_count_acc
            MSE += tmp_MSE
            MAE += tmp_MAE
        acc = acc[0] / acc[1]
        step_acc = step_acc[0] / step_acc[1]
        vacc = vacc[0] / vacc[1]
        tc_acc = tc_acc[0] / tc_acc[1]
        gt_rate = gt_rate[0] / gt_rate[1]
        count_acc = count_acc[0] / count_acc[1]
        MSE = MSE[0] / MSE[1]
        RMSE = np.sqrt(MSE)
        MAE = MAE[0] / MAE[1]

        # print(result.shape)

        print("Evaluation (using beam search): ")
        print("strictly_acc: ", acc)
        print("step_acc: ", step_acc)
        print("virtual_acc: ", vacc)
        print("triangle coverage: ", tc_acc)
        print("Delaunay triangle rate: ", gt_rate)
        print("Count Accuracy: ", count_acc)
        print("MSE of the number of triangles:", MSE)
        print("RMSE of the number of triangles:", RMSE)
        print("MAE of the number of triangles:", MAE)
        for i in range(min(5, BATCH_SIZE)):
            print("=====================step {0}=====================".format(i + 1))
            r_str = "result: ["
            t_str = "target: ["
            for j in range(result.shape[1]):
                r_str += "{:^3}".format(result[i, j, 0])
            for j in range(DEC_MAX_STEP):
                t_str += "{:^3}".format(lab[i, j])
            r_str += "]"
            t_str += "]"
            print(r_str)
            print(t_str)
            # print("attention: ", atten[i])
        """
        print("----------------------------------")
        for i in range(min(5, BATCH_SIZE)):
            print("history {0}".format(i + 1))
        #for i in range(len(history)):
            #print("step {0}".format(i +1))
            #print(history[i][:beam_size])
            #print(history_score[i][:beam_size])
            #print("===========================")
            print(history[-1][i*beam_size:(i + 1)*beam_size])
        """
    return acc, step_acc, vacc, tc_acc, gt_rate


if len(st_accs) == 0:
    validation_print(N=VAL_STEP, use_beam=False)

if not FORWARD_ONLY:
    for epoch in range(EPOCHS):
        start = time.time()

        total_loss = 0

        for batch in range(STEPS_PER_EPOCH):
            inp, targ = train_batcher.get_batch(BATCH_SIZE)

            batch_loss = PN.step(inp, targ, clipping_value=CLIPPING_VALUE)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
            # if batch % 1000 == 0:
            #    validation_print()

        # saving (checkpoint) the model every 2 epochs
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / STEPS_PER_EPOCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        acc, step_acc, vacc, tc_acc, gt_rate = validation_print(N=VAL_STEP,
                                                                use_beam=False,
                                                                FORCED_TO_MAKE_TRIANGLE=False,
                                                                USE_CIRCUMCIRCLE=False)
        st_accs.append(acc)
        st_step_accs.append(step_acc)
        st_vaccs.append(vacc)
        st_tcs.append(tc_acc)
        st_gt_rates.append(gt_rate)
        with open(statistic_path, "wb") as file:
            pickle.dump([st_accs, st_step_accs, st_vaccs, st_tcs, st_gt_rates],
                        file)

print("training end....")
print("===========Options==========")
print("Batch size:", BATCH_SIZE)
print("Units:", UNITS)
print("Beam width:", BEAM_WIDTH)
print("Test Step:", TEST_STEP)
print("Use Self attention:", USE_SELF_ATTENTION)

print("penalty off--")
FORCED_TO_MAKE_TRIANGLE = False
USE_CIRCUMCIRCLE = False
validation_print(N=TEST_STEP,
                 use_beam=True,
                 beam_width=BEAM_WIDTH,
                 FORCED_TO_MAKE_TRIANGLE=FORCED_TO_MAKE_TRIANGLE,
                 USE_CIRCUMCIRCLE=USE_CIRCUMCIRCLE)

print("penalty on--")
USE_CIRCUMCIRCLE = True
FORCED_TO_MAKE_TRIANGLE = True
validation_print(N=TEST_STEP,
                 use_beam=True,
                 beam_width=BEAM_WIDTH,
                 FORCED_TO_MAKE_TRIANGLE=FORCED_TO_MAKE_TRIANGLE,
                 USE_CIRCUMCIRCLE=USE_CIRCUMCIRCLE)

plt.plot(st_accs, label='acc')
plt.plot(st_step_accs, label='step acc')
plt.plot(st_vaccs, label='vacc')
plt.plot(st_tcs, label='triangle coverage')
plt.plot(st_gt_rates, label='Del rate')

plt.legend()
plt.savefig('graph' + suffix + '.png')
# plt.show()
