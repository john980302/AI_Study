import numpy as np


def load_Dataset(data_path, input_step, output_step):
    inputs = []
    inputs_masks = []
    inputs_lengths = []
    labels = []
    labels_masks = []
    labels_lengths = []
    ENC_MAX_STEP = input_step
    DEC_MAX_STEP = output_step
    with open(data_path, 'r') as file:
        # 한 줄씩 읽기
        line = file.readline()

        # 한 줄이 끝나지 않으면
        while line:
            # 전체 입력 분리
            line = line.split()

            # input 부분
            one_input = np.zeros([DEC_MAX_STEP], dtype=np.float32)
            one_input_mask = np.zeros([DEC_MAX_STEP], dtype=int)
            length = 0
            i = 0
            # 'output'이라는 값 이전까지가 input이므로 그 전까지만 받음
            while (line[i] != 'output'):
                one_input[length] = float(line[i])
                one_input_mask[length] = 1
                i += 1
                length += 1
            inputs.append(one_input)
            inputs_masks.append(one_input_mask)
            inputs_lengths.append(length)
            i += 1
            
            # output 부분
            one_label = np.zeros([DEC_MAX_STEP], dtype=int)
            one_label_mask = np.zeros([DEC_MAX_STEP], dtype=int)
            length = 0
            one_label[length] = ENC_MAX_STEP + 1
            one_label_mask[length] = 1
            length += 1
            while (i < len(line)):
                # 0 = start_id, 1 = pad_id, 2 = end_id
                one_label[length] = int(line[i])
                one_label_mask[length] = 1
                length += 1
                i += 1
            one_label[length] = ENC_MAX_STEP + 2  # end token
            one_label_mask[length] = 1
            length += 1

            labels.append(one_label)
            labels_masks.append(one_label_mask)
            labels_lengths.append(length)
            line = file.readline()

    inputs = np.stack(inputs)
    inputs_masks = np.stack(inputs_masks)
    inputs_lengths = np.array(inputs_lengths)
    labels = np.stack(labels)
    labels_masks = np.stack(labels_masks)
    labels_lengths = np.array(labels_lengths)

    return inputs, labels

    print("읽어들인 입력 shape: ", inputs.shape)
    print("읽어들인 레이블 shape: ", labels.shape)
    cut = (inputs.shape[0]//10) * 9
    train_inputs = inputs[:cut]
    train_inputs_masks = inputs_masks[:cut]
    train_inputs_lengths = inputs_lengths[:cut]
    train_labels = labels[:cut]
    train_labels_masks = labels_masks[:cut]
    train_labels_lengths = labels_lengths[:cut]
    val_inputs = inputs[cut:]
    val_inputs_masks = inputs_masks[cut:]
    val_inputs_lengths = inputs_lengths[cut:]
    val_labels = labels[cut:]
    val_labels_masks = labels_masks[cut:]
    val_labels_lengths = labels_lengths[cut:]
    print("학습 입력 shape: ", train_inputs.shape)
    print("학습 레이블 shape: ", train_labels.shape)
    print("검증 입력 shape: ", val_inputs.shape)
    print("검증 레이블 shape: ", val_labels.shape)
    print(val_labels[0])


class Batcher:
    def __init__(self, x, y, randomize=False):
        self.test_x = x
        self.test_y = y
        self.test_random_idx = None
        self.test_random = randomize

    def get_batch(self, batch_size):
        def shuffle_idx(x):
            if (self.test_random):
                np.random.shuffle(x)

        if (self.test_random_idx is None):
            self.test_random_idx = np.arange(self.test_x.shape[0])
            shuffle_idx(self.test_random_idx)
        inp = []
        targ = []
        while (batch_size > self.test_random_idx.shape[0]):
            inp.append(self.test_x[self.test_random_idx])
            targ.append(self.test_y[self.test_random_idx])
            batch_size -= self.test_random_idx.shape[0]
            self.test_random_idx = np.arange(self.test_x.shape[0])
            shuffle_idx(self.test_random_idx)
        inp.append(self.test_x[self.test_random_idx[:batch_size]])
        targ.append(self.test_y[self.test_random_idx[:batch_size]])
        if (self.test_random_idx.shape[0] - batch_size == 0):
            self.test_random_idx = np.arange(self.test_x.shape[0])
            shuffle_idx(self.test_random_idx)
        else:
            self.test_random_idx = self.test_random_idx[batch_size:]
        inp = np.concatenate(inp, axis=0)
        targ = np.concatenate(targ, axis=0)
        return inp, targ

    def reset_random(self):
        self.test_random_idx = None
