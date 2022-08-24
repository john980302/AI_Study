from tqdm import tqdm
from dataset.cad_dataset import get_dataloader
from config import ConfigAE
from utils import ensure_dir
from trainer import TrainerAE
import torch
import numpy as np
import os
import h5py
from cadlib.macro import EOS_IDX, SOL_IDX, EXT_IDX, CIRCLE_IDX, LINE_IDX, ARC_IDX 
import random
from cadlib.visualize import vec2CADsolid

def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigAE('test')

    if cfg.mode == 'rec':
        reconstruct(cfg)
    elif cfg.mode == 'enc':
        encode(cfg)
    elif cfg.mode == 'dec':
        decode(cfg)
    elif cfg.mode == 'enc_mem':
        encode_memory_state(cfg)
    elif cfg.mode == 'dec_mem':
        decode_memory_state(cfg)
    else:
        raise ValueError


def reconstruct(cfg):
    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # create dataloader
    test_loader = get_dataloader('test', cfg)
    print("Total number of test data:", len(test_loader))

    if cfg.outputs is None:
        cfg.outputs = "{}/results/test_{}".format(cfg.exp_dir, cfg.ckpt)
    ensure_dir(cfg.outputs)

    # evaluate
    pbar = tqdm(test_loader)
    for i, data in enumerate(pbar):
        batch_size = data['command'].shape[0]
        commands = data['command']
        args = data['args']
        gt_vec = torch.cat([commands.unsqueeze(-1), args], dim=-1).squeeze(1).detach().cpu().numpy()
        commands_ = gt_vec[:, :, 0]
        with torch.no_grad():
            outputs, _ = tr_agent.forward(data, i)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(batch_size):
            out_vec = batch_out_vec[j]
            seq_len = commands_[j].tolist().index(EOS_IDX)

            data_id = data["id"][j].split('/')[-1]

            save_path = os.path.join(cfg.outputs, '{}_vec.h5'.format(data_id))
            with h5py.File(save_path, 'w') as fp:
                fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)
                fp.create_dataset('gt_vec', data=gt_vec[j][:seq_len], dtype=np.int)


def encode(cfg):
    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # create dataloader
    save_dir = "{}/results".format(cfg.exp_dir)
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, 'all_zs_ckpt{}.h5'.format(cfg.ckpt))
    fp = h5py.File(save_path, 'w')
    for phase in ['train', 'validation', 'test']:
        train_loader = get_dataloader(phase, cfg, shuffle=False)

        # encode
        all_zs = []
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            with torch.no_grad():
                z = tr_agent.encode(data, is_batch=True)
                z = z.detach().cpu().numpy()[:, 0, :]
                all_zs.append(z)
        all_zs = np.concatenate(all_zs, axis=0)
        print(all_zs.shape)
        fp.create_dataset('{}_zs'.format(phase), data=all_zs)
    fp.close()


def decode(cfg):
    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # load latent zs
    with h5py.File(cfg.z_path, 'r') as fp:
        zs = fp['zs'][:]
    save_dir = cfg.z_path.split('.')[0] + '_dec'
    ensure_dir(save_dir)

    # decode
    for i in range(0, len(zs), cfg.batch_size):
        with torch.no_grad():
            batch_z = torch.tensor(zs[i:i+cfg.batch_size], dtype=torch.float32).unsqueeze(1)
            batch_z = batch_z.cuda()
            outputs = tr_agent.decode(batch_z)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(len(batch_z)):
            out_vec = batch_out_vec[j]
            out_command = out_vec[:, 0]
            seq_len = out_command.tolist().index(EOS_IDX)

            save_path = os.path.join(save_dir, '{}.h5'.format(i + j))
            with h5py.File(save_path, 'w') as fp:
                fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)


def encode_memory_state(cfg):
    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # create dataloader
    save_dir = "{}/results".format(cfg.exp_dir)
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, 'all_mem_ckpt{}.h5'.format(cfg.ckpt))
    fp = h5py.File(save_path, 'w')
    for phase in ['train', 'validation', 'test']:
        train_loader = get_dataloader(phase, cfg, shuffle=False)

        # encode memory state
        all_mem = []
        pbar = tqdm(train_loader)
        # memory state를 저장할 데이터만 100개 추출
        if phase == 'train':
            choice_list = ['True']*100 + ['False']*(len(pbar) - 100)
            random.shuffle(choice_list)
        else:
            choice_list = ['True']*len(pbar)
        
        # memory state를 계산해서 저장
        for i, data in enumerate(pbar):
            if choice_list[i] == 'True':
                with torch.no_grad():
                    mem = tr_agent.encode_mem(data, is_batch=True)
                    mem_con = mem[0].detach().cpu().numpy()
                    for i in range(1, len(mem)):
                        mem_val = mem[i].detach().cpu().numpy()
                        mem_con = np.concatenate([mem_con, mem_val], axis=0)
                    all_mem.append(mem_con)
        all_mem = np.concatenate(all_mem, axis=0)
        print(all_mem.shape)
        fp.create_dataset('{}_mem'.format(phase), data=all_mem)
    fp.close()
    
def decode_memory_state(cfg):
    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # save directory 
    save_dir = cfg.m_path.split('.')[0] + '_dec_mem'
    ensure_dir(save_dir)
    
    # memory state
    with h5py.File(cfg.m_path, 'r') as fp:
        train_mem = fp['train_mem'][:]

    # decode memory state
    memory_state_all_size = 300
    memory_state_all_cnt = 5
    
    for i in tqdm(range(0, len(train_mem), memory_state_all_size)):
        with torch.no_grad():
            # memory state 생성
            batch_mems_list = []
            for k in range(memory_state_all_cnt):
                batch_mems = torch.tensor(train_mem[i+60*k:i+60*(k+1)], dtype=torch.float32).cuda()
                batch_mems_list.append(batch_mems)
            
            # random z 생성
            random_z = torch.rand(cfg.batch_size, 1, cfg.d_embed).cuda()
            
            # new CAD Sequence 생성
            outputs = tr_agent.decode_mem(random_z, batch_mems_list)
            batch_out_vec = tr_agent.logits2vec(outputs)
        
        repair_before_good = 0
        repair_after_good = 0
        repair_before_bad = 0
        repair_after_bad = 0
        # len(random_z): 512 (batch_size)
        for j in range(len(random_z)):
            out_vec = batch_out_vec[j]
            out_command = out_vec[:, 0]
            seq_len = out_command.tolist().index(EOS_IDX)

            repair_before = out_vec[:seq_len]
            repair_after = face_check(repair_before)
            
            save_path = os.path.join(save_dir, '{}.h5'.format(i + j))
            with h5py.File(save_path, 'w') as fp:
                fp.create_dataset('out_vec', data=repair_after, dtype=np.int)

       
def face_check(cad_vec):
    sol_idx_list = np.where(cad_vec[:, 0] == SOL_IDX)[0].tolist()
    s_len = len(sol_idx_list)
    for i in range(s_len - 1):
        # face의 처음과 끝을 계산
        start_idx = sol_idx_list[i] + 1
        end_idx = sol_idx_list[i + 1]
        # face의 끝의 값이 extrude인 경우를 제외
        if cad_vec[end_idx - 1, 0] == EXT_IDX:
            end_idx = end_idx - 1
        
        # start_idx와 end_idx가 같은 경우
        if start_idx != end_idx:
            # face 체크하여 수정
            cad_vec[start_idx:end_idx, :] = repair_face(cad_vec[start_idx:end_idx, :])
    
    return cad_vec
        
        
def repair_face(cad_vec):
    #print(cad_vec)
    # command가 Circle(2)인 경우에는 수정하지 않음
    # command가 Line과 Arc로 이뤄진 경우에만 계산
    if cad_vec[0, 0] in [LINE_IDX, ARC_IDX]:
        cad_vec[-1, 1:3] = [128, 128]
    
    return cad_vec
        
def is_unique(cfg, cad_vec):
    # 일단 cad_vec가 생성되어야 계산함
    
    unique_score = 0
    
    # extrude 단위로 분리해서 비교
    commands = cad_vec[:, 0]
    ext_idx = [-1] + np.where(commands == EXT_IDX)[0].tolist()
    ext_len = len(ext_idx)
    
    all_occ = []
    
    # create dataloader
    for phase in ['train', 'validation', 'test']:
        original_data_loader = get_dataloader(phase, cfg, shuffle=False)
        pbar = tqdm(original_data_loader)
        
        for i, data in enumerate(pbar):
            # data.shape: [batch_size(512), Nc(60), cmds+args(17)]
            cmds = data['command'].unsqueeze(2).numpy()
            args = data['args'].numpy()
            origin_batch_data = np.concatenate((cmds, args), axis=2)
            # origin_data.shape: [Nc(60), cmds+args(17)]
            for origin_data in origin_batch_data:
                original_cmds = origin_data[:, 0]
                original_ext_idx = [-1] + np.where(original_cmds == EXT_IDX)[0].tolist()
                original_ext_len = len(original_ext_idx)
                occ = []
                for k in range(original_ext_len-1):
                    o_start, o_end = original_ext_idx[k]+1, original_ext_idx[k+1]+1
                    cc = []
                    for j in range(ext_len-1):
                        start, end = ext_idx[j]+1, ext_idx[j+1]+1
                        cc.append(cross_correlation(origin_data[o_start:o_end, :],cad_vec[start:end, :]))
                    occ.append(min(cc))
                all_occ.append(min(occ))
    
    
    unique_score = min(all_occ)
    print('unique_score: ', unique_score)
    return unique_score

def cross_correlation(origin, pred):
    # correlation score
    cs = 0
    
    idx = 0
    end_len = len(origin) if len(origin) < len(pred) else len(pred)
    # 둘 중 하나가 끝날 때까지
    # cos_sim의 범위가 -1 ~ 1이여서 0 ~ 1으로 변경
    while idx < end_len:
        if origin[idx][0] == pred[idx][0] & origin[idx][0] != SOL_IDX:
            cs = cs + (cos_sim(origin[idx][1:],pred[idx][1:]) + 1) / 2
        idx = idx + 1
    
    # 길이 보정
    cs = cs / end_len
    
    # cs: 0일수록 unique하고 1일수록 unique하지 않는다.
    return 1 - cs

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    main()
