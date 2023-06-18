from eval import main as e
import argparse
import warnings
import multiprocessing as mp
import numpy as np

warnings.filterwarnings("ignore")


def job(val_mAP_res, val_AD_res, val_F_res, index, mAP_res, AD_res, F_res, name, now, l, tag):
    l.acquire()
    i = now.value
    if i < len(name):
        now.value += 1
        l.release()
    else:
        l.release()
        return
    while True:
        max_mAP = 0
        max_mAP_index = 0
        length = 20
        for j in range(length):
            parser = argparse.ArgumentParser(description='Train a point-based transformer for action localization')
            parser.add_argument('--config', type=str, default='./configs/' + name[i] + '.yaml',
                                help='path to a config file', metavar='DIR')
            parser.add_argument('--ckpt', type=str,
                                default='./ckpt/' + name[i] + '_reproduce/epoch_' + str(31 + j).zfill(3) + '.pth.tar',
                                help='path to a checkpoint', metavar='DIR')
            parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions (default: -1)')
            parser.add_argument('--saveonly', action='store_true',
                                help='Only save the ouputs without evaluation (e.g., for test set)')
            parser.add_argument('-p', '--print-freq', default=10, type=int,
                                help='print frequency (default: 10 iterations)')
            args = parser.parse_args()
            buf1, buf2, buf3 = e(args, 'val_split', False)
            # score = buf1  # + buf3# - buf2 / 6
            val_mAP_res[i * 20 + j] = buf1
            val_AD_res[i * 20 + j] = buf2
            val_F_res[i * 20 + j] = buf3
            if buf1 > max_mAP:
                max_mAP = buf1
                max_mAP_index = j
        index[i] = max_mAP_index

        parser = argparse.ArgumentParser(description='Train a point-based transformer for action localization')
        parser.add_argument('--config', type=str, default='./configs/' + name[i] + '.yaml',
                            help='path to a config file', metavar='DIR')
        parser.add_argument('--ckpt', type=str,
                            default='./ckpt/' + name[i] + '_reproduce/epoch_' + str(31 + index[i]).zfill(
                                3) + '.pth.tar', help='path to a checkpoint', metavar='DIR')
        parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions (default: -1)')
        parser.add_argument('--saveonly', action='store_true',
                            help='Only save the ouputs without evaluation (e.g., for test set)')
        parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
        args = parser.parse_args()
        buf1, buf2, buf3 = e(args, 'test_split', False)
        mAP_res[i] = buf1
        AD_res[i] = buf2
        F_res[i] = buf3
        print('Process ' + str(tag) + ' finished ' + name[i] + '.')
        l.acquire()
        i = now.value
        if i < len(name):
            now.value += 1
            l.release()
        else:
            l.release()
            return


name = ['vessel_SpTeAttenRPCon']
val_mAP_res = mp.Array('f', 20 * len(name))
val_AD_res = mp.Array('f', 20 * len(name))
val_F_res = mp.Array('f', 20 * len(name))
index = mp.Array('i', len(name))
mAP_res = mp.Array('f', len(name))
AD_res = mp.Array('f', len(name))
F_res = mp.Array('f', len(name))
now = mp.Value('i', 0)
pool_num = 5
l = mp.Lock()
pool = [
    mp.Process(target=job, args=(val_mAP_res, val_AD_res, val_F_res, index, mAP_res, AD_res, F_res, name, now, l, i))
    for i in range(pool_num)]
for i in range(pool_num):
    pool[i].start()
for i in range(pool_num):
    pool[i].join()
val_mAP_res = np.array(val_mAP_res).reshape([len(name), 20])
index = np.array(index)
mAP_res = np.array(mAP_res)
AD_res = np.array(AD_res)
F_res = np.array(F_res)
print([31 + i for i in index])
for i in val_mAP_res:
    print(['{:.4f}'.format(j) for j in i])
print()
print(['{:.4f}'.format(i) for i in mAP_res])
print(['{:.4f}'.format(i) for i in AD_res])
print(['{:.4f}'.format(i) for i in F_res])
