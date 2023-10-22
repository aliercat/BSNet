# author: muzhan
import os
import sys
import time

cmd = "export CUDA_VISIBLE_DEVICES=0,1 && nohup python -m torch.distributed.launch --nproc_per_node=2 train.py --respath /common_sdb/dfr/STDCNew/18/train_STDC2-Seg/ --backbone STDCNet1446 --mode train --n_workers_train 12 --n_workers_val 1 --n_img_per_gpu 4 --max_iter 120000 --use_boundary_8 True > nohup.out 2>&1 & "



def gpu_info(gpu_index=1):

    info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')
    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])
    return power, memory

def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 1000 or gpu_power > 80:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()