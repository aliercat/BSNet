from __future__ import division

import os
import sys
import logging
import torch
import numpy as np

from thop import profile
sys.path.append("../")

#from utils.darts_utils_RT8 import create_exp_dir, plot_op, plot_path_width, objective_acc_lat
try:
    from utils.darts_utils_RT8 import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from utils.darts_utils_RT8 import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")
# from utils.darts_utils_RT8_FP16 import compute_latency_ms_pytorch as compute_latency
# print("use PyTorch for latency test")


# from utils.darts_utils import create_exp_dir, plot_op, plot_path_width, objective_acc_lat
# try:
#     from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
#     print("use TensorRT for latency test")
# except:
#     from utils.darts_utils import compute_latency_ms_pytorch as compute_latencyl
#     print("use PyTorch for latency test")
# from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
# print("use PyTorch for latency test")


from models.my_model_stages_trt import BiSeNet
# from models.model_stages import BiSeNet

def main():
    
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configuration ##############
    use_boundary_2 = False
    use_boundary_4 = False
    use_boundary_8 = True
    use_boundary_16 = False
    use_conv_last = False
    n_classes = 19
    
    # STDC1Seg-50 250.4FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'STDC1-Seg'
    # inputSize = 512
    # inputScale = 50
    # inputDimension = (1, 3, 512, 1024)

    # # STDC1Seg-75 126.7FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'STDC1-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)

    # # STDC2Seg-50 188.6FPS on NVIDIA GTX 1080Ti
    backbone = 'STDCNet1446'
    methodName = 'STDC2-Seg'
    inputSize = 512
    inputScale = 50
    inputDimension = (1, 3, 512, 1024)

    # n_classes = 11
    # backbone = 'STDCNet1446'
    # methodName = 'Cam'
    # inputScale = 1
    # inputSize = 720
    # inputDimension = (1, 3, 704, 960)

    # # STDC2Seg-75 97.0FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)
    
    model = BiSeNet(backbone=backbone, n_classes=n_classes, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
    use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
    input_size=inputSize, use_conv_last=use_conv_last)
    

    print('loading parameters...')
    respth = './checkpoints/base_sesp_BGA/STDC2-Seg/'

    save_pth = os.path.join(respth, 'model_maxmIOU{}.pth'.format(inputScale))
    model.load_state_dict(torch.load(save_pth), strict=False)
    #, strict=False
    model = model.cuda()
    #####################################################

    latency = compute_latency(model, inputDimension)
    #,iterations=1000
    print("{}{} FPS:".format(methodName, inputScale) + str(1000./latency))
    logging.info("{}{} FPS:".format(methodName, inputScale) + str(1000./latency))

    # calculate FLOPS and params
    '''
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    '''


if __name__ == '__main__':
    main() 
