from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import time
import torch.nn.functional as F
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ckpt_path=r"models/CDDFuse_IVF.pth"
dataset_name = "LLVIP_test"
print("\n"*2+"="*80)
model_name="CDDFuse"
print("The test result of "+dataset_name+' :')
test_folder=os.path.join('test_img',dataset_name)
test_out_folder=os.path.join('test_result',dataset_name)
if not os.path.exists(test_out_folder):
    os.mkdir(test_out_folder)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])


Encoder.eval()
Decoder.eval()
BaseFuseLayer.eval()
DetailFuseLayer.eval()
def vision_features(feature_map,img_type,name):
    count = 0
    # for features in feature_maps:
    count += 1
    root = os.path.join('Maps',dataset_name)
    output_path = root+'/'+img_type +'/'
    if not os.path.exists(root):
      os.mkdir(root)
    if not os.path.exists(output_path):
      os.mkdir(output_path)
    # print(feature_map.shape)
    map = torch.mean(feature_map,1)
    map=(map-torch.min(map))/(torch.max(map)-torch.min(map))
    map = np.squeeze((map * 255).cpu().numpy())
    img_save(map, name, output_path)

with torch.no_grad():
    mse = nn.MSELoss(reduction='none')
    mae = nn.L1Loss(reduction='none')
    time_list=[]
    for img_name in os.listdir(os.path.join(test_folder,"ir")):
        print(img_name)
        data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
        data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
        start = time.time()
        data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        feature_B_V_ori, feature_D_V_ori, feature_V = Encoder(data_VIS)
        feature_B_I_ori, feature_D_I_ori, feature_I = Encoder(data_IR)
        # visual
        feature_B_V = BaseFuseLayer(feature_B_V_ori)
        feature_D_V = DetailFuseLayer(feature_D_V_ori)
        # infrared
        feature_B_I = BaseFuseLayer(feature_B_I_ori)
        feature_D_I = DetailFuseLayer(feature_D_I_ori)
        data_v, _ = Decoder(None, feature_B_V, feature_D_V)
        data_i, _ = Decoder(None, feature_B_I, feature_D_I)
        loss_r_r = mse(data_i,data_IR)
        loss_v_v = mse(data_v,data_VIS)
        w_i = torch.exp(-loss_r_r*50)
        w_v = torch.exp(-loss_v_v*50)
        
        softmax = nn.Softmax(0)
        w = torch.stack([w_i,w_v])
        w = softmax(w)
        w_i = 2*w[0]
        w_v = 2*w[1]
        vision_features(w_i, 'w_i',img_name)
        vision_features(w_v, 'w_v',img_name)
        feature_B = torch.mul(w_i,feature_B_I_ori) + torch.mul(w_v,feature_B_V_ori)
        feature_D = torch.mul(w_i,feature_D_I_ori) + torch.mul(w_v,feature_D_V_ori)
        feature_F_B = BaseFuseLayer(feature_B)
        feature_F_D = DetailFuseLayer(feature_D)
        data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
        data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
        fi = np.squeeze((data_Fuse * 255).cpu().numpy())
        end = time.time()
        time_list.append(end - start)
        img_save(fi, img_name, test_out_folder)

    print('mean times is {}'.format(sum(time_list) / len(time_list)))
               
eval_folder=test_out_folder  
ori_img_folder=test_folder
print("="*80)