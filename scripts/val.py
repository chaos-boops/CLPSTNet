# -*- coding = utf-8 -*-
# @Time:2023/4/9 11:59
# @Author : ZHANG TONG
# @File:val.py
# @Software:PyCharm

import torch
from main import Stegano_Network
from utils import DataLoader
import numpy as np
from MS_SSIM import SSIM, PSNR, MSSSIM, rmse

ssim = SSIM()
msssim = MSSSIM()
import xlsxwriter
import os
from torchvision.utils import save_image

if __name__ == '__main__':
    def load_model(save_name, optimizer, model):
        model_data = torch.load(save_name)
        model.load_state_dict(model_data['model_dict'])
        optimizer.load_state_dict(model_data['optimizer_dict'])
        print("model load success")


    GPU = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('npu:0')
    # torch.npu.set_device(device)
    print(device)

    batch_size = 8

    models = Stegano_Network().to(device)

    checkpoint_encoder = torch.load('./pt/d2/clstnet_nocl.pth')

    models.load_state_dict(checkpoint_encoder['model_dict'])



    val_data = DataLoader("../val", limit=np.inf, shuffle=True, batch_size=batch_size)
    # val_data=DataLoader("./pt/d3/test/easy", limit=np.inf, shuffle=True, batch_size=batch_size)
    # val_data=DataLoader("./pt/d2/test/medium", limit=np.inf, shuffle=True, batch_size=batch_size)
    # val_data=DataLoader("./pt/d2/test/hard", limit=np.inf, shuffle=True, batch_size=batch_size)

    # val_data=DataLoader("./imagenet_cltrain/test/easy", limit=np.inf, shuffle=True, batch_size=batch_size)
    # val_data=DataLoader("./imagenet_cltrain/test/medium", limit=np.inf, shuffle=True, batch_size=batch_size)
    # val_data=DataLoader("./imagenet_cltrain/test/hard", limit=np.inf, shuffle=True, batch_size=batch_size)

    # val_data=DataLoader("./lfw_test", limit=np.inf, shuffle=True, batch_size=batch_size)
    # workbook = xlsxwriter.Workbook('./result_imagenet/stage4.xlsx', {'nan_inf_to_errors': True})
    # worksheet = workbook.add_worksheet('cl')


    # worksheet.write(0, 0, "ssim")
    # worksheet.write(0, 1, "psnr")
    # worksheet.write(0, 2, "msssim")

    j = 1
    models.eval()
    ssim_score = []
    psnr_score = []
    msssim_score = []
    dis_score = []
    rmse2 = []
    with torch.no_grad():
        for batch_i, (inputs, _) in enumerate(val_data):
            inputs = inputs.to(device)
            size = inputs.size()
            carrier = torch.empty(int(batch_size), size[1], size[2], size[3])
            # secret information setting D=2
            secret = torch.empty(int(batch_size), 2, size[2], size[3])
            concatenated_input = torch.empty(int(batch_size), size[1] + 2, size[2], size[3])
            for i in range(len(inputs)):
                carrier[i] = inputs[i]
                secret[i][0] = torch.zeros(size[2], size[3]).random_(0, 2)
                secret[i][1] = torch.zeros(size[2], size[3]).random_(0, 2)
                # secret[i][2] = torch.zeros(size[2], size[3]).random_(0, 2)
                # secret[i][3] = torch.zeros(size[2], size[3]).random_(0, 2)
                # secret[i][4] = torch.zeros(size[2], size[3]).random_(0, 2)
                # secret[i][5] = torch.zeros(size[2], size[3]).random_(0, 2)
                # secret[i][6] = torch.zeros(size[2], size[3]).random_(0, 2)
                # secret[i][7] = torch.zeros(size[2], size[3]).random_(0, 2)
                concatenated_input[i] = torch.cat((carrier[i], secret[i]), 0)

            concatenated_input = concatenated_input.to(device)
            carrier = carrier.to(device)
            secret = secret.to(device)

            # Stego_image, revealed_message = val_model.forward(concatenated_input)
            # Stego_image, revealed_message, discrminator_result = val_model.forward(concatenated_input)

            Stego_image, revealed_message = models.forward(concatenated_input, secret, carrier)

            ssim1 = ssim(carrier, Stego_image)
            # worksheet.write(j, 0, ssim1)
            ssim_score.append(ssim1.item())

            psnr1 = PSNR(carrier, Stego_image)
            # worksheet.write(j, 1, psnr1)
            psnr_score.append(psnr1.item())

            msssim1 = msssim(carrier, Stego_image)
            # worksheet.write(j, 2, msssim1)
            msssim_score.append(msssim1.item())

            rmse1 = rmse(carrier, Stego_image)
            rmse2.append(rmse1.item())

            """save_path = './result_imagenet/cl/'
            filename = save_path + str(j) +  '.jpg'
            filename2 = save_path + str(j) + '_carrier' + '.jpg'
            save_image(Stego_image, filename)
            save_image(carrier, filename2)"""

            # dis_score.append(discriminator_result.cpu())
            j = j + 1

    ssim_score = np.mean(ssim_score)
    print(ssim_score)
    # worksheet.write(j+1,0,ssim_score)

    psnr_score = np.mean(psnr_score)
    print(psnr_score)
    # worksheet.write(j+1,1,psnr_score)

    msssim_score = np.mean(msssim_score)
    print(msssim_score)
    # worksheet.write(j + 1, 2, msssim_score)
    # workbook.close()

    rmse2 = np.mean(rmse2)
    print(rmse2)

    """analyze_loss = torch.mean(torch.stack(dis_score))
    print(analyze_loss)"""

