import torch
from utils import DataLoader
from main import Stegano_Network
import numpy as np
from MS_SSIM import SSIM, PSNR, MSSSIM
import os
from torchvision.utils import save_image

if __name__ == '__main__':
    GPU = '4'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 1

    models = Stegano_Network().to(device)

    checkpoint_encoder = torch.load('./clpstnet.pth')

    models.load_state_dict(checkpoint_encoder['model_dict'])

    # load_model('./'+'nocl_dis.pth',opt,val_model)

    data = DataLoader("../VOC2012/VOC2012/voc_test", limit=np.inf, shuffle=False, batch_size=batch_size)

    j = 1
    models.eval()
    with torch.no_grad():
        for batch_i, (inputs, _) in enumerate(data):
            inputs = inputs.to(device)
            size = inputs.size()
            carrier = torch.empty(int(batch_size), size[1], size[2], size[3])
            secret = torch.empty(int(batch_size), 6, size[2], size[3])
            concatenated_input = torch.empty(int(batch_size), size[1] + 6, size[2], size[3])
            for i in range(len(inputs)):
                carrier[i] = inputs[i]
                secret[i][0] = torch.zeros(size[2], size[3]).random_(0, 2)
                secret[i][1] = torch.zeros(size[2], size[3]).random_(0, 2)
                secret[i][2] = torch.zeros(size[2], size[3]).random_(0, 2)
                secret[i][3] = torch.zeros(size[2], size[3]).random_(0, 2)
                secret[i][4] = torch.zeros(size[2], size[3]).random_(0, 2)
                secret[i][5] = torch.zeros(size[2], size[3]).random_(0, 2)
                concatenated_input[i] = torch.cat((carrier[i], secret[i]), 0)
            concatenated_input = concatenated_input.to(device)
            carrier = carrier.to(device)
            secret = secret.to(device)
            Stego_image, revealed_message = models.forward(concatenated_input, secret, carrier)

            save_path = './d6/'
            filename = save_path + str(j) + '.jpg'
            # filenam2 = save_path + str(j) + 'carrier'+'.jpg'

            save_image(Stego_image, filename)
            # save_image(carrier, filenam2)
            print(str(j), "save")

            j = j + 1
