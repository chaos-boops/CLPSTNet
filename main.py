# -*- coding = utf-8 -*-
# @Time:2022/10/25 17:33
# @Author : ZHANGTONG
# @File:main.py
# @Software:PyCharm

import os
import torch
from torch import nn
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import random
from critic import XuNet
from torchvision import transforms
import torchvision
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import ImageFolder,DataLoader
from clpstnet import Encoder_Network
from decoder import decode_Network
from MS_SSIM import SSIM, MSSSIM, PSNR, rmse

msssim = MSSSIM()
ssim = SSIM()


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epoch of train')
parser.add_argument('--decay_epoch', type=int, default=15, help='start decaying learning rate after this number')
parser.add_argument('--lambdaA', type=float, default=1, help='lambdaA for encoder')
parser.add_argument('--lambdaB', type=float, default=0.5, help='lambdaB for decoder')
parser.add_argument('--lambdaC', type=float, default=0.5, help='lambdaC for discriminator')
parser.add_argument('--lambdaD', type=float, default=0.5, help='lambdaD for encoder ssim')
parser.add_argument('--lambdaE', type=float, default=0.3, help='lambdaE for encoder mse')
parser.add_argument('--lambdaF', type=float, default=0.75, help='lambdaF for decoder_weight')
params = parser.parse_args()
mytransforms = transforms.Compose([transforms.ToTensor()])

class Stegano_Network(nn.Module):
    def __init__(self):
        super(Stegano_Network, self).__init__()
        self.hidden = Encoder_Network()
        self.reveal = decode_Network()

    def forward(self, x, secret, cover):
        Stego = self.hidden(x, secret, cover)
        revealed_message = self.reveal(Stego)
        return Stego, revealed_message


class EncoderLoss(nn.Module):
    def __init__(self):
        super(EncoderLoss, self).__init__()

    def forward(self, cover_image, stego_image):
        encoder_mse = torch.nn.functional.mse_loss(cover_image, stego_image)
        encoder_ssim = ssim(stego_image, cover_image)
        encoder_msssim = msssim(cover_image, stego_image)
        loss = 0.5 * (1 - encoder_ssim) + 0.5 * (1 - encoder_msssim) + 0.3 * encoder_mse
        return loss


class DecoderLoss(nn.Module):
    def __init__(self):
        super(DecoderLoss, self).__init__()

    def forward(self, message, revealed_message):
        decoder_loss = (torch.nn.functional.binary_cross_entropy_with_logits(message, revealed_message))
        return decoder_loss


if __name__ == '__main__':
    GPU = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = 120
    batch_size = 8
    z = 0
    q = 0
    disloss2 = 0
    boost_sample = 0
    w = 0

    """dataset = ask"""
    # stage1_data = DataLoader("../mtrain", limit=np.inf, shuffle=True, batch_size=batch_size)
    # val = DataLoader("../mval", limit=np.inf, shuffle=True, batch_size=batch_size)
    # test = DataLoader("../mtest", limit=np.inf, shuffle=True, batch_size=batch_size)

    """dataset = imagenet"""
    # stage1_data = DataLoader("../imageNet/imageNet/imagenet_train", limit=np.inf, shuffle=True, batch_size=batch_size)
    # val = DataLoader("../imageNet/imageNet/imagenet_val", limit=np.inf, shuffle=True, batch_size=batch_size)
    # test = DataLoader("../imageNet/imageNet/imagenet_test", limit=np.inf, shuffle=True, batch_size=batch_size)

    """dataset = voc2012"""
    stage1_data = DataLoader("../VOC2012/train", limit=np.inf, shuffle=True, batch_size=batch_size)
    val = DataLoader("../VOC2012/val", limit=np.inf, shuffle=True, batch_size=batch_size)
    # test = DataLoader("../VOC2012/VOC2012/test", limit=np.inf, shuffle=True, batch_size=batch_size)

    trainwriter = SummaryWriter('./tensorboard/kneePointTest_tensorboard')

    models = Stegano_Network().to(device)
    encode_network = Encoder_Network().to(device)
    decode_network = decode_Network().to(device)

    discriminator_network = XuNet(kernel_size=3, padding=1).to(device)
    criterion_encoder = EncoderLoss().to(device)
    criterion_decoder = DecoderLoss().to(device)
    BCE_loss = nn.BCELoss().to(device)


    opt = torch.optim.Adam(models.parameters())
    discriminator_optimizer1 = torch.optim.SGD(discriminator_network.parameters(), lr=params.lr / 3, weight_decay=1e-8)

    def train(epoch):
        model = models.train()
        discriminator = discriminator_network.train()
        print("epoch:", epoch)

        for batch_i, (inputs, _) in enumerate(stage1_data):
            inputs = inputs.to(device)
            size = inputs.size()
            carrier = torch.empty(int(batch_size), size[1], size[2], size[3])
            # secret information D=6
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

            Stego_image, revealed_message = model.forward(concatenated_input, secret, carrier)

            discriminator_result = discriminator(Stego_image)
            disloss1 = BCE_loss(discriminator_result,
                                Variable(
                                    torch.zeros(discriminator_result.size()).to(device) * random.uniform(0.8, 1.2)))
            encoder_loss = criterion_encoder(Stego_image, carrier)
            decoder_loss = criterion_decoder(revealed_message, secret)

            opt.zero_grad()
            loss = encoder_loss + decoder_loss + 0.1 * disloss1
            loss.backward()

            opt.step()


            if batch_i % 10 == 0:
                carrier = carrier.clone().detach().requires_grad_(True)
                Stego_image = Stego_image.clone().detach().requires_grad_(True)

                cover_analyze = discriminator(carrier)
                stego_analyze = discriminator(Stego_image)

                discover = BCE_loss(cover_analyze, torch.zeros(cover_analyze.size()).cuda() * random.uniform(0.8, 1.2))
                distego = BCE_loss(stego_analyze, torch.ones(stego_analyze.size()).cuda() * random.uniform(0.0, 0.2))
                disloss2 = discover + distego

                discriminator_optimizer1.zero_grad()
                disloss2.backward()

                discriminator_optimizer1.step()

            global q
            trainwriter.add_scalar("train_loss_ssim", encoder_loss.item(), q)
            trainwriter.add_scalar("train_loss_decode", decoder_loss.item(), q)
            trainwriter.add_scalar("discriminator_train", disloss1.item(), q)
            q = q + 1

            if batch_i % 100 == 0:
                train_string1 = f"batch : {batch_i}  //batch_loss : {loss.item(): 0.5f}// encoder_loss :{encoder_loss.item(): 0.5f} // decoder_loss : {decoder_loss.item(): 0.5f}// stego_loss : {disloss1.item(): 0.5f}// dicriminator_loss : {disloss2.item(): 0.5f} "
                print(train_string1)
                # ssim_tmp = ssim(Stego_image, carrier)
                # stego_image_copy = Stego_image.detach().cpu()
                # carrier_copy = carrier.cpu()
                # psnr_tmp = PSNR(stego_image_copy, carrier_copy)
                # print("ssim:", ssim_tmp)
                # print("psnr:", psnr_tmp)


    def test(z):
        models.eval()
        discriminator_network.eval()
        mse_encode_loss = []
        ssim_encode_loss = []
        psnr_score = []
        msssim_score = []
        test_decode_loss = []
        analyze_loss = []
        rmse1 = []

        test_loss = 0
        with torch.no_grad():
            for batch_i, (inputs, _) in enumerate(val):
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
                discriminator_result = discriminator_network(Stego_image)
                mse1 = torch.nn.functional.mse_loss(carrier, Stego_image)
                mse_encode_loss.append(mse1.item())

                ssim1 = ssim(carrier, Stego_image)
                ssim_encode_loss.append(ssim1.item())

                decode1 = torch.nn.functional.binary_cross_entropy_with_logits(revealed_message, secret)
                test_decode_loss.append(decode1.item())

                msssim1 = msssim(carrier.cpu(), Stego_image.cpu())
                msssim_score.append(msssim1)

                psnr1 = PSNR(carrier.cpu(), Stego_image.cpu())
                psnr_score.append(psnr1)

                rmse2 = rmse(carrier, Stego_image)
                rmse1.append(rmse2.item())

                analyze_loss.append(discriminator_result.cpu())

                # if (z > 60) and (batch_i % 1000 == 0):
                if batch_i % 1000 == 0:
                    for x in range(batch_size):
                        secret_squ1 = torch.squeeze(secret, dim=0)
                        input_sigmoid = torch.nn.Sigmoid()
                        revealed_1 = input_sigmoid(revealed_message)
                        reavealed_squ_1 = torch.squeeze(revealed_1, dim=0)
                        a = reavealed_squ_1.detach().cpu().numpy().astype(np.int32)
                        predict_true = 0
                        predict_total = 0

                        for i in range(128):
                            for j in range(128):
                                if a[x][0][i][j] == secret_squ1[x][0][i][j]:
                                    predict_true = predict_true + 1
                                predict_total = predict_total + 1

                        for i in range(128):
                            for j in range(128):
                                if a[x][1][i][j] == secret_squ1[x][1][i][j]:
                                    predict_true = predict_true + 1
                                predict_total = predict_total + 1

                        for i in range(128):
                            for j in range(128):
                                if a[x][2][i][j] == secret_squ1[x][2][i][j]:
                                    predict_true = predict_true + 1
                                predict_total = predict_total + 1

                        for i in range(128):
                            for j in range(128):
                                if a[x][3][i][j] == secret_squ1[x][3][i][j]:
                                    predict_true = predict_true + 1
                                predict_total = predict_total + 1
                        for i in range(128):
                            for j in range(128):
                                if a[x][4][i][j] == secret_squ1[x][4][i][j]:
                                    predict_true = predict_true + 1
                                predict_total = predict_total + 1
                        for i in range(128):
                            for j in range(128):
                                if a[x][5][i][j] == secret_squ1[x][5][i][j]:
                                    predict_true = predict_true + 1
                                predict_total = predict_total + 1
                    print(batch_i, predict_true / predict_total)

        mse_encode_loss = np.mean(mse_encode_loss)
        print("mse:", mse_encode_loss)
        ssim_encode_loss = np.mean(ssim_encode_loss)
        print("ssim:", ssim_encode_loss)
        test_decode_loss = np.mean(test_decode_loss)
        print("decode:", test_decode_loss)
        psnr_score = np.mean(psnr_score)
        print("psnr:", psnr_score)
        msssim_score = np.mean(msssim_score)
        print("msssim:", msssim_score)
        analyze_loss = torch.mean(torch.stack(analyze_loss))
        print("discriminator_score:", analyze_loss)
        rmse1 = np.mean(rmse1)
        print("RMSE:", rmse1)

        trainwriter.add_scalar("val_mse", mse_encode_loss, z)
        trainwriter.add_scalar("val_ssim", ssim_encode_loss, z)
        trainwriter.add_scalar("val_decode loss", test_decode_loss, z)
        trainwriter.add_scalar("val_psnr", psnr_score, z)
        trainwriter.add_scalar("val_msssim", msssim_score, z)
        trainwriter.add_scalar("val_discrimintor", analyze_loss, z)
        trainwriter.add_scalar("val_rmse", rmse1, z)


    def save_model(save_path, epoch, optimizer, model):
        torch.save({'epoch': epoch + 1,
                    'optimizer_dict': optimizer.state_dict(),
                    'model_dict': model.state_dict()},
                   save_path)


    for epoch in range(1, epochs + 1):
        train(epoch)
        test(z)
        z = z + 1

        if (epoch < 80) and (epoch % 10 == 0):
            save_model('./chapt2/clstnet_d6.pth', epoch, opt, models)
            save_model('./chapt2/discriminator_d6.pth', epoch, discriminator_optimizer1, discriminator_network)
        if epoch >= 80:
            save_model('./chapt2/clstnet_d6.pth', epoch, opt, models)
            save_model('./chapt2/discriminator_d6.pth', epoch, discriminator_optimizer1, discriminator_network)

    trainwriter.close()