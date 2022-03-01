import os.path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import torch
import codecs
import lpips


def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    if itr % configs.display_interval == 0:
        print('itr: ' + str(itr),
              'training L1 loss: ' + str(loss_l1), 'training L2 loss: ' + str(loss_l2))


def test(model, test_input_handle, configs, itr):
    print('test...')
    loss_fn = lpips.LPIPS(net='alex', spatial=True).to(configs.device)
    res_path = configs.gen_frm_dir + '/' + str(itr)

    if not os.path.exists(res_path):
        os.mkdir(res_path)
    f = codecs.open(res_path + '/performance.txt', 'w+')
    f.truncate()

    avg_mse = 0
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    batch_id = 0
    img_mse, img_mae, img_psnr, ssim, img_lpips, mse_list, mae_list, psnr_list, ssim_list, lpips_list = [], [], [], [], [], [], [], [], [], []
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        img_mae.append(0)
        img_psnr.append(0)
        ssim.append(0)
        img_lpips.append(0)

        mse_list.append(0)
        mae_list.append(0)
        psnr_list.append(0)
        ssim_list.append(0)
        lpips_list.append(0)
    for epoch in range(configs.max_epoches):
        if batch_id > configs.num_save_samples:
            break
        for data in test_input_handle:
            if batch_id > configs.num_save_samples:
                break
            print(batch_id)

            batch_size = data.shape[0]
            real_input_flag = np.zeros(
                (batch_size,
                 configs.total_length - configs.input_length - 1,
                 configs.img_height // configs.patch_size,
                 configs.img_width // configs.patch_size,
                 configs.patch_size ** 2 * configs.img_channel))

            img_gen = model.test(data, real_input_flag)
            img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            output_length = configs.total_length - configs.input_length
            output_length = min(output_length, configs.total_length - 1)
            test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
            img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
            img_out = img_gen[:, -output_length:, :]

            # MSE per frame
            for i in range(output_length):
                x = test_ims[:, i + configs.input_length, :]
                gx = img_out[:, i, :]
                gx = np.maximum(gx, 0)
                gx = np.minimum(gx, 1)
                mse = np.square(x - gx).sum()/batch_size
                mae = np.abs(x - gx).sum()/batch_size
                psnr = 0
                t1 = torch.from_numpy((x - 0.5) / 0.5).to(configs.device)
                t1 = t1.permute((0, 3, 1, 2))
                t2 = torch.from_numpy((gx - 0.5) / 0.5).to(configs.device)
                t2 = t2.permute((0, 3, 1, 2))
                shape = t1.shape
                if not shape[1] == 3:
                    new_shape = (shape[0], 3, *shape[2:])
                    t1.expand(new_shape)
                    t2.expand(new_shape)
                d = loss_fn.forward(t1, t2)
                lpips_score = d.mean()
                lpips_score = lpips_score.detach().cpu().numpy() * 100
                for sample_id in range(batch_size):
                    mse_tmp = np.square(
                        x[sample_id, :] - gx[sample_id, :]).mean()
                    psnr += 10 * np.log10(1 / mse_tmp)
                psnr /= (batch_size)
                img_mse[i] += mse
                img_mae[i] += mae
                img_psnr[i] += psnr
                img_lpips[i] += lpips_score
                mse_list[i] = mse
                mae_list[i] = mae
                psnr_list[i] = psnr
                lpips_list[i] = lpips_score
                avg_mse += mse
                avg_mae += mae
                avg_psnr += psnr
                avg_lpips += lpips_score
                score = 0
                for b in range(batch_size):
                    score += compare_ssim(x[b, :], gx[b, :], multichannel=True)
                score /= batch_size
                ssim[i] += score
                ssim_list = score
                avg_ssim += score
            f.writelines(str(batch_id) + ',' + str(psnr_list) + ',' + str(mse_list) + ',' +str(mae_list) + ','+ str(lpips_list) + ',' + str(
                ssim_list) + '\n')
            res_width = configs.img_width
            res_height = configs.img_height
            img = np.ones((2 * res_height,
                           configs.total_length * res_width,
                           configs.img_channel))
            name = str(batch_id) + '.png'
            file_name = os.path.join(res_path, name)
            for i in range(configs.total_length):
                img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :]
            for i in range(output_length):
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,
                :] = img_out[0, -output_length + i, :]
            img = np.maximum(img, 0)
            img = np.minimum(img, 1)
            cv2.imwrite(file_name, (img * 255).astype(np.uint8))
            batch_id = batch_id + 1
    f.close()
    with codecs.open(res_path + '/data.txt', 'w+') as data_write:
        data_write.truncate()

        avg_mse = avg_mse / (batch_id * output_length)
        print('mse per frame: ' + str(avg_mse))
        for i in range(configs.total_length - configs.input_length):
            print(img_mse[i] / batch_id)
            img_mse[i] = img_mse[i] / batch_id
        data_write.writelines(str(avg_mse) + '\n')
        data_write.writelines(str(img_mse) + '\n')

        avg_mae = avg_mae / (batch_id * output_length)
        print('mae per frame: ' + str(avg_mae))
        for i in range(configs.total_length - configs.input_length):
            print(img_mae[i] / batch_id)
            img_mae[i] = img_mae[i] / batch_id
        data_write.writelines(str(avg_mae) + '\n')
        data_write.writelines(str(img_mae) + '\n')

        avg_psnr = avg_psnr / (batch_id * output_length)
        print('psnr per frame: ' + str(avg_psnr))
        for i in range(configs.total_length - configs.input_length):
            print(img_psnr[i] / batch_id)
            img_psnr[i] = img_psnr[i] / batch_id
        data_write.writelines(str(avg_psnr) + '\n')
        data_write.writelines(str(img_psnr) + '\n')

        avg_ssim = avg_ssim / (batch_id * output_length)
        print('ssim per frame: ' + str(avg_ssim))
        for i in range(configs.total_length - configs.input_length):
            print(ssim[i] / batch_id)
            ssim[i] = ssim[i] / batch_id
        data_write.writelines(str(avg_ssim) + '\n')
        data_write.writelines(str(ssim) + '\n')

        avg_lpips = avg_lpips / (batch_id * output_length)
        print('lpips per frame: ' + str(avg_lpips))
        for i in range(configs.total_length - configs.input_length):
            print(img_lpips[i] / batch_id)
            img_lpips[i] = img_lpips[i] / batch_id
        data_write.writelines(str(avg_lpips) + '\n')
        data_write.writelines(str(img_lpips) + '\n')
