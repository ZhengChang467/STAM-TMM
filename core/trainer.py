import os.path
import datetime
import cv2
import numpy as np
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import torch
import codecs
import lpips
# from core.models.classification_model import Classification


def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    if configs.reverse_input:
        ims_rev = torch.flip(ims, dims=[1])
        _, loss_l1_rev, loss_l2_rev = model.train(
            ims_rev, real_input_flag, itr)
        loss_l1 += loss_l1_rev
        loss_l2 += loss_l2_rev
        loss_l1 /= 2
        loss_l2 /= 2

    if itr % configs.display_interval == 0:
        print('itr: ' + str(itr), 'training L1 loss: ' +
              str(loss_l1), 'training L2 loss: ' + str(loss_l2))


def visualization_gates(gates, configs, batch_id):
    # tmp = gates[3][5][5].detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
    # print(tmp.shape)
    gates_name = ['I_t', 'F_t', 'G_t', 'i_t', 'f_t', 'g_t', 'O_t']
    for gate_id in range(7):
        res = np.ones(shape=(configs.img_height*configs.num_layers, configs.img_width*(configs.total_length-1)))*255.
        for layer in range(configs.num_layers):
            for time_step in range(configs.total_length-1):
                gate_data = gates[layer][time_step][gate_id].detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
                gate_data = preprocess.reshape_patch_back(gate_data, configs.patch_size).mean(axis=1).mean(axis=3)[0]
                if gate_id==2 or gate_id==5:
                    gate_data = gate_data*0.5+0.5
                gate_data *= 255.
                gate_data = np.maximum(gate_data, 0)
                gate_data = np.minimum(gate_data, 255.)
                res[layer*configs.img_height:(layer+1)*configs.img_height, time_step*configs.img_width:(time_step+1)*configs.img_width]=gate_data
        cv2.imwrite(configs.gen_frm_dir+'1/gates/'+gates_name[gate_id]+'_'+str(batch_id)+'.png', res)

    # tmp=preprocess.reshape_patch_back(tmp, configs.patch_size).mean(axis=1).mean(axis=3)[0]*255.
    # tmp = np.maximum(tmp, 0)
    # tmp = np.minimum(tmp, 255.)
    # cv2.imwrite('gate.png', tmp)


def test(model, test_input_handle, configs, itr):
    print('test...')
    if not os.path.exists(configs.gen_frm_dir + str(itr) + '/'):
        os.makedirs(configs.gen_frm_dir + str(itr) + '/')
        # os.makedirs(configs.gen_frm_dir + str(itr) + '/gates/')
    os.makedirs(configs.gen_frm_dir + str(itr) + '/gates/', exist_ok=True)

    loss_fn = lpips.LPIPS(net='alex', spatial=True).cuda()
    performance_dir = configs.perforamnce_dir + str(itr)
    os.makedirs(performance_dir, exist_ok=True)
    f = codecs.open(performance_dir + '/performance.txt', 'w+')
    f.truncate()
    res_path = configs.gen_frm_dir + '/' + str(itr)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse = 0
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    batch_id = 0
    # flow2img = F2I()
    img_mse, img_mae, img_psnr, ssim, img_lpips, mse_list, mae_list, psnr_list, ssim_list, lpips_list = [
    ], [], [], [], [], [], [], [], [], []
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
            
            if configs.is_vis_gates:
                img_gen, gates_visual = model.test(data, real_input_flag)
                visualization_gates(gates_visual, configs, batch_id)
            else:
                img_gen, _ = model.test(data, real_input_flag)
            # print(data.shape)
            
            img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
            output_length = configs.total_length - configs.input_length
            output_length = min(output_length, configs.total_length - 1)
            test_ims = preprocess.reshape_patch_back(
                test_ims, configs.patch_size)
            img_gen = preprocess.reshape_patch_back(
                img_gen, configs.patch_size)
            img_out = img_gen[:, -output_length:, :]
            # if img_out
            # tmp = np.ones(img_out.shape)
            # tmp[:, :, :, :, :2] = img_out
            # img_out = tmp

            # MSE per frame
            for i in range(output_length):
                x = test_ims[:, i + configs.input_length, :]
                gx = img_out[:, i, :]
                gx = np.maximum(gx, 0)
                gx = np.minimum(gx, 1)
                # gx = gx * 0.5 + 0.5
                if configs.dataset == 'mnist' or configs.dataset == 'taxibj':
                    mse = np.square(x - gx).sum() / batch_size
                else:
                    mse = np.square(x - gx).mean()
                mae = np.abs(x - gx).mean()
                psnr = 0
                t1 = torch.from_numpy((x - 0.5) / 0.5).cuda()
                t1 = t1.permute((0, 3, 1, 2))
                t2 = torch.from_numpy((gx - 0.5) / 0.5).cuda()
                t2 = t2.permute((0, 3, 1, 2))
                shape = t1.shape
                if not shape[1] == 3:
                    new_shape = (shape[0], 3, *shape[2:])
                    t1 = t1[:, :1, :].expand(new_shape)
                    t2 = t2[:, :1, :].expand(new_shape)
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
            # print('lpips:', np.array(img_lpips).mean())
            f.writelines(str(batch_id) + ',' + str(psnr_list) + ',' + str(mse_list) + ',' + str(mae_list) + ',' + str(lpips_list) + ',' + str(
                ssim_list) + '\n')
            res_width = configs.img_width
            res_height = configs.img_height
            img = np.ones((2 * res_height,
                           configs.total_length * res_width,
                           configs.img_channel))
            # print(img.shape)
            # optical_flow = np.ones((3 * res_height,
            #                         output_length * res_width,
            #                         configs.flow_channel))
            name = str(batch_id) + '.png'
            # flow_name = str(batch_id) + '_flow.png'
            file_name = os.path.join(res_path, name)
            # flow_name = os.path.join(res_path, flow_name)
            # print(test_opt.mean())
            for i in range(configs.total_length):
                img[:res_height, i *
                    res_width:(i + 1) * res_width, :] = test_ims[0, i, :]
                # optical_flow[:res_height, i * res_width:(i + 1) * res_width, :] = test_flow[0, i, :]
            for i in range(output_length):
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,
                    :] = img_out[0, -output_length + i, :]

            img = np.maximum(img, 0)
            img = np.minimum(img, 1)
            # print(img.shape)
            if configs.img_channel == 2:
                # print('ok')

                # print(img_in_raw.mean())
                img_in = img[:, :, 0] * 255.
                img_out = img[:, :, 1] * 255.
                in_diff = np.abs(img_in[configs.img_height:, configs.input_length * configs.img_width:] - img_in[:configs.img_height,
                                                                                                                 configs.input_length * configs.img_width:])
                out_diff = np.abs(img_out[configs.img_height:, configs.input_length * configs.img_width:] - img_out[
                    :configs.img_height,
                    configs.input_length * configs.img_width:])

                im_color_in = cv2.applyColorMap(
                    np.uint8(img_in), cv2.COLORMAP_JET)
                im_color_out = cv2.applyColorMap(
                    np.uint8(img_out), cv2.COLORMAP_JET)
                cv2.imwrite(res_path + '/' + str(batch_id) +
                            '_in.png', im_color_in)
                cv2.imwrite(res_path + '/' + str(batch_id) +
                            '_out.png', im_color_out)
                diff_color_in = cv2.applyColorMap(
                    np.uint8(in_diff), cv2.COLORMAP_JET)
                diff_color_out = cv2.applyColorMap(
                    np.uint8(out_diff), cv2.COLORMAP_JET)
                cv2.imwrite(res_path + '/' + str(batch_id) +
                            '_in_diff.png', diff_color_in)
                cv2.imwrite(res_path + '/' + str(batch_id) +
                            '_out_diff.png', diff_color_out)
            else:
                cv2.imwrite(file_name, (img * 255).astype(np.uint8))
            batch_id = batch_id + 1
    f.close()
    with codecs.open(configs.gen_frm_dir + str(itr) + '/data.txt', 'w+') as data_write:
        data_write.truncate()
        avg_mse = avg_mse / (batch_id * output_length)
        print('mse per seq: ' + str(avg_mse))
        for i in range(configs.total_length - configs.input_length):
            print(img_mse[i] / batch_id)
            img_mse[i] = img_mse[i] / batch_id
        data_write.writelines(str(avg_mse) + '\n')
        data_write.writelines(str(img_mse) + '\n')

        avg_mae = avg_mae / (batch_id * output_length)
        print('mae per seq: ' + str(avg_mae))
        for i in range(configs.total_length - configs.input_length):
            print(img_mae[i] / batch_id)
            img_mae[i] = img_mae[i] / batch_id
        data_write.writelines(str(avg_mae) + '\n')
        data_write.writelines(str(img_mae) + '\n')

        avg_psnr = avg_psnr / (batch_id * output_length)
        print('psnr per seq: ' + str(avg_psnr))
        for i in range(configs.total_length - configs.input_length):
            print(img_psnr[i] / batch_id)
            img_psnr[i] = img_psnr[i] / batch_id
        data_write.writelines(str(avg_psnr) + '\n')
        data_write.writelines(str(img_psnr) + '\n')

        avg_ssim = avg_ssim / (batch_id * output_length)
        print('ssim per seq: ' + str(avg_ssim))
        for i in range(configs.total_length - configs.input_length):
            print(ssim[i] / batch_id)
            ssim[i] = ssim[i] / batch_id
        data_write.writelines(str(avg_ssim) + '\n')
        data_write.writelines(str(ssim) + '\n')

        avg_lpips = avg_lpips / (batch_id * output_length)
        print('lpips per seq: ' + str(avg_lpips))
        for i in range(configs.total_length - configs.input_length):
            print(img_lpips[i] / batch_id)
            img_lpips[i] = img_lpips[i] / batch_id
        data_write.writelines(str(avg_lpips) + '\n')
        data_write.writelines(str(img_lpips) + '\n')


def test_classify(model, test_input_handle, configs, itr):
    print('test...')
    model_classify = Classification(16, 128).cuda()
    model_classify.eval()
    stats = torch.load('F:/classification/checkpoints/model.ckpt-19')
    model_classify.load_state_dict(stats['net_param'])

    batch_id = 0
    test_0_acc = 0
    test_1_acc = 0
    gen_0_acc = 0
    gen_1_acc = 0
    total_acc = 0
    # while (batch_id <= configs.num_save_samples):
    # for epoch in range(configs.max_epoches):
    #     if batch_id > configs.num_save_samples:
    #         break
    for data in test_input_handle:
        # print(test_ims.shape)
        # if batch_id > 9:
        #     break
        print(batch_id)
        test_ims = data['data']
        label = data['label'].cuda()
        batch_size = test_ims.shape[0]
        real_input_flag = np.zeros(
            (batch_size,
             configs.total_length - configs.input_length - 1,
             configs.img_height // configs.patch_size,
             configs.img_width // configs.patch_size,
             configs.patch_size ** 2 * configs.img_channel))

        img_gen = model.test(
            test_ims, real_input_flag).transpose(0, 1, 3, 4, 2)
        test_ims = test_ims.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)
        output_length = configs.total_length - configs.input_length
        output_length = min(output_length, configs.total_length - 1)
        test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        test = []
        gen = []
        for i in range(img_gen.shape[0]):
            gen.append(preprocess.reshape_patch(img_gen[i], 4))
            test.append(preprocess.reshape_patch(test_ims[i], 4))
        img_gen = np.stack(gen, axis=0)
        test_ims = np.stack(test, axis=0)
        img_gen = torch.FloatTensor(img_gen).cuda()
        test_ims = torch.FloatTensor(test_ims).cuda()

        test_out_0 = test_ims[:, -40:-20, :].permute(0, 4, 1, 2, 3)
        test_out_1 = test_ims[:, -20:, :].permute(0, 4, 1, 2, 3)
        gen_out_0 = img_gen[:, -40:-20, :].permute(0, 4, 1, 2, 3)
        gen_out_1 = img_gen[:, -20:, :].permute(0, 4, 1, 2, 3)
        _, classify_gt_0 = torch.max(model_classify(test_out_0), dim=1)
        _, classify_gt_1 = torch.max(model_classify(test_out_1), dim=1)
        _, classify_gen_0 = torch.max(model_classify(gen_out_0), dim=1)
        _, classify_gen_1 = torch.max(model_classify(gen_out_1), dim=1)
        # classify_gt_0 = classify_gt_0.reshape(-1)
        # classify_gt_1 = classify_gt_1.reshape(-1)
        # classify_gen_0 = classify_gen_0.reshape(-1)
        # classify_gen_1 = classify_gen_1.reshape(-1)
        label = label.reshape(-1)
        test_0_acc += (classify_gt_0 == label).sum()
        test_1_acc += (classify_gt_1 == label).sum()
        total_acc += (label == label).sum()
        gen_0_acc += (classify_gen_0 == label).sum()
        gen_1_acc += (classify_gen_1 == label).sum()
        batch_id += 1
    acc_gt_0 = test_0_acc.float() / total_acc.float()
    acc_gt_1 = test_1_acc.float() / total_acc.float()
    acc_gen_0 = gen_0_acc.float() / total_acc.float()
    acc_gen_1 = gen_1_acc.float() / total_acc.float()

    print(total_acc, acc_gt_0, acc_gt_1, acc_gen_0, acc_gen_1)
    # print(classify_gt_0, label)
