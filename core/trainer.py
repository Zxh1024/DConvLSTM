import os.path
import datetime
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from core.utils import preprocess, metrics
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square
from scipy import stats

train_loss = []
val_loss = []

mse = []
rmse = []
mae = []
r2 = []
relevant = []
ssim = []
psnr = []


def train(model, ims, real_input_flag, configs):
    train_loss = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        train_loss += model.train(ims_rev, real_input_flag)
        train_loss = train_loss / 2
    return train_loss


def valid(model, valid_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'valid...')

    valid_input_handle.begin(do_shuffle=False)

    batch_id = 0
    ssim_seq, psnr_seq = [], []
    pre = []
    org = []

    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while not valid_input_handle.no_batch_left():

        batch_id = batch_id + 1
        test_ims = valid_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen, cost = model.test(test_dat, real_input_flag)
        val_loss.append(cost)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]

        for i in range(output_length):

            x = test_ims[:, i + configs.input_length, :, :, 0]

            gx = img_out[:, i, :, :, 0]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            gx = gx * (configs.MaxValue - configs.MinValue) + configs.MinValue
            x = x * (configs.MaxValue - configs.MinValue) + configs.MinValue

            org.append(x)
            pre.append(gx)

            for b in range(configs.batch_size):
                score, _ = SSIM(x[b], gx[b], full=True, multichannel=True)
                ssim_seq.append(score)

        print("test_batch_id:", batch_id)
        valid_input_handle.next()

    pre = np.array(pre).flatten()
    org = np.array(org).flatten()
    r, p = stats.pearsonr(org, pre)
    print("MSE:", mean_squared_error(org, pre))
    print("RMSE: ", np.sqrt(mean_squared_error(org, pre)))
    print("MAE：", mean_absolute_error(org, pre))
    print("R2：", r2_score(org, pre))
    print("相关系数：", r)
    print("ssim: ", np.mean(ssim_seq))

    return sum(val_loss) / len(val_loss)


def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    index = 0
    test_input_handle.begin(do_shuffle=False)

    batch_id = 0
    ssim_seq, psnr_seq = [], []
    pre = []
    org = []

    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while (test_input_handle.no_batch_left() == False):

        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen, cost = model.test(test_dat, real_input_flag)
        val_loss.append(cost)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]

        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, 0]
            gx = img_out[:, i, :, :, 0]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            gx = gx * (configs.MaxValue - configs.MinValue) + configs.MinValue
            x = x * (configs.MaxValue - configs.MinValue) + configs.MinValue
            org.append(x)
            pre.append(gx)
            for b in range(configs.batch_size):
                index += 1
                from PIL import Image
                Image.fromarray(gx[b]).save(configs.gx_out_file + '/%d.tif' % index)
                Image.fromarray(x[b]).save(configs.x_out_file + '/%d.tif' % index)

        print("test_batch_id:", batch_id)
        test_input_handle.next()

    pre2 = pre
    org2 = org

    pre_li = []
    org_li = []
    mae_li = []
    rmse_li = []
    mse_li = []
    relevant_li = []

    r2_li = []
    x = np.array(pre2).shape[0] * np.array(pre2).shape[1]
    y = np.array(pre2).shape[2] * np.array(pre2).shape[3]
    pre2 = np.array(pre2).reshape(x, y)
    org2 = np.array(org2).reshape(x, y)
    for i in range(y):
        pre_li.append(pre2.take(i, 1))
        org_li.append(org2.take(i, 1))
    for i in range(y):
        r, p = stats.pearsonr(org_li[i], pre_li[i])
        mae_li.append(mean_absolute_error(org_li[i], pre_li[i]))
        rmse_li.append(np.sqrt(mean_squared_error(org_li[i], pre_li[i])))
        r2_li.append(r2_score(org_li[i], pre_li[i]))
        mse_li.append(mean_squared_error(org_li[i], pre_li[i]))
        relevant_li.append(r)
    mae_li = np.array(mae_li).reshape(19, 19)
    rmse_li = np.array(rmse_li).reshape(19, 19)
    r2_li = np.array(r2_li).reshape(19, 19)
    mse_li = np.array(mse_li).reshape(19, 19)
    relevant_li = np.array(relevant_li).reshape(19, 19)
    from PIL import Image
    Image.fromarray(mae_li).save(configs.gen_frm_dir + '/%d_mae.tif' % itr)
    Image.fromarray(rmse_li).save(configs.gen_frm_dir + '/%d_rmse.tif' % itr)
    Image.fromarray(r2_li).save(configs.gen_frm_dir + '/%d_r2.tif' % itr)
    Image.fromarray(mse_li).save(configs.gen_frm_dir + '/%d_mse.tif' % itr)
    Image.fromarray(relevant_li).save(configs.gen_frm_dir + '/%d_relevant.tif' % itr)

    pre = np.array(pre).flatten()
    org = np.array(org).flatten()
    r, p = stats.pearsonr(org, pre)
    print("MSE:", mean_squared_error(org, pre))
    print("RMSE: ", np.sqrt(mean_squared_error(org, pre)))
    print("MAE：", mean_absolute_error(org, pre))
    print("R2：", r2_score(org, pre))
    print("相关系数：", r)
    print("ssim: ", np.mean(ssim_seq))

    filename = configs.gen_frm_dir + '/test_result.txt'
    with open(filename, 'a') as f:
        f.write(str(itr) + "\n")
        f.write(str('MSE: ' + str(mean_squared_error(org, pre))) + "\n")
        f.write(str('RMSE: ' + str(np.sqrt(mean_squared_error(org, pre)))) + "\n")
        f.write(str('MAE：' + str(mean_absolute_error(org, pre))) + '\n')
        f.write(str('R2：' + str(r2_score(org, pre))) + '\n')
        f.write(str('相关系数：' + str(r)) + '\n')
    if itr % configs.max_iterations == 0:
        with open(filename, 'a') as f:
            f.write(str(configs) + "\n")
