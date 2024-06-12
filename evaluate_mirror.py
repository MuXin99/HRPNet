import os
import time

import numpy as np
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader

from toolbox import get_dataset, setup_seed
from toolbox import get_model
from toolbox import class_to_RGB, load_ckpt, save_ckpt
from toolbox.datasets.MSD import MSD

from toolbox.datasets.mirrorrgbd import MirrorRGBD
from toolbox.msg import runMsg
import matplotlib.pyplot as plt

setup_seed(33)


def evaluate(logdir, save_predict=False, options=['test'], prefix=''):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')

    loaders = []
    for opt in options:
        dataset = MirrorRGBD(cfg, mode=opt)
        # dataset = MSD(cfg, mode=opt)
        # dataset = PST900(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))

    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(logdir, prefix + 'xxxmodel.pth'), map_location={'cuda:0': 'cuda:0'}))
    # model = load_ckpt(logdir, model, prefix=prefix)

    running_metrics_val = runMsg()
    # time_meter = averageMeter()

    save_path = os.path.join(logdir, 'xxxpredicts/')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    for name, test_loader in loaders:
        # running_metrics_val.reset()
        print('#' * 50 + '    ' + name + prefix + '    ' + '#' * 50)
        model.eval()
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            # time_start = time.time()
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                with torch.no_grad():
                    predict = model(image)
            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                with torch.no_grad():
                    predict = model.forward(image, depth)[0]

            predict = predict.cpu().float()  # [1, h, w]
            label = torch.unsqueeze(label, 1).cpu().float()

            # predict = torch.sigmoid(predict-predict*label+label)

            # label = label.cpu().float()
            running_metrics_val.update(label, predict)
            # if i == 0:
            #     total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
            #     print('Total network parameters: ' + str(total_paramters / 1e6) + "M")
            # time_meter.update(time.time() - time_start, n=image.size(0))
            if save_predict:
                predict = predict.numpy()[0][0]
                # predict = torch.sigmoid(predict+label).numpy()[0][0]
                plt.imsave(save_path + sample['label_path'][0], arr=predict, cmap='gray')
                # for j in range(predict.shape[0]):
                #     predict1 = predict.numpy()[j]
                #     predict1 = predict1[0]
                #     plt.imsave(save_path + sample['label_path'][j], arr=predict1, cmap='gray')

        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        iou = metrics["iou: "].item()
        ber = metrics["ber: "].item() * 100
        mae = metrics["mae: "].item()
        F_beta = metrics["F_measure: "].item()
        # PA = metrics["Acc: "].item()
        print('iou:', iou, 'ber:', ber, 'mae:', mae, 'F_beta:', F_beta)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str,
                        # default="run/2023-09-08-23-47(mirrorRGBD-worker4_s-知识蒸馏_B3-B0_消融gcn)",
                        default="/media/user/shuju/Mirror_jzj/Mirror_ss/Work1/weight/PVTNet_T2/",
                        help="run logdir")
    parser.add_argument("-s", type=bool, default=True, help="save predict or not")
    args = parser.parse_args()
    evaluate(args.logdir, save_predict=args.s, options=['test'], prefix='')
