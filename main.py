from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from model import TSegFormer
import numpy as np
from torch.utils.data import DataLoader
from util import *
import sklearn.metrics as metrics
from data import Teeth
import json
import tqdm

seg2colors = {}

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'


def _init_(args):
    if not os.path.exists(args.save_path + '/' + args.exp_name + '/' + 'models'):
        os.makedirs(args.save_path + '/' + args.exp_name + '/' + 'models')

    # set up index2colors
    colors = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for i in range(1, 33):
        seg2colors[i] = colors[i % 3]
    seg2colors[0] = (0, 0, 0)


def train(args, io):
    train_dataset = Teeth(num_points=args.num_points, ROOT_PATH=args.data_path, partition="train")
    if len(train_dataset) < 100:
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True,
                              drop_last=drop_last)
    test_loader = DataLoader(Teeth(num_points=args.num_points, ROOT_PATH=args.data_path, partition="val"),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda:0")
    print("Train dataset length", len(train_loader.dataset))
    print("Val dataset length", len(test_loader.dataset))
    # Try to load models
    seg_num_all = 33
    model = TSegFormer(args, seg_num_all).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda(device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion1 = cal_loss
    criterion2 = mask_cur_loss

    best_test_iou = 0
    best_test_epoch = -1
    for epoch in tqdm.tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0.0
        train_loss_ging = 0.0
        train_loss_seg = 0.0
        train_loss_cur = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        sample_count = 0
        for data, label, seg in train_loader:
            sample_count += label.shape[0]
            label_one_hot = label

            AvgAngle_curvature = data[:, :, 7]
            high_cur_num = int(len(AvgAngle_curvature[0]) * args.cur_threshold)
            AvgAngle_idx = np.argpartition(AvgAngle_curvature, -high_cur_num, axis=1)[:, -high_cur_num:]
            idx_base = np.arange(0, AvgAngle_curvature.size(0)).reshape(-1, 1) * AvgAngle_curvature.size(1)
            AvgAngle_idx = AvgAngle_idx + idx_base
            AvgAngle_idx = AvgAngle_idx.reshape(-1)

            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            AvgAngle_idx = AvgAngle_idx.to(device)
            data = data.permute(0, 2, 1)  # [B, 8, N]
            batch_size = data.size()[0]
            opt.zero_grad()

            seg_pred, ging_pred = model(data, label_one_hot)  # [B, 33, N], [B, 2, N]
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            ging_pred = ging_pred.permute(0, 2, 1).contiguous()
            ging_gold = (seg.view(-1, 1).squeeze() > 0).to(dtype=torch.int64)

            loss_ging = args.w_ging * criterion1(ging_pred.view(-1, 2), ging_gold)
            loss_seg = criterion1(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            loss_cur = args.w_cur * criterion2(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze(), AvgAngle_idx)
            loss = loss_ging + loss_seg + loss_cur
            loss.backward()
            with torch.no_grad():
                opt.step()
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_loss_ging += loss_ging.item() * batch_size
                train_loss_cur += loss_cur.item() * batch_size
                train_loss_seg += loss_seg.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                train_true_cls.append(seg_np.reshape(-1))
                train_pred_cls.append(pred_np.reshape(-1))
                train_true_seg.append(seg_np)
                train_pred_seg.append(pred_np)
                train_label_seg.append(label[:, 0].type(torch.int32).reshape(-1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)

        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg)
        outstr = 'Train %d, loss: %.6f, loss_ging: %.6f, loss_seg: %.6f, loss_cur: %.6f, ' \
                 'train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % \
                 (epoch,
                  train_loss * 1.0 / count,
                  train_loss_ging * 1.0 / count,
                  train_loss_seg * 1.0 / count,
                  train_loss_cur * 1.0 / count,
                  train_acc,
                  avg_per_class_acc,
                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        test_loss_ging = 0.0
        test_loss_seg = 0.0
        test_loss_cur = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        sample_count = 0
        with torch.no_grad():
            for data, label, seg in test_loader:
                sample_count += data.shape[0]
                label_one_hot = label

                AvgAngle_curvature = data[:, :, 7]
                high_cur_num = int(len(AvgAngle_curvature[0]) * args.cur_threshold)
                AvgAngle_idx = np.argpartition(AvgAngle_curvature, -high_cur_num, axis=1)[:, -high_cur_num:]
                idx_base = np.arange(0, AvgAngle_curvature.size(0)).reshape(-1, 1) * AvgAngle_curvature.size(1)
                AvgAngle_idx = AvgAngle_idx + idx_base
                AvgAngle_idx = AvgAngle_idx.reshape(-1)

                data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
                AvgAngle_idx = AvgAngle_idx.to(device)
                data = data.permute(0, 2, 1)  # [B, 8, N]
                batch_size = data.size()[0]

                seg_pred, ging_pred = model(data, label_one_hot)  # [B, 33, N], [B, 2, N]
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                ging_pred = ging_pred.permute(0, 2, 1).contiguous()
                ging_gold = (seg.view(-1, 1).squeeze() > 0).to(dtype=torch.int64)

                loss_ging = args.w_ging * criterion1(ging_pred.view(-1, 2), ging_gold)
                loss_seg = criterion1(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
                loss_cur = args.w_cur * criterion2(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze(),
                                                   AvgAngle_idx)
                loss = loss_ging + loss_seg + loss_cur

                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_loss_ging += loss_ging.item() * batch_size
                test_loss_cur += loss_cur.item() * batch_size
                test_loss_seg += loss_seg.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                test_label_seg.append(label[:, 0].type(torch.int32).reshape(-1))
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_label_seg = np.concatenate(test_label_seg)
            test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg)
            outstr = 'Test :: loss: %.6f, test_loss_ging: %.6f, test_loss_seg: %.6f, test_loss_cur: %.6f, ' \
                     'test acc: %.6f, test avg acc: %.6f, test iou: %.6f, ' % \
                     (test_loss * 1.0 / count,
                      test_loss_ging * 1.0 / count,
                      test_loss_seg * 1.0 / count,
                      test_loss_cur * 1.0 / count,
                      test_acc,
                      avg_per_class_acc,
                      np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            best_test_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.save_path, '/%s/models/best_model.t7' % args.exp_name))
    io.cprint("best_test_iou: " + str(best_test_iou) + '; best_test_epoch: ' + str(best_test_epoch))


def test(args, io):
    test_loader = DataLoader(Teeth(partition='test', ROOT_PATH=args.data_path, num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda:0")
    sample_num = "Number of samples: " + str(len(test_loader.dataset))
    io.cprint(sample_num)
    # Try to load models
    seg_num_all = 33
    model = TSegFormer(args, seg_num_all).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda(device)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    criterion1 = cal_loss
    criterion2 = mask_cur_loss

    test_loss = 0.0
    test_loss_ging = 0.0
    test_loss_seg = 0.0
    test_loss_cur = 0.0
    count = 0.0
    model.eval()
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    sample_count = 0

    with torch.no_grad():
        for data, label, seg in tqdm.tqdm(test_loader):
            sample_count += data.shape[0]
            label_one_hot = label

            AvgAngle_curvature = data[:, :, 7]
            high_cur_num = int(len(AvgAngle_curvature[0]) * args.cur_threshold)
            AvgAngle_idx = np.argpartition(AvgAngle_curvature, -high_cur_num, axis=1)[:, -high_cur_num:]
            idx_base = np.arange(0, AvgAngle_curvature.size(0)).reshape(-1, 1) * AvgAngle_curvature.size(1)
            AvgAngle_idx = AvgAngle_idx + idx_base
            AvgAngle_idx = AvgAngle_idx.reshape(-1)

            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            AvgAngle_idx = AvgAngle_idx.to(device)
            data = data.permute(0, 2, 1)  # [B, 8, N]
            batch_size = data.size()[0]

            seg_pred, ging_pred = model(data, label_one_hot)  # [B, 33, N], [B, 2, N]
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            ging_pred = ging_pred.permute(0, 2, 1).contiguous()
            ging_gold = (seg.view(-1, 1).squeeze() > 0).to(dtype=torch.int64)

            loss_ging = args.w_ging * criterion1(ging_pred.view(-1, 2), ging_gold)
            loss_seg = criterion1(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
            loss_cur = args.w_cur * criterion2(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze(),
                                               AvgAngle_idx)
            loss = loss_ging + loss_seg + loss_cur

            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_loss_ging += loss_ging.item() * batch_size
            test_loss_cur += loss_cur.item() * batch_size
            test_loss_seg += loss_seg.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label[:, 0].type(torch.int32).reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg)
        all_f1, all_ppv, all_npv, all_sensitivity, all_specificity = calculate_metrics(test_pred_seg, test_true_seg,
                                                                                       test_label_seg)  # B,
        outstr = 'Test :: loss: %.6f, test_loss_ging: %.6f, test_loss_seg: %.6f, test_loss_cur: %.6f, ' \
                 'test acc: %.6f, test avg acc: %.6f, test iou: %.6f, ' \
                 'F1 score: %.6f, ppv: %.6f, npv: %.6f, sensitivity: %.6f, specificity: %.6f' % \
                 (test_loss * 1.0 / count,
                  test_loss_ging * 1.0 / count,
                  test_loss_seg * 1.0 / count,
                  test_loss_cur * 1.0 / count,
                  test_acc, avg_per_class_acc, np.mean(test_ious),
                  np.mean(all_f1), np.mean(all_ppv), np.mean(all_npv), np.mean(all_sensitivity),
                  np.mean(all_specificity))
    io.cprint(outstr)


def inference(args, io):
    f = open(args.file_path, 'r')
    device = torch.device("cuda" if args.cuda else "cpu")
    model = TSegFormer(args, 33).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    teeth_dict = json.load(f)
    feature = np.array(teeth_dict['feature'][:args.num_points], dtype=np.float32)
    xyz = feature[:, 0:3]
    feature = torch.Tensor(feature).type(torch.FloatTensor).to(device).unsqueeze(0).permute(0, 2, 1)
    seg_gold = np.array(teeth_dict['label'][:args.num_points], dtype=np.int64)
    category = torch.Tensor(teeth_dict['category']).type(torch.FloatTensor).to(device).unsqueeze(0)
    seg_pred = model(feature, category)[0].permute(0, 2, 1)
    seg_pred = seg_pred.max(dim=2)[1].reshape(-1).cpu().numpy()
    if category[0, 0] == 1:
        seg_pred[seg_pred > 0] += 16
    acc = metrics.accuracy_score(seg_gold, seg_pred)
    class_averaged_recall = metrics.balanced_accuracy_score(seg_gold, seg_pred)
    iou = calculate_shape_IoU(np.expand_dims(seg_pred, 0), np.expand_dims(seg_gold, 0), None)[0]

    print("Inference Accuracy             :", acc)
    print("Inference Class-averaged Recall:", class_averaged_recall)
    print("part-averaged IoU              :", iou)
    print("set of teeth in gold           :", set(seg_gold))
    print("set of teeth in gold           :", set(seg_pred.flatten()))
    generate_obj(xyz, seg_pred, "objs/pred.obj")
    generate_obj(xyz, seg_gold, "objs/gold.obj")


def generate_obj(xyz, seg, path):
    print(xyz.shape)
    print(seg.shape)
    with open(path, "w") as f:
        for idx in range(xyz.shape[0]):
            f.write(
                "v %.6f %.6f %.6f %.4f %.4f %.4f\n" % ((xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]) + seg2colors[seg[idx]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D tooth Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--data_path', type=str, default='./data', metavar='D',
                        help='Root path for storing IOS data')
    parser.add_argument('--save_path', type=str, default='./outputs', metavar='O',
                        help='Pre-trained model and log saving path')
    parser.add_argument('--batch_size', type=int, default=6, metavar='bs',
                        help='Size of batch when training)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='test_bs',
                        help='Size of batch when testing)')
    parser.add_argument('--epochs', type=int, default=200, metavar='E',
                        help='Number of episode to train')
    parser.add_argument('--cur_threshold', type=float, default=0.4,
                        help='Ratio of high curvature points used in the geometry-guided loss')
    parser.add_argument('--w_cur', type=float, default=0.001,
                        help='Weight of geometry-guided loss in the total loss')
    parser.add_argument('--w_ging', type=float, default=1,
                        help='Weight of auxiliary segmentation loss in the total loss')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='Sche',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluate the model')
    parser.add_argument('--num_points', type=int, default=10000,
                        help='Number of downsampling points')
    parser.add_argument('--model_path', type=str, default='', metavar='path',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_(args)
    io = IOStream(args.save_path + '/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    # inference(args, io)
    if not args.eval:
        train(args, io)
    else:
        test(args, io)
