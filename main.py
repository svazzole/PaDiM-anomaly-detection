import random
from random import sample
import argparse
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.cfrp as cfrp

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='./crfp')
    parser.add_argument('--save_path', type=str, default='./cfrp_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()

def main():
    torch.cuda.empty_cache()
    args = parse_args()
    
    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
        
    model.to(device)
    model.eval()

    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))
    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig_img_rocauc = ax

    total_roc_auc = []
    # starting...
    torch.cuda.empty_cache()

    mean_outputs, cov_inv_outputs = [], []

    for class_name in cfrp.CLASS_NAMES:
        train_dataset = cfrp.CFRPDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = cfrp.CFRPDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        iteration = 0
        num_epochs = 5
        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for epoch in range(num_epochs):
                for (x, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

                    # model prediction
                    with torch.no_grad():
                        _ = model(x.to(device))
                    # get intermediate layer outputs
                    for k, v in zip(train_outputs.keys(), outputs):
                        train_outputs[k].append(v.cpu().detach())
                    # initialize hook outputs
                    outputs = []

                    for k, v in train_outputs.items():
                        train_outputs[k] = torch.cat(v, 0)

                    # Embedding concat
                    embedding_vectors = train_outputs['layer1']
                    try:
                        for layer_name in ['layer2', 'layer3']:
                            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
                    except:
                        print('memory lack error')

                    # randomly select d dimension
                    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
                    # calculate multivariate Gaussian distribution
                    B, C, H, W = embedding_vectors.size()
                    print('B: {}, C: {}, H: {}, W: {}'.format(B, C, H, W))

                    embedding_vectors = embedding_vectors.view(B, C, H * W)
                    mean = torch.mean(embedding_vectors, dim=0).numpy()
                    cov = torch.zeros(C, C, H * W).numpy()
                    cov_inv = torch.zeros(C, C, H * W).numpy()
                    I = np.identity(C)

                    for i in range(H * W):
                        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
                        cov_inv[:, :, i] =  np.linalg.inv(cov[:, :, i])
                    
                    if not isinstance(mean_outputs, np.ndarray):
                        mean_outputs = mean
                    else:
                        mean_outputs = np.add(mean_outputs, mean)

                    if not isinstance(cov_inv_outputs, np.ndarray):
                        cov_inv_outputs = cov_inv
                    else:
                        cov_inv_outputs = np.add(cov_inv_outputs, cov_inv)

                    iteration += 1

            mean_outputs /= iteration
            cov_inv_outputs /= iteration

            train_outputs = [mean_outputs, cov_inv_outputs]
            
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f,protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)
        
        torch.cuda.empty_cache()

        gt_list = []
        test_imgs = []
        
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for (x, y) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):    
            x = x.view(-1, 3, 224, 224)
            y = torch.Tensor(list(k for k in y for _ in range(4)))
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        start = time.time()
        print('step 1....')
        B, C, H, W = embedding_vectors.size()
        embedding_vectors_tensor = embedding_vectors.view(B, C, H * W).to(device)
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_tensor = torch.Tensor()
        mean_tensor = torch.Tensor(np.array(train_outputs[0])).to(device)
        cov_inv_tensor = torch.Tensor(np.array(train_outputs[1])).to(device)
        
        print('step 2....')
        for i in range(H * W):
            delta = embedding_vectors_tensor[:,:,i] - mean_tensor[:, i]
            m_dist = torch.sqrt(torch.diag(torch.mm(torch.mm(delta, cov_inv_tensor[:, :, i]),delta.t())))
            m_dist = m_dist.view(1, -1)
            if dist_tensor.shape[0] == 0:
                dist_tensor = m_dist
            else:
                dist_tensor = torch.cat([dist_tensor, m_dist], dim=0)
        
        print('step 3....')
        dist_tensor = dist_tensor.transpose(1, 0).view(B, H, W)
        score_map = F.interpolate(dist_tensor.unsqueeze(1), size=x.size(2), mode='bilinear',
                                align_corners=False).squeeze().cpu().numpy()
        print("time :", time.time() - start)

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        
        scores = (score_map - min_score) / (max_score - min_score)
        
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, thresholds = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        distances = (tpr - 1.) ** 2 + fpr ** 2 # distances from (1,0) in roc curve
        best_index = np.argmin(distances)

        threshold = (thresholds[best_index] + thresholds[best_index +1]) / 2
        # threshold = 0.38
        print('best_threshold: {}, threshold: {}'.format(thresholds[best_index], threshold))

        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_list, threshold, save_dir, class_name)

        print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
        fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
        fig_img_rocauc.legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)

    torch.cuda.empty_cache()

def plot_fig(test_img, scores, gt_list, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    preds = []
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        # gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        if mask[mask == 255].size > 0:
            preds.append(1)
        else:
            preds.append(0)
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')
        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

    confusion = confusion_matrix(gt_list, preds)
    accuracy = accuracy_score(gt_list, preds)
    precision = precision_score(gt_list, preds)
    recall = recall_score(gt_list, preds)
    f1 = f1_score(gt_list, preds)

    print('Confusion Matrix')
    print(confusion)

    print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1 score: {3:.4f}'\
        .format(accuracy, precision, recall, f1))

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y, use_cuda=False):
    B, C1, H1, W1 = x.size() # x [B, 256, 56, 56]
    _, C2, H2, W2 = y.size() # y [B, 512, 28, 28]
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2) # view = reshape
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    if use_cuda:
        z = z.to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
    
