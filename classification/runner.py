import torch
import copy
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, MultiStepLR, CyclicLR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from aux_dataset import res_im, crop_im
from sklearn import metrics
import sys
import os



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, dataloaders, criterion, optimizer, fold, aux, penalize, num_epochs, scheduler, v2_size):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_score = 0.0
    cl = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.3)

    for epoch in range(num_epochs):
        tst_outs = []
        pred_labels = []
        gt_labels = []
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels, gt_segs0, gt_segs01, gt_segs1, gt_segs2 in tqdm(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()
                gt_segs0 = gt_segs0.cuda()
                gt_segs01 = gt_segs01.cuda()
                gt_segs1 = gt_segs1.cuda()
                gt_segs2 = gt_segs2.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs, seg_mask0, seg_mask01, seg_mask1, seg_mask2, seg_mask3, seg_mask4 = model(inputs)


                    if len(outputs.shape) == 1:
                        outputs = outputs.unsqueeze(0)

                    if not penalize:
                        for i, gt_seg in enumerate(gt_segs1):
                            if gt_seg.sum() == 0:
                                seg_mask0[i] = gt_segs0[i]
                                seg_mask1[i] = gt_segs1[i]
                                seg_mask2[i] = gt_segs2[i]
                                seg_mask3[i] = gt_segs2[i]
                                seg_mask4[i] = gt_segs2[i]

                    
                    if aux == "567":
                        loss = criterion[0](outputs, labels) + 3 * (criterion[1](seg_mask1.squeeze().squeeze(), gt_segs1.squeeze()) +
                                                                    criterion[1](seg_mask2.squeeze().squeeze(), gt_segs2.squeeze()) +
                                                                    criterion[1](seg_mask3.squeeze().squeeze(), gt_segs2.squeeze()))
                        #loss = criterion[0](outputs, labels)
                    elif aux == "56":
                        loss = criterion[0](outputs, labels) + 3 * (
                                    criterion[1](seg_mask1.squeeze().squeeze(), gt_segs1.squeeze()) +
                                    criterion[1](seg_mask2.squeeze().squeeze(), gt_segs2.squeeze()))                    

                    _, preds = torch.max(outputs, 1)

                    if phase == "val":
                        tst_outs.extend(outputs.cpu().numpy())
                        pred_labels.extend(preds.cpu().numpy())
                        gt_labels.extend(labels.cpu().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == "val":
                scheduler.step(epoch_loss)

                print(confusion_matrix(gt_labels, pred_labels))
                tst_lbls = F.one_hot(torch.tensor(gt_labels), num_classes=4).numpy()
                tst_outs = np.array(tst_outs)

                score = 0

                softmax = torch.nn.Softmax(dim=1)
                tst_outs = softmax(torch.tensor(tst_outs))

                for x in range(4):
                    print(f"class {x}: ", average_precision_score(tst_lbls[:, x], tst_outs[:, x]))
                    score += average_precision_score(tst_lbls[:, x], tst_outs[:, x])

                print("Study level: ", score / 6)

                if score > best_score:
                    best_score = score
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, f"best_models/epoch_{epoch}_map_{best_score/6}_acc_{epoch_acc}_{aux}_fold{fold}_im{res_im}to{crop_im}_effv2_{v2_size}.pth")
                #print("AUC score: ", metrics.roc_auc_score(np.array(gt_labels), tst_outs))
                print()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            print('learning_rate:', str(get_lr(optimizer)))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_acc_model_wts, f"best_models/epoch_{epoch}_acc_{best_acc}_map_{score/6}_{aux}_fold{fold}_im{res_im}to{crop_im}_effv2_{v2_size}_.pth")
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best study level score: {:4f}'.format(best_score / 6))
