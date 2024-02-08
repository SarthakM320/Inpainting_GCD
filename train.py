import argparse

import math
import numpy as np
from data.officehome import OfficeHome
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from sklearn.cluster import KMeans
from util.general_utils import init_experiment, str2bool
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import ContrastiveLearningViewGenerator, ViTModel, Discriminator, Decoder
from losses import ReconstructionLoss, ContrastiveLossLabelled, info_nce_logits, AdverserialLoss, MarginLoss, VGG16FeatureExtractor, SupConLoss
import warnings
# warnings.filterwarnings('ignore')

def get_entropy(features, centers,args, temperature=0.7):
    cos = F.cosine_similarity(features.unsqueeze(1), centers, dim = 2)
    # return torch.log(torch.divide(F.softmax(cos, dim=1), temperature))
    return F.softmax(cos, dim = 1)

# target domain
# min max(cross entropy loss)
# min - discriminator
# max - generator


def train(vit, decoder, disc, train_loader, test_loader, args):
    optim_decoder = SGD(
        list(vit.parameters())+list(decoder.parameters()), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    optim_disc = SGD(
        list(disc.parameters()),
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    optims = [optim_decoder, optim_disc]
    schedulers = []
    
    for optimizer in optims:
        schedulers.append(lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        ))

    recon_loss_fn = ReconstructionLoss(VGG16FeatureExtractor().to(args.device))
    margin_loss_fn = MarginLoss(1, reconstruction_loss_object=recon_loss_fn)
    adv_loss_fn = AdverserialLoss(args.num_labeled_classes)
    sup_con_fn = ContrastiveLossLabelled(args.num_labeled_classes)
    # sup_con_fn = SupConLoss()

    
    centers = torch.zeros((args.num_labeled_classes,768))
    num_updates_centers = torch.zeros(40)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        vit.train()
        decoder.train()
        disc.train()
        for batch_idx, batch in enumerate(train_loader):
            
            images, masked_images, pos, neg, mask, labels, uq_idxs,labelled_or_not = batch
            images = torch.cat(images, dim=0)
            masked_images = torch.cat(masked_images, dim=0)
            pos = torch.cat(pos, dim=0)
            neg = torch.cat(neg, dim=0)
            mask = torch.cat(mask, dim=0).unsqueeze(1)
            labels = torch.cat([labels for _ in range(2)])
            labelled_or_not = torch.concat([labelled_or_not.bool().reshape(-1) for _ in range(2)])
            

            # extract features from vit 
            cls_images = vit(images)[:,0,:]
            cls_masked_image = vit(masked_images)[:,0,:]
            cls_pos = vit(pos)[:,0,:]
            cls_neg = vit(neg)[:,0,:]


            # editing the centers
            labelled_features = cls_images[labelled_or_not]
            labelled_data_label = labels[labelled_or_not]

            for i in range(labelled_features.shape[0]):
                l = labelled_data_label[i]
                centers[l] = (centers[l]*num_updates_centers[l] + labelled_features[i])/(num_updates_centers[l] + 1)
                num_updates_centers[l] = num_updates_centers[l]+1

            # print('CENTERS')
            # print(centers)


            mask_and_image = torch.concat([cls_masked_image.unsqueeze(1), cls_images.unsqueeze(1)], dim = 1).permute(0,2,1)
            mask_and_pos = torch.concat([cls_masked_image.unsqueeze(1), cls_pos.unsqueeze(1)], dim = 1).permute(0,2,1)
            mask_and_neg = torch.concat([cls_masked_image.unsqueeze(1), cls_neg.unsqueeze(1)], dim = 1).permute(0,2,1)

            mask_and_image_reconstructed = decoder(mask_and_image.unsqueeze(-1))
            mask_and_pos_reconstructed = decoder(mask_and_pos.unsqueeze(-1))
            mask_and_neg_reconstructed = decoder(mask_and_neg.unsqueeze(-1))
            # print(mask_and_image_reconstructed.shape)
            
            if sum(~labelled_or_not)>0:
                contrastive_logits, contrastive_labels  = info_nce_logits(features = cls_images[~labelled_or_not], args = args)
                info_nce_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
            else:
                print('info_nce_loss = 0')
                info_nce_loss = torch.tensor(0)
            if sum(labelled_or_not)>0:
                sup_con_loss = sup_con_fn(cls_images, centers.detach(), labels,labelled_or_not) # supervised loss with the centers 
                # sup_con_loss = sup_con_fn(cls_images[labelled_or_not].clone().unsqueeze(1), labels=label[labelled_or_not].clone())
            else:
                print('sup_con_loss = 0')
                sup_con_loss = torch.tensor(0)

            output_disc = disc(cls_images.detach())
            entropy = get_entropy(output_disc, centers.detach(), args)
            adv_loss_disc = adv_loss_fn(entropy, labelled_or_not) # this is negative 
            
            for optim in optims:
                optim.zero_grad()
            
            with torch.autograd.set_detect_anomaly(True):
                adv_loss_disc.backward()
                optim_disc.step()

            disc.set_lambda(1)
            output_disc = disc(cls_images,reverse = True)
            entropy_2 = get_entropy(output_disc, centers.detach(), args)
            adv_loss_gen = adv_loss_fn(entropy_2, labelled_or_not) # this is positive overall

            recon_loss = recon_loss_fn(mask, mask_and_image_reconstructed, images) #mask, output, gt
            margin_loss = margin_loss_fn(mask_and_pos_reconstructed,mask_and_neg_reconstructed, mask, images) #sim, diff, mask, gt
            
            # print()
            # print(f'recon_loss: {recon_loss}')
            # print(f'margin_loss: {margin_loss}')
            # print(f'adv_loss_gen: {adv_loss_gen}') # negative
            # print(f'adv_loss_disc: {adv_loss_disc}')
            # print(f'sup_con_loss: {sup_con_loss}') # negative
            # print(f'info_nce_loss: {info_nce_loss}')
            # print()

            # sup_con_loss and adv_loss_gen is failing

            loss = info_nce_loss + recon_loss + margin_loss + sup_con_loss + adv_loss_gen
            # loss = sup_con_loss

            loss_record.update(loss.item(), labels.size(0))
            with torch.autograd.set_detect_anomaly(False):
                loss.backward()
                optim_decoder.step()

            pstr = ''
            pstr += f'adv_loss_disc: {adv_loss_disc.item():.4f} '
            pstr += f'adv_loss_gen: {adv_loss_gen.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'recon_loss: {recon_loss.item():.4f} '
            pstr += f'margin_loss: {margin_loss.item():.4f} '
            pstr += f'info_nce_loss: {info_nce_loss.item():.4f} '

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

            break

        # print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg,
        #                                                                           train_acc_record.avg))
        print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg, 0))
        
        with torch.no_grad():
            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_kmeans(vit, test_loader, epoch=epoch, save_name='Test ACC', args=args)
            
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        for scheduler in schedulers:
            scheduler.step()

        save_dict = {
            'vit': vit.state_dict(),
            'disc':disc.state_dict(),
            'decoder':decoder.state_dict(),
            'optim_disc': optim_disc.state_dict(),
            'optim_decoder': optim_decoder.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        # yet to add model saving according to best accuracy
    

def test_kmeans(model, test_loader,
                epoch, save_name,
                args):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, data in enumerate(tqdm(test_loader)):
        
        images, _, _, _, _, label, _ = data
        images = torch.cat(images, dim=0)
        label = torch.cat([label for _ in range(2)])
        # images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)[:,0,:]

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
        

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='officehome', help='options:officehome')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=True)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    # device = torch.device('cuda:0')
    device = 'cpu'
    args.device = device
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    init_experiment(args, runner_name=[args.exp_name])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')


    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    vit = ViTModel().to(device)
    decoder = Decoder(
        in_channels = args.feat_dim, 
        output_resolution=args.image_size, 
        out_channels=3
    )
    disc = Discriminator(output_dim = args.feat_dim)

    for m in vit.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in vit.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[2])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True,sampler = sampler)
    # val_dataloader = DataLoader(datasets['val'], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
    #                                     batch_size=args.batch_size, shuffle=False)
    test_loader_unlabelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    
    train(vit, decoder, disc, train_loader, test_loader_unlabelled, args)
    

