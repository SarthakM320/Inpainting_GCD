
import torch
from torch import nn 
from torch.nn import functional as F
from torchvision import models

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=1.5, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)/self.temperature
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = torch.clamp(log_prob, max = 10000, min = -10000)
        # print(log_prob.isnan().any())
        # compute mean of log-likelihood over positive
        
        
        # print(mask)
        # print(log_prob)
        # print((mask * log_prob).sum(1))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print(mean_log_prob_pos.isnan().any(), (mean_log_prob_pos == 0).all())
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(loss.isnan().any())
        return loss


def info_nce_logits(features, args):
    device = args.device
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

# Alignment loss
    



# class AdverserialLoss(nn.Module): 
#     def __init__(self, num_classes, temperature = 0.2):
#         super().__init__()
#         # self.num_classes = num_classes
#         # self.y_t = torch.tensor([1/num_classes for _ in range(num_classes)])
#         # self.temperature = temperature


#     def forward(self, entropy, labelled_or_not):
#         source_entropy = torch.sum(entropy[labelled_or_not])
#         target_entropy = torch.sum(entropy[~labelled_or_not])
#         return source_entropy, target_entropy
    
class AdverserialLoss(nn.Module): 
    def __init__(self, num_classes, temperature = 1):
        super().__init__()
        self.num_classes = num_classes
        self.y_t = torch.tensor([1/num_classes for _ in range(num_classes)])
        self.temperature = temperature


    def forward(self, entropy, labelled_or_not):
        y = self.y_t.expand(entropy[~labelled_or_not].shape[0],-1)
        maxi,_ = torch.max(entropy[~labelled_or_not], dim = 0) # for normalizing

        return torch.sum(-y*F.softmax(
                    torch.divide(entropy[~labelled_or_not]-maxi.detach(),self.temperature), dim = 1
                ))
    
# def get_entropy(self, features, centers):
#     cos = []
#     for i in range(self.num_classes):
#         cos.append(nn.CosineSimilarity(dim = -1)(features, centers[:,i,:]).unsqueeze(1))

#     return torch.divide(F.softmax(torch.concat(cos, dim = 1), dim=1), self.temperature)



# only for labelled ones
class ContrastiveLossLabelled(nn.Module):
    def __init__(self, num_classes, normalize = True):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

    def forward(self, features, centers, labels, labelled_or_not):
        labels_temp = labels.clone()[labelled_or_not]
        features_temp = features.clone()[labelled_or_not]
        cos_dist_with_centers = F.cosine_similarity(features_temp.unsqueeze(1), centers, dim = 2)
        # print('COSINE')
        # print(cos_dist_with_centers)
        # positions = torch.zeros(len(labels_temp), self.num_classes, device = features.device) + -1
        # positions = positions.scatter(1, labels_temp.unsqueeze(1), 1)
        for i,l in enumerate(labels_temp):
            cos_dist = cos_dist_with_centers[i]
            cos_dist[cos_dist > 0] *= -1
            if cos_dist[l] < 0:
                cos_dist[l] *= -1

            cos_dist_with_centers[i] = cos_dist     
            # if cos_dist_with_centers[i][l]<0:
            #     cos_dist_with_centers[i][l] *- -1
            # for j in range(self.num_classes):
            #     if cos_dist_with_centers[i][j]>0 and j != l:
            #         cos_dist_with_centers[i][j] *= -1
                

        if self.normalize:
            return -torch.sum(F.normalize(
                cos_dist_with_centers,dim =1, p=1
            ))
        else:
            return -torch.sum(cos_dist_with_centers)

def gram_matrix(feat):
    # https://github.com/pytorch/ex`amples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, extractor=VGG16FeatureExtractor()):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, mask, output, gt):
        # loss = torch.tensor(0.0)
        loss_dict = {}
        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)
        loss_dict['prc'] = torch.tensor(0.0)
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
        loss_dict['style'] = torch.tensor(0.0)
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))

        # loss_dict['tv'] = total_variation_loss(output_comp)
        # for k,v in loss_dict.items():
        #     loss+=v
        
        return torch.sum(torch.stack(list(loss_dict.values())))

class MarginLoss(nn.Module):
    def __init__(self, margin, reconstruction_loss_object):
        super().__init__()
        self.margin = margin
        self.reconstruction_loss = reconstruction_loss_object

    def forward(self, sim, diff, mask, gt):
        loss = max(
            0, 
            self.reconstruction_loss(mask, sim, gt) - self.reconstruction_loss(mask, diff, gt) + self.margin
        )

        return loss
