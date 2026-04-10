import torch
import torch.nn as nn


class model_weight(nn.Module):
    def __init__(self):
        super(model_weight, self).__init__()

    def cc(self, s_map, gt):
        s_map_norm = (s_map - torch.mean(s_map)) / torch.std(s_map)
        gt_norm = (gt - torch.mean(gt)) / torch.std(gt)
        a = s_map_norm
        b = gt_norm
        r = torch.sum(a * b) / torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
        return r

    def sim(self, s_map, gt):

        s_map_norm = (s_map - torch.min(s_map)) / (torch.max(s_map) - torch.min(s_map))
        s_map_norm = s_map_norm / torch.sum(s_map_norm)
        gt_norm = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
        gt_norm = gt_norm / torch.sum(gt_norm)
        t1 = (torch.minimum(s_map_norm, gt_norm))
        t2 = torch.sum(t1)

        return t2

    def compute_cc_sim(self, s_map, gt):
        cc_score = torch.zeros(s_map.size(0))
        sim_score = torch.zeros(s_map.size(0))
        for i in range(s_map.size(0)):
            cc_score[i] = self.cc(s_map[i], gt[i])
            sim_score[i] = self.sim(s_map[i], gt[i])
        return cc_score, sim_score

    def forward(self, x):
        x1 = torch.unbind(x, dim=2)
        x_mean = torch.mean(x, dim=2)

        cc_score = torch.zeros((x.size(2), x.size(0))).cuda()
        sim_score = torch.zeros((x.size(2), x.size(0))).cuda()

        for i in range(16):
            cc, sim = (self.compute_cc_sim(x1[i], x_mean))
            cc_score[i] = cc
            sim_score[i] = sim

        return cc_score, sim_score


class model_weight_2(nn.Module):
    def __init__(self):
        super(model_weight_2, self).__init__()

    def cc(self, s_map, gt):
        s_map_norm = (s_map - torch.mean(s_map)) / torch.std(s_map)
        gt_norm = (gt - torch.mean(gt)) / torch.std(gt)
        a = s_map_norm
        b = gt_norm
        r = torch.sum(a * b) / torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
        return r

    def sim(self, s_map, gt):

        s_map_norm = (s_map - torch.min(s_map)) / (torch.max(s_map) - torch.min(s_map))
        s_map_norm = s_map_norm / torch.sum(s_map_norm)
        gt_norm = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
        gt_norm = gt_norm / torch.sum(gt_norm)
        t1 = (torch.minimum(s_map_norm, gt_norm))
        t2 = torch.sum(t1)

        return t2

    def compute_cc_sim(self, s_map, gt):
        cc_score = torch.zeros(s_map.size(0))
        sim_score = torch.zeros(s_map.size(0))
        for i in range(s_map.size(0)):
            cc_score[i] = self.cc(s_map[i], gt[i])
            sim_score[i] = self.sim(s_map[i], gt[i])
        return cc_score, sim_score

    def forward(self, x):

        with torch.no_grad():
            x1 = torch.unbind(x, dim=2)
            x_mean = torch.mean(x, dim=2)

            cc_score = torch.zeros((x.size(2), x.size(0))).cuda()
            sim_score = torch.zeros((x.size(2), x.size(0))).cuda()

            for i in range(16):
                cc, sim = (self.compute_cc_sim(x1[i], x_mean))
                cc_score[i] = cc
                sim_score[i] = sim

        return cc_score, sim_score, x_mean
