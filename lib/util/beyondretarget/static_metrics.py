"""Masked BCE and count-aggregated PR/F1. No mean-of-batch F1 bias."""
import torch
from torch.nn import functional as F
from lib.models.tram_static import JOINT_NAMES


def masked_bce_sum(logits, labels, mask):
    loss=F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction='none')
    return (loss*mask).sum(), mask.sum()


@torch.no_grad()
def metric_counts(logits, labels, mask):
    dims=tuple(range(logits.ndim-1));m=mask.bool();y=labels>.5;p=logits.sigmoid()
    values=[m.sum(dims), (m & y).sum(dims),
            (F.binary_cross_entropy_with_logits(logits.float(),labels.float(),reduction='none')*m).sum(dims)]
    for threshold in (.5,.8):
        pred=p>threshold
        values.extend([(m & pred & y).sum(dims), (m & pred & ~y).sum(dims), (m & ~pred & y).sum(dims)])
    return torch.stack(values).double()


def summarize_counts(counts):
    c=counts.detach().cpu().numpy();result={}
    for index,name in enumerate(JOINT_NAMES):
        n,pos,bce=c[:3,index]
        entry=dict(valid=int(n),positive_rate=float(pos/n) if n else None,bce=float(bce/n) if n else None)
        for k,threshold in enumerate(('0.5','0.8')):
            tp,fp,fn=c[3+3*k:6+3*k,index]
            entry[threshold]=dict(precision=float(tp/(tp+fp)) if tp+fp else None,
                recall=float(tp/(tp+fn)) if tp+fn else None,
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None)
        result[name]=entry
    n=c[0].sum()
    return dict(valid=int(n),bce=float(c[2].sum()/n) if n else None,joints=result)
