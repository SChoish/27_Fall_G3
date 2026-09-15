"""Rank-N-Contrast-style loss; validated against direct reference."""
import torch

def rank_n_contrast(z, y, temperature=1.0):
    """RNC over one view per physical design, vector labels normalized on train only.

    Every j != i competes only with k != i at least as far away in label space.
    Stable O(B^2) storage/O(B^2 log B) ranking, exact tie handling, no cubic mask.
    """
    n = z.shape[0]
    if n < 2:
        return z.sum()*0
    distances = torch.cdist(y.detach(),y.detach(),p=1)
    logits = -torch.cdist(z,z,p=2)/temperature
    diagonal = torch.eye(n,dtype=torch.bool,device=z.device)
    distances = distances.masked_fill(diagonal,float("-inf"))
    logits = logits.masked_fill(diagonal,float("-inf"))
    order = distances.argsort(dim=1,descending=True)
    sorted_dist = distances.gather(1,order)
    sorted_logits = logits.gather(1,order)
    denom = torch.logcumsumexp(sorted_logits,dim=1)
    # Rightmost entry of a tie group includes all equally distant competitors.
    positions = torch.arange(n,device=z.device).expand(n,n)
    ends = torch.ones_like(sorted_dist,dtype=torch.bool)
    ends[:,:-1] = sorted_dist[:,:-1] != sorted_dist[:,1:]
    end_position = torch.where(ends,positions,n)
    end_position = torch.flip(torch.cummin(torch.flip(end_position,[1]),dim=1).values,[1])
    denom = denom.gather(1,end_position)
    return (denom[:,:-1]-sorted_logits[:,:-1]).mean()

def rnc_reference(z,y,temperature=1.0):
    """Small, intentionally direct reference used only to validate rank/tie logic."""
    d = torch.cdist(y,y,p=1)
    s = -torch.cdist(z,z,p=2)/temperature
    terms = []
    for i in range(len(z)):
        for j in range(len(z)):
            if i != j:
                m = (d[i] >= d[i,j])
                m[i] = False
                terms.append(torch.logsumexp(s[i,m],0)-s[i,j])
    return torch.stack(terms).mean()
