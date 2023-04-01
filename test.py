import torch

def make_causal_mask(attention_logit):
    length = attention_logit.size(-2)
    mask = torch.triu(torch.ones(length, length), 1) * -100000.0
    return mask.to(attention_logit.device)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

x = torch.randn([1, 1, 10, 10])
print(make_causal_mask(x)==0)
print(generate_square_subsequent_mask(10)==0)