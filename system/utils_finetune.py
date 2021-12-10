import torch
import torch
import torch.nn as nn
import sklearn.linear_model
from torch.nn.functional import normalize


@torch.no_grad()
def LR(encoder, support, support_ys, query, norm=False):
    """logistic regression classifier"""
    support = encoder(support).detach()
    query = encoder(query).detach()
    if norm:
        support = normalize(support)
        query = normalize(query)

    clf = sklearn.linear_model.LogisticRegression(random_state=0,
                                                  solver='lbfgs',
                                                  max_iter=1000,
                                                  C=1,
                                                  multi_class='multinomial')
    support_features_np = support.data.cpu().numpy()
    support_ys_np = support_ys.data.cpu().numpy()
    clf.fit(support_features_np, support_ys_np)

    query_features_np = query.data.cpu().numpy()
    query_ys_pred = clf.predict(query_features_np)

    pred = torch.from_numpy(query_ys_pred).to(support.device,
                                              non_blocking=True)
    return pred, query


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x


def set_bn_to_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
