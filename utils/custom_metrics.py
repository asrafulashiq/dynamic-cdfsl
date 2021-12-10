import torch
import torch.distributed as dist


class AverageMeter():
    def __init__(self, use_ddp=False):
        super().__init__()
        self.reset()
        self.use_ddp = use_ddp
        self.best_val = -1

    def add(
        self,
        value,
        n=1,
    ):
        if self.use_ddp is False:
            self.val = value
            self.sum += value
            self.var += value * value
            self.n += n
        else:
            var = value**2
            dist.all_reduce(value)
            dist.all_reduce(var)
            self.sum += value
            self.var += var
            self.n += n * dist.get_world_size()
            self.val = value / dist.get_world_size()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0

    @property
    def value(self):
        return self.val

    @property
    def mean(self):
        return (self.sum / self.n)

    @property
    def avg(self):
        return self.mean

    @property
    def max(self):
        mean_val = self.mean
        if self.use_ddp:
            dist.all_reduce(mean_val)
            mean_val /= dist.get_world_size()

        if mean_val > self.best_val:
            self.best_val = mean_val

        return self.best_val

    @property
    def std(self):
        acc_std = torch.sqrt(self.var / self.n - self.mean**2)
        return 1.96 * acc_std / np.sqrt(self.n)
