class AvgMeter:
    """Running average tracker for losses/metrics."""

    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        """Reset sum/count/avg to zero."""
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        """Add a new observation.

        Args:
            val: average value of the incoming batch (scalar float).
            count: number of samples in the batch (int).
        """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    """Return current learning rate (first param group, scalar float)."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
