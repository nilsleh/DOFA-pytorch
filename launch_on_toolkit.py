experiments = [
    {
        "model": "dinov2_seg",
        "dataset": "flair2",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "dinov2_seg",
        "dataset": "loveda",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
    {
        "model": "dinov2_seg",
        "dataset": "caffe",
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.002,
        "warmup_epochs": 3,
    },
]

# for each experiment launch a toolkit job

from dispatch_toolkit import start
