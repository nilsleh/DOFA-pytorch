import pytorch_lightning as pl
import torch

class LightningTask(pl.LightningModule):
    def __init__(self, args, config, data_config):
        super().__init__()
        self.config = config #model_config
        self.args = args # args for optimization params
        self.data_config = data_config # dataset_config
        self.criterion = None

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def log_metrics(self, outputs, targets, prefix="train"):
        """Abstract method for logging task-specific metrics."""
        raise NotImplementedError("This method should be implemented in task-specific classes")

    def forward(self, samples):
        raise NotImplementedError("This method should be implemented in task-specific classes")

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.long()
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log_metrics(outputs, targets, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.long()
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log_metrics(outputs, targets, prefix="val")
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        if self.config.task == 'classification':
            optimizer = torch.optim.SGD(self.params_to_optimize(),
                           lr=self.args.lr,
                           weight_decay=self.args.weight_decay)
        else:
            param_groups = [
                {'params': self.neck.parameters(), 'lr': 0.001},
                {'params': self.decoder.parameters(), 'lr': 0.001},
                {'params': self.aux_head.parameters(), 'lr': 0.001}
            ]
            optimizer = torch.optim.AdamW(param_groups)
            
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lr_lambda=lambda epoch: epoch / self.args.warmup_epochs if epoch < self.args.warmup_epochs else 1.0)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs-self.args.warmup_epochs,
            eta_min=0.0001,
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.args.warmup_epochs])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
