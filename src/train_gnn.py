import pytorch_lightning as pyl
import torch
import torch.nn.functional as F
import numpy as np
import datasets as dataset
import torch.utils.data
import sklearn
from option import args
from model.tgat import TGAT



class ModelLightning(pyl.LightningModule):
    def __init__(self, config, backbone):
        super().__init__()
        self.config = config
        self.backbone = backbone
        pass


    def forward(self, batch):
        ##ToDo

        x = self.backbone(
            batch['src_edge_feat'],
            batch['src_edge_to_time'],
            batch['src_center_node_idx'],
            batch['src_neigh_edge'],
            batch['src_node_features']
        )
        return x

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        lables = batch['labels']

        loss = F.binary_cross_entropy_with_logits(
            logits, lables, reduction='none')
        loss = torch.mean(loss)
        self.log("loss2", loss, on_step=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):

        org_logits = self(batch).sigmoid()

        return {'org_proba': org_logits, 'label':batch['labels']}

    def validation_epoch_end(self, outputs):
        org_pred = torch.cat([output['org_proba'] for output in outputs])
        label = torch.cat([output['label'] for output in outputs])

        if torch.sum(label > 0):
            org_valid_auc = sklearn.metrics.roc_auc_score(label.cpu().numpy().flatten(), org_pred.cpu().numpy().flatten())
        else:
            org_valid_auc = 0
        self.log('org_valid_auc', org_valid_auc, sync_dist=True)
        self.log('learning rate', self.optimizers(0).param_groups[0]['lr'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.7)
        return [optimizer], [scheduler]

    def backward(
            self, loss, *args, **kargs):

        super().backward(loss, *args, **kargs)

        for p in self.parameters():
            if (p.grad is not None and torch.any(torch.isnan(p.grad))) or \
                    torch.any(torch.isnan(p)):
                raise RuntimeError('nan happend')
            pass
        pass

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        scores, _ = self(batch)
        proba = torch.sigmoid(scores)
        labels = batch['labels']
        return proba.cpu().numpy().flatten(), labels.cpu().numpy().flatten()




if __name__=='__main__':

    config = args
    dataset_train = dataset.DygDataset(config, 'train')
    dataset_valid = dataset.DygDataset(config, 'valid')

    gpus = None if config.gpus == 0 else config.gpus

    collate_fn = dataset.Collate(config)

    backbone = TGAT(config)
    model = ModelLightning(
         config, backbone=backbone)

    loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_data_workers,
        pin_memory=True,
        #sampler=dataset.RandomDropSampler(dataset_train, 0),
        collate_fn=collate_fn.dyg_collate_fn
    )

    loader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_data_workers,
        collate_fn=collate_fn.dyg_collate_fn
    )

    trainer = pyl.Trainer(
        logger=pyl.loggers.CSVLogger('../lightning_logs_gnn'),
        gradient_clip_val=0.1,
        replace_sampler_ddp=False,
        max_epochs=10,
        gpus=gpus
    )

    trainer.fit(
        model, train_dataloaders=loader_train,
        val_dataloaders=loader_valid
    )