import pytorch_lightning as pyl
import torch
import datasets as dataset
import torch.utils.data
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
            batch['src_node_features'],
            batch['init_edge_index'],
            batch['batch_idx'],
            self.global_step
        )
        return x

    def training_step(self, batch, batch_idx):
        x = self(batch)

        if self.global_step > 500:
            lambda1 = 0.01
        else:
            lambda1 = 0

        loss_mi = x['loss']
        loss_sparse = x['loss_sparse']
        loss_edge_pre = x['loss_edge_pred']
        self.log('loss_mi', loss_mi, on_step=True, prog_bar=True, logger=False)
        self.log('loss_sparse', loss_sparse, on_step=True, prog_bar=True, logger=False)
        self.log('loss_edge_pre', loss_edge_pre, on_step=True, prog_bar=True, logger=False)
        self.log('max_probs', x['max_probs'], on_step=True, prog_bar=True, logger=False)
        self.log('min_probs', x['min_probs'], loss_mi, on_step=True, prog_bar=True, logger=False)
        loss = loss_mi + 0.01 * loss_sparse + lambda1 * loss_edge_pre
        return loss

    def validation_step(self, batch, batch_idx):

        output = self(batch)
        loss_mi = output['loss']
        loss_sparse = output['loss_sparse']
        loss_edge_pre = output['loss_edge_pred']

        return {'loss_mi': loss_mi, 'loss_sparse': loss_sparse, 'loss_edge_pre':loss_edge_pre}

    def validation_epoch_end(self, outputs):
        loss_mi = torch.cat([output['loss_mi'].reshape([1]) for output in outputs])
        loss_sparse = torch.cat([output['loss_sparse'].reshape([1]) for output in outputs])
        loss_edge_pre = torch.cat([output['loss_edge_pre'].reshape([1]) for output in outputs])
        loss_mi = torch.mean(loss_mi)
        loss_sparse = torch.mean(loss_sparse)
        loss_edge_pre = torch.mean(loss_edge_pre)


        self.log('loss_mi', loss_mi, sync_dist=True)
        self.log('loss_sparse', loss_sparse, sync_dist=True)
        self.log('loss_edge_pre', loss_edge_pre, sync_dist=True)
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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        scores, _ = self(batch)
        proba = torch.sigmoid(scores)
        labels = batch['labels']
        return proba.cpu().numpy().flatten(), labels.cpu().numpy().flatten()


if __name__ == '__main__':
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
        # sampler=dataset.RandomDropSampler(dataset_train, 0),
        collate_fn=collate_fn.dyg_collate_fn
    )

    loader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_data_workers,
        collate_fn=collate_fn.dyg_collate_fn
    )

    checkpoint_callback = pyl.callbacks.ModelCheckpoint(
        monitor = None,
        save_top_k = -1,
        save_last=True
    )

    trainer = pyl.Trainer(
        logger=pyl.loggers.CSVLogger('../lightning_logs_gsn'),
        gradient_clip_val=0.1,
        replace_sampler_ddp=False,
        max_epochs=10,
        gpus=gpus,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model, train_dataloaders=loader_train,
        val_dataloaders=loader_valid
    )