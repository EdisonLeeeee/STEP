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
        x =  self.backbone(
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
        return {'org_proba': org_logits,  'label': batch['labels']}

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
        scores = self(batch).sigmoid()
        labels = batch['labels']

        return scores.cpu().numpy().flatten(), labels.cpu().numpy().flatten()




if __name__=='__main__':

    config = args
    dataset_valid = dataset.DygDataset(config, 'test')

    gpus = None if config.gpus == 0 else config.gpus

    collate_fn = dataset.Collate(config)

    backbone = TGAT(config)
    model = ModelLightning(
         config, backbone=backbone)

    ckpt_file = config.ckpt_file
    pretrained_dict = torch.load(ckpt_file)['state_dict']
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in pretrained_dict.items() if
                  k.split('.')[1] in ['embedding_module', 'time_encoder', 'node_preocess_fn',
                                      'edge_preocess_fn', 'affinity_score']}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()


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

    with torch.no_grad():
        pred = trainer.predit(model, loader_valid)
        pass

    prob, label = [x[0] for x in pred], [x[1] for x in pred]
    prob = np.hstack(prob)
    label = np.hstack(label)

    org_valid_auc = sklearn.metrics.roc_auc_score(label.astype(int), prob)
    print('test_acu>>>>>>>>>>>>>>>>>>', org_valid_auc)