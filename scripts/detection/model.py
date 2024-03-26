import lightning.pytorch as pl
import transformers as tr
import torch


class DetectionModel(pl.LightningModule):

    def __init__(self, max_class_id: int, lr=1e-4):
        super(DetectionModel, self).__init__()
        self.lr = lr
        self.detr = tr.DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=max_class_id + 1,
            ignore_mismatched_sizes=True)

        # freeze all modules except the classification head
        self.detr.model.requires_grad_(False)
        self.detr.bbox_predictor.requires_grad_(False)
        self.detr.class_labels_classifier.requires_grad_(True)

        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask, labels=None):
        return self.detr(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.detr(**batch)

        self.log('train/loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in outputs['loss_dict'].items():
            self.log(f'train/{k}', v, on_step=True, on_epoch=True, sync_dist=True)

        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        outputs = self.detr(**batch)

        self.log('val/loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in outputs['loss_dict'].items():
            self.log(f'val/{k}', v, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.detr.parameters(), lr=self.lr)

    def unfreeze(self):
        for param in self.detr.parameters():
            param.requires_grad = True