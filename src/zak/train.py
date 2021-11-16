import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from zak.dataset import Parameters
from zak.loss import MSSLoss
from zak.model import Decoder

wandb_logger = WandbLogger(project="zak-ddsp", log_model='all')


# TODO: Noise goes to zero during training
class Zak(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = Decoder()
        self.loss = MSSLoss([2048, 1024, 512, 256, 128, 64])

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, params: Parameters, _batch_idx: int) -> torch.Tensor:
        audio, pitch, loudness = params
        x_hat = self.model(pitch.permute(0, 2, 1), loudness.permute(0, 2, 1))

        # Trim x_hat to size
        # TODO: this function makes or breaks the alignment. Seems correct at this time.
        # diff = x_hat.shape[1] - x['audio'].shape[1]
        # x_hat = x_hat[:, :-diff]

        loss = self.loss(x_hat, audio)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, params: Parameters, batch_idx: int) -> None:
        audio, pitch, loudness = params
        x_hat = self.model(pitch.permute(0, 2, 1), loudness.permute(0, 2, 1))

        columns = ['original', 'generated']
        data = []
        # Log 4 examples to tensorboard
        for i in range(4):
            data.append([wandb.Audio(audio[i].squeeze().cpu(), 48000, f':step {batch_idx}'),
                         wandb.Audio(x_hat[i].squeeze().cpu(), 48000, f':step {batch_idx}')])

        wandb_logger.log_table(key='samples', columns=columns, data=data)


def main() -> None:
    dataset = Parameters()
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4
    )
    model = Zak()
    wandb_logger.watch(model, log="all")
    trainer = pl.Trainer(
        gpus=1,
        limit_val_batches=1,
        logger=wandb_logger,
        log_every_n_steps=30,
        gradient_clip_val=3.0,
        val_check_interval=0.2,
    )
    trainer.fit(model, train_loader, train_loader)


if __name__ == '__main__':
    main()
