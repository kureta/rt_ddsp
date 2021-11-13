import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from zak.model import Decoder
from zak.dataset import Parameters
from zak.loss import MSSLoss


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

    def on_epoch_start(self) -> None:
        if self.current_epoch == 0:
            f0 = torch.randn(1, 1, 1)
            amp = torch.randn(1, 1, 1)
            self.logger.experiment.add_graph(Decoder(), (f0, amp))

    def training_step(self, params: Parameters, _batch_idx: int) -> torch.Tensor:
        audio, pitch, loudness = params
        x_hat = self.model(pitch.permute(0, 2, 1), loudness.permute(0, 2, 1))

        # Trim x_hat to size
        # TODO: this function makes or breaks the alignment. Seems correct at this time.
        # diff = x_hat.shape[1] - x['audio'].shape[1]
        # x_hat = x_hat[:, :-diff]

        loss = self.loss(x_hat, audio)

        self.log('train_loss', loss)
        self.logger.experiment.add_histogram(
            'impulse', self.model.reverb.ir, self.global_step
        )
        self.logger.experiment.add_histogram(
            'noise/weight', self.model.controller.dense_filter.weight, self.global_step
        )
        self.logger.experiment.add_histogram(
            'noise/bias', self.model.controller.dense_filter.bias, self.global_step
        )
        return loss

    def validation_step(self, params: Parameters, batch_idx: int) -> None:
        audio, pitch, loudness = params
        x_hat = self.model(pitch.permute(0, 2, 1), loudness.permute(0, 2, 1))

        # Log 4 examples to tensorboard
        for i in range(4):
            self.logger.experiment.add_audio(
                f'{batch_idx}-{i}',
                x_hat[i].unsqueeze(1),
                self.global_step,
                sample_rate=48000,
            )
            self.logger.experiment.add_audio(
                f'{batch_idx}-{i}-orig',
                audio[i].unsqueeze(1),
                self.global_step,
                sample_rate=48000,
            )


def main() -> None:
    dataset = Parameters()
    train_loader = DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4
    )
    model = Zak()
    logger = TensorBoardLogger('lightning_logs', default_hp_metric=False)
    trainer = pl.Trainer(
        gpus=1,
        limit_val_batches=1,
        logger=logger,
        log_every_n_steps=30,
        gradient_clip_val=3.0,
        val_check_interval=1.0,
    )
    trainer.fit(model, train_loader, train_loader)


if __name__ == '__main__':
    main()
