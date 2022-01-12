import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.swa_utils import AveragedModel, update_bn
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
#from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import accuracy_score

pl.seed_everything(7)

batch_size = 256

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)              # (Batch,  1, 28, 28) -> (Batch, 32, 26, 26)
        x = F.relu(x)
        x = self.conv2(x)              # (Batch, 32, 26, 26) -> (Batch, 64, 24, 24)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)         # (Batch, 64, 24, 24) -> (Batch, 64, 12, 12)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)        # (Batch, 64, 12, 12) -> (Batch, 9216)
        x = self.fc1(x)                # (Batch, 9216) -> (Batch, 128)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)                # (Batch, 128) -> (Batch, 10)

        return x


class LitSimplenet(pl.LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds, y)
        #acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = 45000 // batch_size
        # scheduler_dict = {
        #     'scheduler': OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
        #     'interval': 'step',
        # }
        return {'optimizer': optimizer}