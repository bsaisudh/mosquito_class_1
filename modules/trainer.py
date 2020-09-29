from mosquito_class_1.utils.gen_utils import OutPutDir
from mosquito_class_1.zoo.model_utils import generate_model
from mosquito_class_1.utils.torch_utils import (find_all_substr,
                                                get_best_epoch_and_accuracy,
                                                get_loss_fn,
                                                get_optimizer,
                                                weights_init,
                                                SummaryStatistics)
from mosquito_class_1.modules.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import pprint
import math
import os
import time
import copy
import datetime
from datetime import datetime
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class Trainer():
    def __init__(self, args, _out_dir: OutPutDir):
        self.gen_args = args["GENERAL"]
        self.model_args = args["MODEL"]
        self.train_args = args["TRAIN_PARAM"]
        self.data_args = args["DATA_PARAM"]

        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.accuracy_updated = False

        self.out_dir = _out_dir
        self.device = torch.device(self.gen_args["DEVICE"])
        self.writer = SummaryWriter(log_dir=self.out_dir.log_dir,
                                    comment=self.gen_args["COMMENT"])
        self.tr_sequence = [[1, 2, 3, 4, 0],
                            [0, 2, 3, 4, 1],
                            [0, 1, 3, 4, 2],
                            [0, 1, 2, 4, 3],
                            [0, 1, 2, 3, 4]]

        self.summary_statistics = None
        self.epoch = 0
        self.num_epochs = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_wts = None
        self.best_acc = 0.0
        self.best_epoch = None
        self.best_loss = math.inf
        self.model = None
        self.loader = None
        self.loss = None

        self.init_model()
        self.init_train_patameters()
        self.init_data_loader()
        self.load_model()

    def init_train_patameters(self):
        self.num_epochs = self.train_args["EPOCH"]
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.train_args["LR"])
        # Decay LR by a factor of 0.75 every 15 epochs
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.train_args["STEP_SIZE"],
            gamma=self.train_args["LR_DECAY"]
        )
        self.loss = nn.CrossEntropyLoss()

    def init_model(self):
        self.model = generate_model(self.model_args, self.device)
        self.model.to(self.device)
        self.model.cuda()

    def init_data_loader(self):
        self.loader = DataLoader(self.gen_args, self.data_args, self.out_dir)
        self.summary_statistics = SummaryStatistics(self.loader.n_classes)

    def save_model(self, comment=""):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss},
            self.out_dir.get_checkpoint_file(self.epoch, comment))

    def load_model(self):
        """Load pretrained weights for model."""
        if self.train_args["PRETRAIN_MODEL"] != "":
            path = os.path.join(self.out_dir.root_dir,
                                self.train_args["PRETRAIN_MODEL"])
            checkpoint = torch.load(path, map_location=self.gen_args["DEVICE"])
            try:
                self.model.load_state_dict(
                    checkpoint['model_state_dict'], strict=True)
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                self.meta_info['epoch'] = checkpoint['epoch']
                # self.epoch_info['mean_loss'] = checkpoint['loss_value']
                self.loss = checkpoint['loss']
            except:
                self.model.load_state_dict(checkpoint, strict=True)
            print("loaded model : ", self.train_args["PRETRAIN_MODEL"])

    def train(self):
        # 1. Average validation accuracy for each epochs.
        since = time.time()
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0

        for self.epoch in range(self.num_epochs):
            epoch_since = time.time()
            if(self.epoch % self.train_args["TEST_INTERVAL"] == 0):
                log_text = f"Checkpoint saved ... Epoch : {self.epoch}"
                print(log_text)
                self.save_model()
            if(self.epoch > 0 and self.epoch % self.train_args["TEST_INTERVAL"] == 0):
                self.test()
            self.scheduler.step()  # A step : An epoch
            log_text = "----- Epoch : {} -----".format(self.epoch)
            print(log_text)
            for param_group in self.optimizer.param_groups:
                # Current learning rate ...
                print("Learning rate : ", param_group['lr'])
                self.writer.add_scalar(
                    'LearningRate', param_group['lr'], self.epoch)

            epoch_val_sum = 0

            for i in range(5):  # 5 folds...
                log_text = "---> Fold : {}".format(i)
                print(log_text)
                for f in self.tr_sequence[i]:  # Iterate for folds...
                    if f is not i:  # Training Set
                        self.model.train()  # Set model to training mode
                    else:  # Validation Set
                        self.model.eval()   # Set model to evaluation mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data for current set of the fold.
                    for inputs, labels in self.loader.train_dataloader[f]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(f is not i):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self.loss(outputs, labels)

                            # backward + optimize only if in training phase
                            if f is not i:
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / self.loader.dataset_sizes[f]
                    epoch_acc = running_corrects.double(
                    ) / self.loader.dataset_sizes[f]

                    if(f == i):
                        log_text = 'Validation({}) - Loss: {:.4f}, Acc: {:.4f}'.format(
                            f, epoch_loss, epoch_acc)
                        print(log_text)
                        self.writer.add_scalar(
                            f'Validation{f}/Loss', epoch_loss, self.epoch)
                        self.writer.add_scalar(
                            f'Validation{f}/Accuracy', epoch_acc, self.epoch)
                        epoch_val_sum = epoch_val_sum + epoch_acc
                    else:
                        log_text = 'Training({}) - Loss: {:.4f}, Acc: {:.4f}'.format(
                            f, epoch_loss, epoch_acc)
                        print(log_text)
                        self.writer.add_scalar(
                            f'Training{f}/Loss', epoch_loss, self.epoch)
                        self.writer.add_scalar(
                            f'Training{f}/Accuracy', epoch_acc, self.epoch)

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_since
            log_text = "-"*20
            print(log_text)
            log_text = 'Epoch time : {:.0f}m {:.0f}s'.format(
                epoch_time // 60, epoch_time % 60)
            print(log_text)
            estimated_time = (self.num_epochs - self.epoch)*epoch_time
            log_text = 'Estimated time ({:.0f}/{:.0f}) : {:.0f}m {:.0f}s'.format(
                self.epoch,
                self.num_epochs,
                estimated_time // 60,
                estimated_time % 60)
            print(log_text)

            epoch_val = epoch_val_sum/5
            if epoch_val > self.best_acc:
                self.best_acc = epoch_val
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(f"_best_{self.best_acc}")
            log_text = "Epoch {} accuracy : {}".format(self.epoch, epoch_val)
            print(log_text)
            self.writer.add_scalar(f'Epoch/Accuracy', epoch_val, self.epoch)

        time_elapsed = time.time() - since
        log_text = 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60)
        print(log_text)
        log_text = 'Best val Acc: {:4f}'.format(self.best_acc)
        print(log_text)

        self.model.load_state_dict(self.best_model_wts)
        return self.model

    def per_test(self):
        loss_value = []
        result_frag = []
        label_frag = []
        self.summary_statistics.reset()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data for current set of the fold.
        for inputs, labels in self.loader.test_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.eval()

            # inference
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            loss = self.loss(outputs, labels)

            result_frag.append(preds.data.cpu().numpy())
            loss_value.append(loss.item())
            label_frag.append(labels.data.cpu().numpy())

            # statistics
            self.summary_statistics.update(labels.data.cpu().numpy(),
                                           preds.data.cpu().numpy().astype(int))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.loader.test_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(self.loader.test_dataloader.dataset)

        print("-"*20)
        log_text = 'Test - Loss: {:.4f}, Acc: {:.4f}'.format(
            epoch_loss, epoch_acc)
        print(log_text)
        print("-"*20)
        self.writer.add_scalar(
            'Test/Loss', epoch_loss, self.epoch)
        self.writer.add_scalar(
            'Test/Accuracy', epoch_acc, self.epoch)

    def test(self):
        self.per_test()

        result_summary = self.summary_statistics.get_metrics()

        file_name = 'test_result_' + str(self.epoch)
        save_file = os.path.join(self.out_dir.result_dir, file_name+'.pkl')
        save_file_summary = os.path.join(
            self.out_dir.result_dir, file_name + '_summary.txt')
        save_file_summary_yaml = os.path.join(
            self.out_dir.result_dir, file_name + '_summary.yaml')
        save_file_confusion = os.path.join(
            self.out_dir.result_dir, file_name + '_confusion.csv')
        np.savetxt(save_file_confusion,
                   result_summary['conf_matrix'], delimiter=',')
        with open(save_file_summary, 'w') as handle:
            handle.write(pprint.pformat(result_summary))
