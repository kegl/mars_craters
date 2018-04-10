from __future__ import division, print_function

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import multivariate_normal
from rampwf.workflows.image_classifier import get_nb_minibatches

is_cuda = torch.cuda.is_available()
factor = 100000


class ObjectDetector(object):
    def __init__(self):
        self.net = Net()

    def fit(self, X, y):
        # torch.load('model.pt')
        shape = X.shape[1:]
        print('Computing target map...')
        y = np.array([gaussian_detection_map(yi, shape) for yi in y])
        batch_size = 16
        nb_epochs = 30
        lr = 1e-3
        valid_ratio = 0.05

        if is_cuda:
            self.net = self.net.cuda()
        net = self.net
        optimizer = optim.Adam(net.parameters(), lr=lr)
        nb_valid = int(valid_ratio * len(X))
        nb_train = len(X) - nb_valid
        nb_train_minibatches = get_nb_minibatches(nb_train, batch_size)
        criterion = nn.MSELoss()
        if is_cuda:
            criterion = criterion.cuda()

        for epoch in range(nb_epochs):
            if epoch % 10 == 0:
                lr /= 10
            print('learning rate =', lr)
            t0 = time.time()
            net.train()  # train mode
            nb_trained = 0
            nb_updates = 0
            train_loss = []
            train_mae = []
            train_rmse_n = []
            train_err_n = []
            X_train = X[:nb_train]
            X_valid = X[nb_train:]
            y_train = y[:nb_train]
            y_valid = y[nb_train:]
            for i in range(0, len(X_train), batch_size):
                net.train()  # train mode
                idxs = slice(i, i + batch_size)
                X_minibatch = X_train[idxs]
                X_minibatch = self._make_X_minibatch(X_minibatch)
                y_minibatch = y_train[idxs]
                y_minibatch = _make_variable(y_minibatch)
                # zero-out the gradients because they accumulate by default
                optimizer.zero_grad()
                y_minibatch_pred = self._predict_map_torch(X_minibatch)
                loss = criterion(y_minibatch_pred, y_minibatch)
                loss.backward()  # compute gradients
                optimizer.step()  # update params

                # Loss and accuracy
                train_mae.append(
                    self._get_mae_torch(y_minibatch_pred, y_minibatch))
                train_rmse_n.append(
                    self._get_rmse_n_torch(y_minibatch_pred, y_minibatch))
                train_err_n.append(
                    self._get_err_n_torch(y_minibatch_pred, y_minibatch))
                train_loss.append(loss.data[0])
                nb_trained += X_minibatch.size(0)
                nb_updates += 1
                if nb_updates % 100 == 0 or nb_updates == nb_train_minibatches:
                    print(
                        'Epoch [{}/{}], [trained {}/{}]'
                        ', avg_loss: {:.8f}'
                        ', avg_train_mae: {:.4f}'
                        ', avg_train_err_n: {:.4f}'
                        ', avg_train_rmse_n: {:.4f}'.format(
                            epoch + 1, nb_epochs, nb_trained, nb_train,
                            np.mean(train_loss), np.mean(train_mae),
                            np.mean(train_err_n), np.mean(train_rmse_n)))

            torch.save(self.net.state_dict(), 'model.pt')
            net.eval()  # eval mode
            y_valid_pred = self._predict_map(X_valid)
            valid_mae = self._get_mae(y_valid_pred, y_valid)
            valid_err_n = self._get_err_n(y_valid_pred, y_valid)
            valid_rmse_n = self._get_rmse_n(y_valid_pred, y_valid)

            np.save('x.npy', X_valid)
            np.save('y.npy', y_valid)
            np.save('y_pred.npy', y_valid_pred)

            delta_t = time.time() - t0
            print('Finished epoch {}'.format(epoch + 1))
            print('Time spent : {:.4f}'.format(delta_t))
            print('Train mae : {:.4f}'.format(np.mean(train_mae)))
            print('Train err_n : {:.4f}'.format(np.mean(train_err_n)))
            print('Train rmse_n : {:.4f}'.format(np.mean(train_rmse_n)))
            print('Valid mae : {:.4f}'.format(np.mean(valid_mae)))
            print('Valid err_n : {:.4f}'.format(np.mean(valid_err_n)))
            print('Valid rmse_n : {:.4f}'.format(np.mean(valid_rmse_n)))

    def _make_X_minibatch(self, X_minibatch):
        X_minibatch = np.expand_dims(X_minibatch, axis=1)
        X_minibatch = _make_variable(X_minibatch.astype(np.float32))
        return X_minibatch

    def _get_mae_torch(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy()
        y_true = y_true.cpu().data.numpy()
        return self._get_mae(y_pred, y_true)

    def _get_mae(self, y_pred, y_true):
        return np.sum(np.abs(y_pred - y_true), axis=0) / len(y_pred)

    def _get_rmse_n_torch(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy()
        y_true = y_true.cpu().data.numpy()
        return self._get_rmse_n(y_pred, y_true)

    def _get_rmse_n(self, y_pred, y_true):
        n_pred = np.sum(y_pred, axis=(1, 2))
        n_true = np.sum(y_true, axis=(1, 2))
        return np.sqrt(np.sum((n_true - n_pred) ** 2) / len(n_true)) / factor

    def _get_err_n(self, y_pred, y_true):
        n_pred = np.round(np.sum(y_pred, axis=(1, 2)) / factor + 1.0)
        n_true = np.round(np.sum(y_true, axis=(1, 2)) / factor)
        return np.mean(n_pred < n_true)

    def _get_err_n_torch(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy()
        y_true = y_true.cpu().data.numpy()
        return self._get_err_n(y_pred, y_true)

    def _predict_map_torch(self, X_minibatch):
        y_map_pred = self.net(X_minibatch)
        s = y_map_pred.size
        y_map_pred = y_map_pred.view(s(0), s(2), s(3))
        return y_map_pred

    def _predict_map(self, X):
        y_map_pred = np.empty(X.shape, dtype=np.float32)
        batch_size = 16
        for i in range(0, len(X), batch_size):
            idxs = slice(i, i + batch_size)
            X_minibatch = X[idxs]
            X_minibatch = self._make_X_minibatch(X_minibatch)
            y_map_pred[idxs] = self._predict_map_torch(
                X_minibatch).cpu().data.numpy()
        return y_map_pred

    def predict(self, X):
        y_map_pred = self._predict_map(X)
        y_pred_array = np.empty(len(y_map_pred), dtype=object)
        y_pred_array[:] = y_map_pred
        return y_pred_array


def _make_variable(X):
    variable = Variable(torch.from_numpy(X))
    if is_cuda:
        variable = variable.cuda()
    return variable


class Net(nn.Module):
    def __init__(self, w=224, h=224):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
#         self.block5 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
# #            nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.block6 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 3, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128,
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64,
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32,
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=1,
                kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block6(x)
        return x

    def _initialize_weights(self):
        # Source: https://github.com/pytorch/vision/blob/master/torchvision/
        # models/vgg.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def _flatten(x):
    return x.view(x.size(0), -1)


def single_gaussian_detection_map(gdm, pixels, cx, cy, r):
    next_gdm = multivariate_normal.pdf(
        pixels, mean=[cx, cy], cov=[[r, 0], [0, r]]).reshape(gdm.shape)
    next_gdm /= next_gdm.sum()
    next_gdm *= factor
    return gdm + next_gdm


def gaussian_detection_map(list_of_circles, shape):
    pixels = np.array(
        np.meshgrid(range(shape[0]), range(shape[1]))).T.reshape(-1, 2)
    gdm = np.zeros(shape, dtype=np.float32)
    gdm += np.sum([single_gaussian_detection_map(gdm, pixels, *y)
                   for y in list_of_circles], axis=0)
    return gdm
