import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self,
                 img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256):

        assert img_height % 16 == 0
        assert img_width % 4 == 0

        super(CRNN, self).__init__()
        self.cnn = nn.Sequential()

        self.cnn.add_module('conv0', nn.Conv2d(img_channel, 64, 3, 1, 1))
        self.cnn.add_module('relu0', nn.ReLU(inplace=True))
        self.cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))

        self.cnn.add_module('conv1', nn.Conv2d(64, 128, 3, 1, 1))
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.cnn.add_module('conv2', nn.Conv2d(128, 256, 3, 1, 1))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))

        self.cnn.add_module('conv3', nn.Conv2d(256, 256, 3, 1, 1))
        self.cnn.add_module('relu3', nn.ReLU(inplace=True))
        self.cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))

        self.cnn.add_module('conv4', nn.Conv2d(256, 512, 3, 1, 1))
        self.cnn.add_module('relu4', nn.ReLU(inplace=True))
        self.cnn.add_module('batchnorm4', nn.BatchNorm2d(512))

        self.cnn.add_module('conv5', nn.Conv2d(512, 512, 3, 1, 1))
        self.cnn.add_module('relu5', nn.ReLU(inplace=True))
        self.cnn.add_module('batchnorm5', nn.BatchNorm2d(512))
        self.cnn.add_module('pooling5', nn.MaxPool2d(kernel_size=(2, 1)))

        self.cnn.add_module('conv6', nn.Conv2d(512, 512, 2, 1, 0))
        self.cnn.add_module('relu6', nn.ReLU(inplace=True))

        self.map_to_seq = nn.Linear(512 * (img_height // 16 - 1),
                                    map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)

        return output  # shape: (seq_len, batch, num_class)
