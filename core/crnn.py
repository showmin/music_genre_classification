import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, p=0.75):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # self.embedding = nn.Linear(nHidden * 2, nOut)
        # self.embedding = nn.Linear(123, nOut)
        # self.embedding = nn.LazyLinear(nOut)
        self.embedding = nn.Sequential(
          nn.Linear(31744, 512),
          nn.ReLU(),
          nn.BatchNorm1d(512),
          nn.Dropout(p=p),
          nn.Linear(512, nOut)
        )

    def forward(self, input):
        # print('=== input size:', input.shape)
        recurrent, _ = self.rnn(input)
        # print('=== after rnn:', recurrent.shape)
        T, b, h = recurrent.size() # [186,32,512]
        # t_rec = recurrent.view(T * b, h)
        t_rec = recurrent.permute(1,0,2) # [32,56,512]
        # print('=== t_rec1:', t_rec.shape)
        t_rec = t_rec.reshape(b,-1)
        # print('=== t_rec2:', t_rec.shape)

        output = self.embedding(t_rec)  # [T * b, nOut]
        # print('=== final:', t_rec.shape)
        # output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):

    def __init__(self, data_shape, nc=1, nclass=10, nh=256, n_rnn=2, 
                leakyRelu=False, p=0):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False, p=0):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

            if p > 0:
                print('crnn dropout:', p)
                cnn.add_module('drop{}'.format(i), nn.Dropout2d(p=p))

        convRelu(0, p=p)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1, p=p)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True, p=p)
        convRelu(3, p=p)
        cnn.add_module('pooling{0}'.format(2),
                        nn.MaxPool2d((2, 2), (2, 1)))  # 256x4x16
                       #nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True, p=p)
        # convRelu(5, p=p)
        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        # convRelu(6, True, p=p)  # 512x1x16

        self.cnn = cnn
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(512, nh, nh),
        #     BidirectionalLSTM(nh, nh, nclass))
        self.rnn = BidirectionalLSTM(512, nh, nclass)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # print('cnn size: ', conv.size())
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        # conv = conv.squeeze(2)
        # conv = conv.permute(2, 0, 1)  # [w, b, c]
        conv = conv.view(b,c,-1,1)
        # print('cnn size: ', conv.size())
        conv = conv.squeeze(3)
        # print('cnn size: ', conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # print('cnn size: ', conv.size())


        # rnn features
        output = self.rnn(conv)

        return output