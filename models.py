import torch
import torch.nn as nn
import torch.nn.functional as F
class hSwish(nn.Module):
    """
    h-swish激活函数
    """
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class Swish(nn.Module):
    """
    Swish激活函数
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class NewSEBlock(nn.Module):
    """
    新的SEBlock
    """
    def __init__(self, channel, reduction=16):
        super(NewSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 进行均值池化
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        # 注意：注意力机制的第一层命名为 fc
        self.swish = Swish()
        # 函数激活
        self.fc2 = nn.Linear(channel // reduction + channel, channel, bias=False)
        # 注意：注意力机制的第二层命名为fc2
        self.sigmoid = nn.Sigmoid()
        # 注意：注意力机制的激活函数命名为sigmoid

    def forward(self, x):
        b, c, _, _ = x.size()
        # b是batch_size，c是通道数
        y = self.avg_pool(x).view(b, c)
        # 进行均值池化

        y = self.fc1(y)
        # fc1层级训练
        y = self.swish(y)
        # 激活函数


        x_compressed = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        # 将fc1的输出和原始x进行拼接
        # 需要确保x在拼接维度上的尺寸与y匹配
        # 这里我们将x压缩到与y相同的尺寸

        combined = torch.cat((y, x_compressed), dim=1)
        # 将fc1的输出和原始x进行拼接

        y = self.fc2(combined)
        # fc2层级训练
        y = self.sigmoid(y).view(b, c, 1, 1)
        # 激活函数

        return x * y.expand_as(x)


class BnFire5x5(nn.Module):
    """
    Fire模块
    """
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, expand5x5_channels):
        super(BnFire5x5, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=False)
        # squeeze层级输入通道数，输出通道数，卷积核大小，是否使用偏置
        self.batch_norm = nn.BatchNorm2d(squeeze_channels)
        # 批标准化也叫正则化层级将上层的输出进行标准化到均值为0方差为1
        self.relu = nn.ReLU(inplace=True)
        # 激活函数
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, bias=False)
        # expand1x1层级输入通道数，输出通道数，卷积核大小，是否使用偏置
        self.batch_norm1 = nn.BatchNorm2d(expand1x1_channels)
        # 批标准化也叫正则化层级将上层的输出进行标准化到均值为0方差为1
        self.squeeze_activate = nn.ReLU(inplace=True)
        # 激活函数
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1, bias=False)
        # expand3x3层级输入通道数，输出通道数，卷积核大小3x3，是否使用偏置
        self.batch_norm3 = nn.BatchNorm2d(expand3x3_channels)
        # 批标准化也叫正则化层级将上层的输出进行标准化到均值为0方差为1
        self.squeeze3_activate = nn.ReLU(inplace=True)
        # 激活函数
        self.expand5x5 = nn.Conv2d(squeeze_channels, expand5x5_channels, kernel_size=5, padding=2, bias=False)
        # expand5x5层级输入通道数，输出通道数，卷积核大小5x5，是否使用偏置
        self.batch_norm5 = nn.BatchNorm2d(expand5x5_channels)
        # 批标准化也叫正则化层级将上层的输出进行标准化到均值为0方差为1
        self.squeeze5_activate = nn.ReLU(inplace=True)
        # 激活函数
    def forward(self, x):
        x = self.squeeze(x)
        # squeeze层级输入
        x = self.batch_norm(x)
        # 批标准化
        x = self.relu(x)
        # 激活函数
        cat_x_1 = torch.cat([self.squeeze_activate(self.batch_norm1(self.expand1x1(x))),
                             self.squeeze3_activate(self.batch_norm3(self.expand3x3(x)))], 1)
        # 将expand1x1和expand3x3的输出进行拼接
        x = torch.cat([cat_x_1, self.squeeze5_activate(self.batch_norm5(self.expand5x5(x)))], 1)
        # 将expand1x1和expand3x3的输出和expand5x5的输出进行拼接
        return self.relu(x) # 激活函数


class miniSplicing_SEsqueezenet(nn.Module):
    """
    也是 mini-SSEsqueezenet 这里考虑文件调用问题我就不改名字了
    """
    def __init__(self, num_classes=10):
        super(miniSplicing_SEsqueezenet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2),
            # 输入通道数，输出通道数，卷积核大小，步长 作为输入层
            NewSEBlock(96),
            # 新的SEBlock做一次注意力机制
            nn.SELU(inplace=True),
            # 激活函数
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # 最大池化层级
            nn.BatchNorm2d(96),
            # 批标准化
            BnFire5x5(96, 16, 64, 64, 64),
            # Fire模块引入
            nn.BatchNorm2d(192),
            # 批标准化
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # 最大池化层级
            BnFire5x5(192, 32, 128, 128, 128),
            # Fire模块引入
            BnFire5x5(384, 48, 192, 192, 192),
            # Fire模块引入
            nn.BatchNorm2d(576),
            # 批标准化
        )
        self.classifier = nn.Sequential(
            # 分类器
            nn.Dropout(p=0.3),
            # 随机失活
            nn.Conv2d(576, num_classes, kernel_size=1),
            # 卷积层级
            nn.Softmax(dim=1),
            # 激活函数
            nn.AdaptiveAvgPool2d(1)
            # 自适应平均池化
        )
        self._initialize_weights()
        # he 权重初始化

    def forward(self, x):
        x = self.features(x)
        # 特征提取
        x = self.classifier(x)
        # 分类器
        return x.view(x.size(0), -1)

    def _initialize_weights(self):
        """
        He初始化
        He初始化，也称作He正态初始化，是针对ReLU激活函数（及其变种如Leaky ReLU）优化的。He初始化以0为中心，其标准差为
        sqrt(2/n)，其中
        n是权重矩阵中的输入单元数。在PyTorch中，你可以使用以下方式应用He初始化：
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier[1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Splicing_SEsqueezenet(nn.Module):
    def __init__(self, num_classes=1000, ):
        super(Splicing_SEsqueezenet, self).__init__()
        """
        在每个Fire模块之后：将SE模块加在每个Fire模块之后，可以让网络在每个阶段都进行特征重标定。这种方式能够使得网络在每个Fire模块处理完数据后立即对特征通道进行优化，可能会带来更细致的性能提升。

在最大池化层之前：在每个最大池化层之前加入SE模块，可以在降维之前对特征进行重标定。这种方式可能有助于在特征降维前保留更多重要信息。

在最终Fire模块之后：如果您希望简化模型的修改，可以选择在最后一个Fire模块之后加入一个SE模块。这种方式能够在不大幅增加计算负担的情况下，为整个网络的输出提供一个全局的特征重标定。
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2),
            NewSEBlock(96), # 这里独特说明，这里的SEBlock是在卷积层之后也是在输入层之后
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(96),
            BnFire5x5(96, 16, 64, 64, 64),
            BnFire5x5(192, 16, 64, 64, 64),
            BnFire5x5(192, 32, 128, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(384),
            BnFire5x5(384, 32, 128, 128, 128),
            BnFire5x5(384, 48, 192, 192, 192),
            BnFire5x5(576, 48, 192, 192, 192),
            BnFire5x5(576, 64, 256, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(768),
            BnFire5x5(768, 64, 256, 256, 256),
        )

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(768, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            # 增加一层seblock
            # AvgSEBlock(num_classes),
            nn.Softmax(dim=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """
        He初始化
        He初始化，也称作He正态初始化，是针对ReLU激活函数（及其变种如Leaky ReLU）优化的。He初始化以0为中心，其标准差为
        sqrt(2/n)，其中
        n是权重矩阵中的输入单元数。在PyTorch中，你可以使用以下方式应用He初始化：
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier[1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # 特征提取
        x = self.classifier(x)
        # 分类器
        return x.view(x.size(0), -1)



if __name__ == '__main__':
    # 测试
    model = Splicing_SEsqueezenet()
    print(model)
    model = miniSplicing_SEsqueezenet()
    print(model)



