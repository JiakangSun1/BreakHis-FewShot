from CBAM import *
import torchvision.models as models
# from our_train import parser
# args = parser.parse_args()
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels, 4)
        # 如果输入和输出通道数不一致，使用1x1卷积进行下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 如果有下采样，将输入进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        return out
class ResnetwithCBAM(nn.Module):
    def __init__(self):
        super(ResnetwithCBAM, self).__init__()
        self.layer2 = BasicBlock(512,128,1)
        self.layer3 = BasicBlock(128,256,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class VGGInceptionModule(nn.Module):
    def __init__(self, in_channels, f1, f2, f3):
        super(VGGInceptionModule, self).__init__()
        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, padding=0)
        # 3x3 convolution
        self.conv3 = nn.Conv2d(in_channels, f2, kernel_size=3, padding=1)
        # 5x5 convolution
        self.conv5 = nn.Conv2d(in_channels, f3, kernel_size=5, padding=2)
        # 3x3 max pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.res=ResnetwithCBAM()
    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv3 = F.relu(self.conv3(x))
        conv5 = F.relu(self.conv5(x))

        pool = self.pool(x)

        res=self.res(x)

        out = torch.cat((conv1, conv3, conv5, pool,res), 1)  # Concatenate along the channels dimension
        return out


class VGGInceptionNet(nn.Module):
    def __init__(self):
        super(VGGInceptionNet, self).__init__()

        torch.hub.set_dir('/para')
        self.vgg16=models.vgg16(pretrained=True)
        classifier = nn.Sequential()
        for layer in range(24, 31):
            self.vgg16.features[layer] = classifier
        self.vgg16.features.requires_grad_(False)
        # Custom Inception module
        self.inception_module = VGGInceptionModule(512, 64, 128, 32)

        # Classifier
        self.conv2d = nn.Conv2d(in_channels=736+256, out_channels=512, kernel_size=1)
        self.bn=nn.BatchNorm2d(512)
        self.bn1=nn.BatchNorm2d(1)
        self.flatten=nn.Flatten()
        self.dp=nn.Dropout(0.3)
        self.fn=nn.Linear(512, 44)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=3)
        self.avgpool1=nn.AvgPool2d(kernel_size=3,stride=3)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)

        self.fn1=nn.Linear(196,8)
        self.fn2=nn.Linear(512,8)
    def forward(self, x):
        x=self.vgg16.features(x)

        x = self.inception_module(x)
        x=self.conv2d(x)#调整为512大小

        x1 = x.view(x.shape[0], 512, -1)#（512,196）
        x2 = torch.transpose(x1,1,2)#（196,512）
        #通道融合
        cha=torch.matmul(x1,x2)
        cha_value = torch.zeros(cha.shape[0], cha.shape[1]).cuda()
        # 遍历第一个维度
        for i in range(cha.shape[0]):
            cha_value[i] = torch.linalg.eigvals(cha[i])
        cha_value = self.dp(cha_value)
        cha_value = F.normalize(self.fn2(cha_value))

        #空间融合
        spa = torch.matmul(x2, x1)
        spa_value = torch.zeros(spa.shape[0], spa.shape[1]).cuda()
        # 遍历第一个维度
        for i in range(spa.shape[0]):
            spa_value[i] = torch.linalg.eigvals(spa[i])
        spa_value = self.dp(spa_value)
        spa_value = F.normalize(self.fn1(spa_value))

        x=F.relu(x)
        x=self.bn(x)
        x=self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.dp(x)
        x=self.fn(x)

        return torch.cat((x,cha_value,spa_value), dim=1)
