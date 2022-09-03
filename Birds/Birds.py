import os
import random
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.models as models
from get_tree_target import *

BATCH_SIZE = 64
nb_epoch = 100
learning_rate = 0.01
hidden_number = 256
criterion = nn.CrossEntropyLoss()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # inplanes = 64, planes = 64, stride=1, downsample=None

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        

class SimpleFPN(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(SimpleFPN, self).__init__()
        self.channels_cond = in_planes
        # Master branch
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # b是batch_size，因为进网络是一批一批进去的
        # Master branch
        x_master = self.conv_master(x)
        # Global pooling branch
        out = x_master
        return out


class PyramidFeatures(nn.Module):
    """Feature pyramid module with top-down feature pathway"""

    def __init__(self, B3_size, B4_size, B5_size, feature_size=256):
        # B3_size = 512, B4_size = 1024, B5_size = 2048
        super(PyramidFeatures, self).__init__()

        self.P5_1 = SimpleFPN(B5_size, feature_size)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(B4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(B3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        B3, B4, B5 = inputs

        P5_x = self.P5_1(B5)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(B4)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(B3)
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class ResNet(nn.Module):

    # block为Bottleneck, layers为[3, 4, 6, 3]
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 池化后 channel = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)
        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)
        self.max4 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.max5 = nn.MaxPool2d(kernel_size=7, stride=7)

        fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels,
                     self.layer2[layers[1] - 1].conv3.out_channels,
                     self.layer3[layers[2] - 1].conv3.out_channels,
                     self.layer4[layers[3] - 1].conv3.out_channels]
        self.fpn = PyramidFeatures(fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

        self.num_ftr2 = 256
        self.num_ftr3 = 256
        self.num_ftr4 = 256
        self.num_ftr5 = 2048

        self.feature2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftr2),
            nn.Linear(self.num_ftr2, hidden_number),
            nn.BatchNorm1d(hidden_number),
            nn.ELU(inplace=True),
        )
        self.feature3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftr3),
            nn.Linear(self.num_ftr3, hidden_number),
            nn.BatchNorm1d(hidden_number),
            nn.ELU(inplace=True),
        )
        self.feature4 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftr4),
            nn.Linear(self.num_ftr4, hidden_number),
            nn.BatchNorm1d(hidden_number),
            nn.ELU(inplace=True),
        )
        self.feature5 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftr5),
            nn.Linear(self.num_ftr5, hidden_number),
            nn.BatchNorm1d(hidden_number),
            nn.ELU(inplace=True),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(self.num_ftr2, 13),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.num_ftr3, 38),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(self.num_ftr4, 200),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(hidden_number, 200),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # block为Bottleneck, planes = 64, block = 3
        # block为Bottleneck, planes = 128, block = 4, stride = 2
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, targets):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        f2, f3, f4 = self.fpn([x2, x3, x4])
        # layer 2
        f2 = self.max2(f2)
        f2 = f2.view(f2.size(0), -1)
        f2 = self.feature2(f2)
        # layer 3
        f3 = self.max3(f3)
        f3 = f3.view(f3.size(0), -1)
        f3 = self.feature3(f3)
        # layer 4
        f4 = self.max4(f4)
        f4 = f4.view(f4.size(0), -1)
        f4 = self.feature4(f4)
        # Output
        x4 = self.max5(x4)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.feature5(x4)

        order_targets, family_targets = get_order_family_target(targets)

        order_output = self.classifier2(f2)
        ce_loss_order = criterion(order_output, order_targets)

        family_output = self.classifier3(f3)
        ce_loss_family = criterion(family_output, family_targets)

        species_out = self.classifier4(f4)
        ce_loss_spec = criterion(species_out, targets)

        species_output = self.classifier5(x4)
        ce_loss_species = criterion(species_output, targets)

        ce_loss = ce_loss_order + ce_loss_family + ce_loss_spec + ce_loss_species

        return ce_loss, [species_output, targets], [family_output, family_targets], \
               [order_output, order_targets]


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def train(epoch, net, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    order_total = 0
    family_total = 0
    species_total = 0
    order_correct = 0
    family_correct = 0
    species_correct = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        idx = batch_idx + 1
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        ce_loss, [species_output, species_targets], [family_output, family_targets], \
        [order_output, order_targets] = net(inputs, targets)
        loss = ce_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, order_predicted = torch.max(order_output.data, 1)
        order_total += order_targets.size(0)
        order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

        _, family_predicted = torch.max(family_output.data, 1)
        family_total += family_targets.size(0)
        family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

        _, species_predicted = torch.max(species_output.data, 1)
        species_total += species_targets.size(0)
        species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()

    train_order_acc = 100. * order_correct / order_total
    train_family_acc = 100. * family_correct / family_total
    train_species_acc = 100. * species_correct / species_total
    train_loss = train_loss / idx
    print('Epoch %d, train_order_acc = %.5f, train_family_acc = %.5f, train_species_acc = %.5f, '
          'train_loss = %.6f' % (epoch, train_order_acc, train_family_acc, train_species_acc, train_loss))
    return train_order_acc, train_family_acc, train_species_acc, train_loss


def test(epoch, net, testloader):
    net.eval()
    test_loss = 0
    order_total = 0
    family_total = 0
    species_total = 0
    order_correct = 0
    family_correct = 0
    species_correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            idx = batch_idx + 1
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            ce_loss, [species_output, species_targets], [family_output, family_targets], \
            [order_output, order_targets] = net(inputs, targets)
            test_loss += ce_loss.item()

            _, order_predicted = torch.max(order_output.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, family_predicted = torch.max(family_output.data, 1)
            family_total += family_targets.size(0)
            family_correct += family_predicted.eq(family_targets.data).cpu().sum().item()

            _, species_predicted = torch.max(species_output.data, 1)
            species_total += species_targets.size(0)
            species_correct += species_predicted.eq(species_targets.data).cpu().sum().item()

    test_order_acc = 100. * order_correct / order_total
    test_family_acc = 100. * family_correct / family_total
    test_species_acc = 100. * species_correct / species_total
    test_loss = test_loss / idx
    print('Epoch %d, test_order_acc = %.5f, test_family_acc = %.5f, test_species_acc = %.5f, '
          'test_loss = %.6f' % (epoch, test_order_acc, test_family_acc, test_species_acc, test_loss))
    return test_order_acc, test_family_acc, test_species_acc, test_loss


def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % nb_epoch)
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1
    return float(learning_rate / 2 * cos_out)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def main():
    # preparing data
    setup_seed(20)
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = torchvision.datasets.ImageFolder(root=r'/data/chenjunhan/Cross-Granularity/Birds/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                              drop_last=True)

    testset = torchvision.datasets.ImageFolder(root=r'/data/chenjunhan/Cross-Granularity/Birds/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                             drop_last=True)
    print('==> Successfully Preparing data..')

    # building model
    print('==> Building model..')
    net = resnet50(num_classes=200)
    if torch.cuda.is_available():
        net.cuda()
        # net.fpn = torch.nn.DataParallel(net.fpn)
        # net.feature2 = torch.nn.DataParallel(net.feature2)
        # # net.feature3 = torch.nn.DataParallel(net.feature3)
        # # net.feature4 = torch.nn.DataParallel(net.feature4)

        # net.classifier2 = torch.nn.DataParallel(net.classifier2)
        # net.classifier3 = torch.nn.DataParallel(net.classifier3)
        # net.classifier4 = torch.nn.DataParallel(net.classifier4)
        cudnn.benchmark = True
    # net_dict = net.state_dict()
    # pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    # for k, v in net.named_parameters():
    #     print(k)
    # net.load_state_dict(pretrained_dict, strict=False)
    pretrained_resnet = models.resnet50(pretrained=True)
    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print('==> Successfully Building model..')

    # building optimizer
    print('==> Building optimizer..')
    optimizer = optim.SGD([
        {'params': nn.Sequential(*list(net.children())[:-13]).parameters(), 'lr': learning_rate / 10},
        # {'params': nn.Sequential(*list(net.children())[7:]).parameters(), 'lr': learning_rate}],
        {'params': net.fpn.parameters(), 'lr': learning_rate},
        {'params': net.feature2.parameters(), 'lr': learning_rate},
        {'params': net.feature3.parameters(), 'lr': learning_rate},
        {'params': net.feature4.parameters(), 'lr': learning_rate},
        {'params': net.feature5.parameters(), 'lr': learning_rate},
        {'params': net.classifier2.parameters(), 'lr': learning_rate},
        {'params': net.classifier3.parameters(), 'lr': learning_rate},
        {'params': net.classifier4.parameters(), 'lr': learning_rate},
        {'params': net.classifier5.parameters(), 'lr': learning_rate}],
        momentum=0.9, weight_decay=5e-4)
    # print("-15")
    # print(list(net.children())[-15])
    # print("-14")
    # print(list(net.children())[-14])
    # print("-13")
    # print(list(net.children())[-13])
    # print("-12")
    # print(list(net.children())[-12])
    # print("-11")
    # print(list(net.children())[-11])
    # print("-10")
    # print(list(net.children())[-10])
    # print('==> Successfully Building optimizer..')

    # optimizer = optim.Adam([
    #     {'params': nn.Sequential(*list(net.children())[:-10]).parameters(), 'lr': learning_rate / 10},
    #     # {'params': nn.Sequential(*list(net.children())[7:]).parameters(), 'lr': learning_rate}],
    #     {'params': net.fpn.parameters(), 'lr': learning_rate},
    #     {'params': net.feature2.parameters(), 'lr': learning_rate},
    #     {'params': net.feature3.parameters(), 'lr': learning_rate},
    #     {'params': net.feature4.parameters(), 'lr': learning_rate},
    #     {'params': net.feature5.parameters(), 'lr': learning_rate},
    #     {'params': net.classifier2.parameters(), 'lr': learning_rate},
    #     {'params': net.classifier3.parameters(), 'lr': learning_rate},
    #     {'params': net.classifier4.parameters(), 'lr': learning_rate},
    #     {'params': net.classifier5.parameters(), 'lr': learning_rate}],
    #     betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # print('==> Successfully Building optimizer..')

    for epoch in range(nb_epoch):
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch) / 10
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[2]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[3]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[4]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[5]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[6]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[7]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[8]['lr'] = cosine_anneal_schedule(epoch)
        optimizer.param_groups[9]['lr'] = cosine_anneal_schedule(epoch)
        train(epoch, net, trainloader, optimizer)
        test(epoch, net, testloader)


if __name__ == '__main__':
    main()
