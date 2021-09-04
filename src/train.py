import os
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to the training data')
args = parser.parse_args()

# Data Loading
transform_image_train = transforms.Compose([transforms.ToTensor(), transforms.RandomAffine(degrees=0, translate=(0.1, 0)),
                                            transforms.RandomAffine(degrees=0, translate=(0.15, 0)),
                                            transforms.RandomAffine(degrees=0, translate=(0.2, 0)),

                                            transforms.RandomAffine(degrees=0, translate=(0, 0.1)),
                                            transforms.RandomAffine(degrees=0, translate=(0, 0.15)),
                                            transforms.RandomAffine(degrees=0, translate=(0, 0.2)),

                                            transforms.RandomRotation(degrees=(-3, 3), expand=False, fill=0),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_image_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                       std=[0.229, 0.224, 0.225])])

batch_size = 512

trainset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=transform_image_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, 'validation'), transform=transform_image_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = tuple(os.listdir(os.path.join(args.data_path, 'train')))

net = torchvision.models.resnet34(pretrained=True)

for param in net.parameters():
    param.requires_grad = False

num_ftrs = net.fc.in_features
net.fc = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, len(classes))
)

net = net.to(device)

# Optimizer
l1_crit = nn.L1Loss()
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.0007, momentum=0.9, weight_decay=0.005)
factor = 0.01

# Training CNN
for epoch in range(500):
    print('Starting epoch {}'.format(epoch))

    train_loss = 0
    correct = 0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs).to(device)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)

        regularization_loss = 0
        for param in net.fc.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        loss += factor * regularization_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss_epoch = train_loss / len(trainset)
    # print('Train error for epoch {} : {}'.format(epoch, train_loss_epoch))
    print('Train Accuracy for epoch {} : {}'.format(epoch, correct / len(trainset)))

    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            # print('labels : {}'.format(labels))
            # print('preds : {}'.format(predicted))
            # print('-------------------------------------------')
            correct += (predicted == labels).sum().item()

    print('Test Accuracy for epoch {} : {}'.format(epoch, correct / len(testset)))
