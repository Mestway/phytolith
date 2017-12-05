import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def tune_and_train(net_class, train_data, dev_data, params_space):

    best_param = None
    best_train_acc = None
    best_dev_acc = None

    for p in params_space:

        print("params: {}".format(p))

        nn = net_class(p)

        train_acc, dev_acc = train(nn, trainloader, testloader, p)
        if best_dev_acc is None or np.max(dev_acc) > np.max(best_dev_acc):
            best_param = p
            best_train_acc = train_acc
            best_dev_acc = dev_acc
            print("     train_acc={}".format(np.max(train_acc)))
            print("     dev_acc={}".format(np.max(dev_acc)))

    return best_param, best_train_acc, best_dev_acc


def train(net, train_data, dev_data, params, cuda_enable):

    if cuda_enable:
        net = net.cuda()

    epoch_num = params["epoch_num"]
    learning_rate = params["learning_rate"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
                #optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    train_acc = []
    dev_acc = []

    for epoch in range(epoch_num):  # loop over the dataset multiple times

        losses = []
        running_loss = 0.0
        cnt = 0

        for i, data in enumerate(train_data):

            # get the inputs
            inputs, labels = torch.from_numpy(data[0]), torch.from_numpy(data[1])
            if cuda_enable:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            #if True or epoch == 5:
                #print(outputs)
            #    _, pred = torch.max(outputs, 1)
                #print(pred.cpu().data.numpy())
                #print(labels.cpu().data.numpy())
                #print("")
            #if epoch == 50:
            #    sys.exit(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            cnt += 1
            # print statistics
            #losses.append(loss.data[0])
            
        print('[epoch %d] loss: %.3f' % (epoch + 1, running_loss / cnt))

        train_acc.append(test(net, train_data, cuda_enable))
        dev_acc.append(test(net, dev_data, cuda_enable))

    #print('Finished Training')
    return train_acc, dev_acc


def test(net, test_data, cuda_enable, show_example=False):

    if cuda_enable:
        net = net.cuda()

    correct = 0
    total = 0
    for data in test_data:
        
        images, labels = torch.from_numpy(data[0]), torch.from_numpy(data[1])
        
        if cuda_enable:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        if show_example:
            plt.imshow(np.mean(data[0][0], 0))
            print("Predicted: {}".format(predicted[0]))
            print("Ground Truth: {}".format(labels[0]))
            #print(vocab["genus"][data[0][1]])
            plt.gray()
            plt.show()
    
    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return float(correct) / total
