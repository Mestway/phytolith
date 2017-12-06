import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def train(net, train_data, dev_data, vocab, params, cuda_enable):

    if cuda_enable:
        net = net.cuda()

    epoch_num = params["epoch_num"]
    learning_rate = params["learning_rate"]

    loss_fn =  nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
                #optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    train_acc = []
    dev_acc = []
    best_acc_so_far = 0.

    for epoch in range(epoch_num):  # loop over the dataset multiple times

        losses = []

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

            if params["tree_prediction"]:
                r1 = len(vocab["subfamily"])
                loss1 = loss_fn(outputs[:,0:r1], labels[:,0])
                r2 = r1 + len(vocab["tribe"])
                loss2 = loss_fn(outputs[:,r1:r2], labels[:,1])
                r3 = r2 + len(vocab["genus"])
                loss3 = loss_fn(outputs[:,r2:r3], labels[:,2])
                loss = loss1 + loss2 + loss3
            else:
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            
            losses.append(loss.data[0])

        t_acc = round(test(net, train_data, vocab, cuda_enable, params["tree_prediction"]), 3)
        d_acc = round(test(net, dev_data, vocab, cuda_enable, params["tree_prediction"]), 3)

        print(".", end="", flush=True)
        if d_acc > best_acc_so_far:
            print("[{}]".format(d_acc), end="", flush=True)
            best_acc_so_far = d_acc
        #print('[epoch %d] loss: %.3f, train_acc: %.3f, dev_acc: %.3f' 
        #        % (epoch + 1, np.mean(losses), t_acc, d_acc))

        train_acc.append(t_acc)
        dev_acc.append(d_acc)

    print("")

    #print('Finished Training')
    return train_acc, dev_acc


def test(net, test_data, vocab, cuda_enable, tree_prediction=False, test_index=2):

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

        if tree_prediction:
            if test_index == 0:
                l = 0
                r = l + len(vocab["subfamily"])
            elif test_index == 1:
                l = len(vocab["subfamily"])
                r = l + len(vocab["tribe"])
            elif test_index == 2:
                l = len(vocab["subfamily"]) + len(vocab["tribe"])
                r = l + len(vocab["genus"])
            outputs = outputs[:,l:r]
            labels = labels[:,test_index]
        else:
            outputs = outputs

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return float(correct) / total
