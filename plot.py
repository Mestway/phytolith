import matplotlib.pyplot as plt
import csv
import os

def plot_acc(train_acc, test_acc, fig=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('number of epochs')
    ax.set_ylabel('accuracies')
    line1, = plt.plot([i for i in range(len(train_acc))], train_acc, "b-", label='train_acc')
    line2, = plt.plot([i for i in range(len(train_acc))], test_acc, "r-", label='dev_acc')
    plt.ylim(0, 1.)
    plt.legend()

def plot_from_tuples(ps):

    icons = ["r", 'g', "b", "y"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Accuracy')

    N = 50

    for i in range(len(ps)):
        plt.plot([i for i in range(N)], ps[i][0][:N], icons[i]+'-') #, label=ps[i][2] + "(train)"
        plt.plot([i for i in range(N)], ps[i][1][:N], icons[i]+'--', label=ps[i][2] + "(dev)")
    
    plt.ylim(0, 1.)
    plt.legend()



def load_from_file(file_name, name):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        train_acc, test_acc = [], []
        header = None
        for l in reader:
            if header == None:
                header = " ".join(l)
                continue
            train_acc.append(float(l[0]))
            test_acc.append(float(l[1]))
    return train_acc, test_acc, name

if __name__ == '__main__':

    fd = "result/synth"

    p1 = load_from_file(os.path.join(fd, "result_LinearModel.log"), "Linear")
    p2 = load_from_file(os.path.join(fd, "result_SimpleNet.log"), "FC")
    p3 = load_from_file(os.path.join(fd, "result_SimpleConvNet.log"), "Conv")
    p4 = load_from_file(os.path.join(fd, "result_ConvNet.log"), "ConvFC")

    plot_from_tuples([p1, p2, p3, p4])
    plt.show()
