import matplotlib.pyplot as plt
import csv

def plot_acc(train_acc, test_acc):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('number of epochs')
    ax.set_ylabel('accuracies')
    line1, = plt.plot([i for i in range(len(train_acc))], train_acc, "b-", label='train_acc')
    line2, = plt.plot([i for i in range(len(train_acc))], test_acc, "r-", label='dev_acc')
    plt.ylim(0, 1.)
    plt.legend()

def plot_from_file(file_name):

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

	plot_acc(train_acc, test_acc)

if __name__ == '__main__':
	plot_from_file("output/result.log")
	plt.show()