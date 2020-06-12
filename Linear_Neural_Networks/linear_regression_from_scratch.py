from matplotlib import pyplot as plt
import torch
import random
import sys
sys.path.append('/home/flying-dutchman/PycharmProjects/')
from dl_functions import linear_regression, grad_descent, squared_loss

def use_svg_display():
    # Display in vector graphics
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # Set the size of the graph to be plotted
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    indices = range(len(features))
    # for id in range(batch_size):
    rand_id = torch.tensor(random.sample(range(0, len(features)), batch_size))
    yield features[rand_id], labels[rand_id]



def main():
    # generating dataset
    num_inputs = 2
    num_examples = 2000

    true_w = torch.Tensor([2, -3.4])
    true_b = 0.6

    features = torch.zeros(size=(num_examples, num_inputs)).normal_()
    labels = torch.matmul(features, true_w) + true_b
    labels += torch.zeros(size=(labels.shape)).normal_(std=0.01)

    # initialize model parameters from scratch

    w = torch.zeros(size=(num_inputs, 1)).normal_(0.01)
    b = torch.zeros(size=(1, ))

    w.requires_grad_(True)
    b.requires_grad_(True)

    lr = 0.03
    batch_size = 10
    epochs = 100
    net = linear_regression
    loss = squared_loss

    for epoch in range(epochs):
        for X, y_labels in data_iter(batch_size, features, labels):
            y_pred = linear_regression(X, w, b)
            loss_val = squared_loss(y_pred, y_labels)
            loss_val.mean().backward()  # computing gradient
            grad_descent([w, b], lr, batch_size)  # updating parameters
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print('epoch %d, loss %f' % (epoch + 1, train_l.mean().numpy()))

    print('Error in estimating w', true_w - w.reshape(true_w.shape))
    print('Error in estimating b', true_b - b)


if __name__ == '__main__':
  main()




