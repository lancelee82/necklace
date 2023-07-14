from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class ModelPartWrapper(nn.Module):  # TODO: moved to necklace.frmwrk.pytorch
    def __init__(self, submdls=[], md_device='cuda:0', mp_order=0):
        super(ModelPartWrapper, self).__init__()

        self.submdls = submdls
        self.md_device = md_device
        self.mp_order = mp_order

        submdls = [m.to(md_device) for m in submdls]
        net = nn.Sequential(*submdls)
        self.net = net.to(md_device)

        self.inp_cache = None
        self.out_cache = None

    def cache_graded_input(self, *args, **kwargs):
        if self.mp_order > 0:  # not the first submodule which is not need to get output gradients
            inputs = []
            for a in args:
                # NOTE: here the `a` may be a Tensor with requires_grad=True and so it is not a
                # leaf of graph and we can NOT save its grads, so here we must detch it from the graph
                a = a.detach()
                a.requires_grad = True
                inputs.append(a)

            self.inp_cache = inputs
        else:
            inputs = args
        return inputs

    # NOTE: now we can NOT put functions like torch.xx and F.xx to nn.Sequential,
    #       and the forward of the submodel may be very special or complex,
    #       so in the most cases we must implement submodel mannually with
    #       a whole forward function, and then wrap them with this class
    def forward(self, *args, **kwargs):
        inputs = self.cache_graded_input(*args)
        out = self.net(*inputs)
        self.out_cache = out  # NOTE: cache the out for mp_backward
        return out

    def get_input_grads(self):
        if self.mp_order > 0:
            grads = [inp.grad for inp in self.inp_cache]
        else:
            grads = None

        return grads

    def mp_backward(self, grads=None, *args, **kwargs):
        if grads is not None:  # not the last submodule which has already done the backward with loss
            if not isinstance(self.out_cache, (list, tuple, set)):
                outs = [self.out_cache]
            else:
                outs = self.out_cache
            for out, grad in zip(outs, grads):
                torch.autograd.backward(out, grad)
                # or,
                #out.backward(grad)

        grads = self.get_input_grads()

        return grads


class ModelPartBase(nn.Module):  # TODO: not used now
    def __init__(self, submdls={}, md_device='cuda:0'):
        super(ModelPartBase, self).__init__()

    def forward(self, *args, **kwargs):
        out = self.net(*args)
        self.out_cache = out
        return out

    def mp_backward(self, grads, *args, **kwargs):
        pass


class NetPart1(nn.Module):
    def __init__(self):
        super(NetPart1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x


class NetPart2(nn.Module):
    def __init__(self):
        super(NetPart2, self).__init__()
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.submdl_1 = NetPart1()
        self.submdl_2 = NetPart2()

    def md_split_to_submdls(self, devs=[]):
        if not devs:
            raise Exception('no multi devices')

        self.devs = devs
        device1, device2 = devs[0], devs[1]  # TODO: hard code for now

        self.submdl_wrp_1 = ModelPartWrapper(
            submdls=[self.submdl_1,], md_device=device1, mp_order=0)
        self.submdl_wrp_1 = self.submdl_wrp_1.to(device1)

        self.submdl_wrp_2 = ModelPartWrapper(
            submdls=[self.submdl_2,], md_device=device2, mp_order=1)
        self.submdl_wrp_2 = self.submdl_wrp_2.to(device2)

        return self.submdl_wrp_1, self.submdl_wrp_2

    def forward(self, x):
        device1, device2 = self.devs[0], self.devs[1]  # TODO: hard code for now

        x = x.to(device1)

        x = self.submdl_wrp_1(x)

        # ----------------------------------------
        x = x.to(device2)  # transmitting the intermediate-outputs

        output = self.submdl_wrp_2(x)
        return output

    # ==================================================================

    def mp_backward(self, loss):
        device1, device2 = self.devs[0], self.devs[1]  # TODO: hard code for now

        loss.backward()
        # -----------------------------------------------------------------------
        # NOTE: the submodl_wrp_2 is already done backward after loss backward
        # because the loss is on the same device with it,
        # and if they are not on the same device, submodl_wrp_2 should do backward
        # like submodl_wrp_1, where its grads come from its output with loss
        # -----------------------------------------------------------------------
        grads_inp_2 = self.submdl_wrp_2.get_input_grads()  # now in device2

        grads_out_1 = [g.to(device1) for g in grads_inp_2]  # transmitting the gradients
        self.submdl_wrp_1.mp_backward(grads_out_1)


def train(args, model, md1, md2, device1, device2, train_loader, optimizer, optimizer1, optimizer2, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device1), target.to(device2)  # NOTE: the two are on the different devices
        #optimizer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # -----------------------------------------------------------------------
        # here we do a MP backward with mannually transmitting the gradients
        # -----------------------------------------------------------------------
        #loss.backward()
        model.mp_backward(loss)
        # -----------------------------------------------------------------------
        # NOTE: here we do not use the original optimizer because its containing
        # parameters are not on the right GPU device (default is GPU0)
        # -----------------------------------------------------------------------
        #optimizer.step()
        optimizer1.step()
        optimizer2.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device1, device2, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device1), target.to(device2)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device1 = torch.device("cuda:0" if use_cuda else "cpu")
    device2 = torch.device("cuda:1" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net()#.to(device)
    md1, md2 = model.md_split_to_submdls(devs=[device1, device2])
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer1 = optim.Adadelta(md1.parameters(), lr=args.lr)
    optimizer2 = optim.Adadelta(md2.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler1 = StepLR(optimizer1, step_size=1, gamma=args.gamma)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, md1, md2, device1, device2, train_loader, optimizer, optimizer1, optimizer2, epoch)
        test(model, device1, device2, test_loader)
        #scheduler.step()
        scheduler1.step()
        scheduler2.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
