import torch
import torch.nn.functional as F
import numpy as np
import random
import scipy.io as scio
import h5py
from .model_factory import HermNet


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def power_normalization_new(x, eps=1e-12):
    # x: (N, 2, L)
    power = np.mean(x[:,0,:]**2 + x[:,1,:]**2, axis=1, keepdims=True)  # (N,1)
    scale = np.sqrt(power + eps)  # 防止除0

    x[:,0,:] = x[:,0,:] / scale
    x[:,1,:] = x[:,1,:] / scale
    return x


def load_dataset(dataset_name, dataset_indexes, ratio, train_state, seed):
    setup_seed(seed)

    allowed_indexes = {
        'CD2025': ['L1','L2', 'L3', 'L4']
    }

    if dataset_name not in allowed_indexes:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    invalid_indexes = [idx for idx in dataset_indexes if idx not in allowed_indexes[dataset_name]]
    if invalid_indexes:
        raise ValueError(f"Invalid dataset indexes '{invalid_indexes}' for dataset '{dataset_name}'. Allowed indexes: {allowed_indexes[dataset_name]}")
    
    x_train, x_val, y_train, y_val = [], [], [], []
    d_train = []
    x_test, y_test = [], []

    for i, dataset_index in enumerate(dataset_indexes):
        filename = f'./dataset/{dataset_name}/{dataset_index}.h5'
        with h5py.File(filename, 'r') as f:
            x = f['X'][:]
            y = f['Y'][:]
        x = power_normalization_new(x)
        if train_state:
            index = np.random.permutation(x.shape[0])
            x_, y_ = x[index], y[index]
            split_point = int(round(ratio * x_.shape[0]))

            x_train.append(x_[:split_point])
            y_train.append(y_[:split_point])

            x_val.append(x_[split_point:])
            y_val.append(y_[split_point:])

            d_train.append(np.full_like(y_[:split_point], i))
        else:
            x_test.append(x)
            y_test.append(y)

    if train_state:
        return x_train, y_train, d_train, np.concatenate(x_val, axis=0), np.concatenate(y_val, axis=0)
    else:
        return np.concatenate(x_test, axis=0), np.concatenate(y_test, axis=0)

def train(model, loss, train_dataloader, optimizer, epoch):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        n = len(data_nn)//2
        data_list, target_list = data_nn[:n], data_nn[n:]
        data = torch.cat(data_list, 0).float().cuda()
        target = torch.cat(target_list, 0).long().cuda()
        #print(data.shape)
        optimizer.zero_grad()
        cls_output = model(data)
        cls_output = F.log_softmax(cls_output, dim=1)
        result_loss = loss(cls_output, target)
        result_loss.backward()

        optimizer.step()
        all_loss += result_loss.item()*data.size()[0]
        pred = cls_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset)/n,
        correct,
        len(train_dataloader.dataset)*n,
        100.0 * correct / len(train_dataloader.dataset)/n)
    )

def evaluate(model, loss, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            cls_output = model(data)
            cls_output = F.log_softmax(cls_output, dim=1)
            test_loss += loss(cls_output, target).item()*data.size()[0]
            pred = cls_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return test_loss

def test(model, test_dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            cls_output = model(data)
            pred = cls_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_dataloader.dataset)


def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, save_path):
    current_min_test_loss = float('inf')
    epochs_without_improvement = 0
    patience=10

    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_dataloader, optimizer, epoch)
        test_loss = evaluate(model, loss_function, val_dataloader, epoch)

        if test_loss < current_min_test_loss:
            print(f"The validation loss is improved from {current_min_test_loss:.4f} to {test_loss:.4f}, new model weight is saved.")
            current_min_test_loss = test_loss
            epochs_without_improvement = 0  # 重置未改善的计数
            torch.save(model.state_dict(), save_path)
        else:
            print("The validation loss is not improved.")
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping: No improvement for {patience} epochs. Training stopped.")
            break

def get_model(modelname, num_cls, d_model, fused_dim, gmlp_layers, dropout, eta):
    if modelname == "HermNet":
        model = HermNet.HermNet(num_cls, d_model, fused_dim, gmlp_layers, dropout, eta)
    return model