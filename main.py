import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from utils.train_tools import *
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")
    parser.add_argument('--model_name', type=str, default='HermNet')
    parser.add_argument('--mode', type=str, default='train_test', help='Mode: train_test or only_test')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--dataset_name', type=str, default='CD2025')
    parser.add_argument('--num_cls', type=int, default=10)
    parser.add_argument('--dataset_indexes_for_train', type=str, nargs='+', default=['L1','L2','L3'])
    parser.add_argument('--dataset_indexes_for_test', type=str, nargs='+', default=['L4'])
    parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device index')

    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--fused_dim', type=int, default=8)
    parser.add_argument('--gmlp_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--reduction',type=int, default=2)
    parser.add_argument('--version',type=str, default='full')
    return parser.parse_args()

def main(args):
    setup_seed(args.seed)
    args.batch_size = args.batch_size//len(args.dataset_indexes_for_train)

    model_weight_path = f'weight/{args.model_name}_dm{args.d_model}_fd{args.fused_dim}_gmlp{args.gmlp_layers}_red{args.reduction}_{args.version}_{args.dataset_indexes_for_test}.pth'

    if args.mode in ['only_train', 'train_test']:
        x_train, y_train, _, x_val, y_val = load_dataset(args.dataset_name, args.dataset_indexes_for_train, args.train_ratio, True, args.seed)
        train_dataset = TensorDataset(*[torch.tensor(x) for x in x_train], *[torch.tensor(y) for y in y_train])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        print("args.batch_size =", args.batch_size)
        print("train_dataloader.batch_size =", train_dataloader.batch_size)
        print("len(train_dataset) =", len(train_dataset))
        print("len(train_dataloader) =", len(train_dataloader))
        val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        
        model = get_model(args.model_name, args.num_cls,args.d_model,args.fused_dim,args.gmlp_layers,args.dropout,args.reduction).cuda()   #for Hyperparameter experment

        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss = nn.NLLLoss().cuda()

        train_and_evaluate(
            model, loss_function=loss, train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, optimizer=optim, epochs=args.epochs,
            save_path=model_weight_path
        )

    if args.mode in ['only_test', 'train_test']:
        model = get_model(args.model_name, args.num_cls,args.d_model,args.fused_dim,args.gmlp_layers,args.dropout,args.reduction).cuda()
        model.load_state_dict(torch.load(model_weight_path, weights_only=True))

        x_test, y_test = load_dataset(args.dataset_name, args.dataset_indexes_for_test, args.train_ratio, False, args.seed)
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        acc = test(model, test_dataloader)
      
        return acc

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    test_acc = main(args)
    print(f'Model Name: {args.model_name}, Dataset Name: {args.dataset_name}, Training Dataset: {args.dataset_indexes_for_train}, Testing Dataset: {args.dataset_indexes_for_test}')
    print(f'Test Accuracy = {test_acc:.4f}')