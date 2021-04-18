import torch
import time
import numpy as np
import torch.nn.functional as F
from gcn.models import GCN
from gcn.utils import load_data, accuracy, train_ending
import syft as sy
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

# set clients
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id='bob')
alice = sy.VirtualWorker(hook, id='alice')
james = sy.VirtualWorker(hook, id='james')

compute_nodes = [bob, alice]


# super parameters
class Parser:
    def __init__(self):
        self.epoch = 200  # 迭代次数
        self.E = 5  # 迭代多少次交换一轮
        self.lr = 0.01
        self.seed = 10
        self.nhid = 32
        self.nclass = 7
        self.dropout = 0.5
        self.no_cuda = False
        self.weight_decay = 5e-4
        self.zubie = 5


args = Parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {}

# load data
remote_dataset = [[], []]
# 画图用
loss_train_list = [[], [], [], [], []]
accuracy_train_list = [[], [], [], [], []]
loss_val_list = [[], [], [], []]
accuracy_val_list = [[], [], [], []]
adj, features, labels, idx_bob, idx_alice, idx_test, idx_val, idx_train = load_data()

if args.cuda:
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_bob = idx_bob.cuda()
    idx_alice = idx_alice.cuda()
    idx_test = idx_test.cuda()
    idx_val = idx_val.cuda()
    idx_train = idx_train.cuda()

bob_adj = adj.send(bob)
bob_features = features.send(bob)
bob_labels = labels[idx_bob].send(bob)
bob_idx = idx_bob.send(bob)

alice_adj = adj.send(alice)
alice_features = features.send(alice)
alice_labels = labels[idx_alice].send(alice)
alice_idx = idx_alice.send(alice)

# 以列表嵌套list的形式存放数据
remote_dataset[0] = [bob_adj, bob_features, bob_labels, bob_idx]
remote_dataset[1] = [alice_adj, alice_features, alice_labels, alice_idx]

# model and optimizer
local_model = GCN(nfeat=features.shape[1], nhid=args.nhid, nclass=args.nclass, dropout=args.dropout)

bob_alone_model = GCN(nfeat=features.shape[1], nhid=args.nhid, nclass=args.nclass, dropout=args.dropout)
alice_alone_model = GCN(nfeat=features.shape[1], nhid=args.nhid, nclass=args.nclass, dropout=args.dropout)

bob_fed_model = GCN(nfeat=features.shape[1], nhid=args.nhid, nclass=args.nclass, dropout=args.dropout)
alice_fed_model = GCN(nfeat=features.shape[1], nhid=args.nhid, nclass=args.nclass, dropout=args.dropout)

local_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

bob_alone_optimizer = torch.optim.Adam(bob_alone_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
alice_alone_optimizer = torch.optim.Adam(alice_alone_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

bob_fed_optimizer = torch.optim.Adam(bob_fed_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
alice_fed_optimizer = torch.optim.Adam(alice_fed_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

models = [bob_fed_model, alice_fed_model]
params = [list(bob_fed_model.parameters()), list(alice_fed_model.parameters())]
optimizers = [bob_fed_optimizer, alice_fed_optimizer]

if args.cuda:
    local_model = local_model.cuda()
    bob_alone_model = bob_alone_model.cuda()
    alice_alone_model = alice_alone_model.cuda()
    bob_fed_model = bob_fed_model.cuda()
    alice_fed_model = alice_fed_model.cuda()


# define update process
def update(epoch, adj_train, features_train, labels_train, model_train, optimizer, index):
    model_train.send(labels_train.location)
    model_train.train()

    optimizer.zero_grad()
    output = model_train(features_train, adj_train)
    loss_train = F.nll_loss(output[index], labels_train)
    acc_train = accuracy(output[index], labels_train)
    loss_train.backward()
    optimizer.step()
    loss_train = loss_train.get()
    acc_train = acc_train.get()
    print(
        'Epoch: {:04d}'.format(epoch + 1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
    )
    model_train.get()
    return model_train, loss_train, acc_train


# define training process
def bob_alone_train(epoch):
    t = time.time()
    bob_alone_model.train()
    bob_alone_optimizer.zero_grad()
    output = bob_alone_model(features, adj)
    loss_train = F.nll_loss(output[idx_bob], labels[idx_bob])
    acc_train = accuracy(output[idx_bob], labels[idx_bob])
    loss_train.backward()
    bob_alone_optimizer.step()
    # val
    bob_alone_model.eval()
    output = bob_alone_model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))

    loss_train_list[0].append(loss_train.cpu().item())
    accuracy_train_list[0].append(acc_train.cpu().item())
    loss_val_list[0].append(loss_val.cpu().item())
    accuracy_val_list[0].append(acc_val.cpu().item())


def alice_alone_train(epoch):
    t = time.time()
    alice_alone_model.train()
    alice_alone_optimizer.zero_grad()
    output = alice_alone_model(features, adj)
    loss_train = F.nll_loss(output[idx_alice], labels[idx_alice])
    acc_train = accuracy(output[idx_alice], labels[idx_alice])
    loss_train.backward()
    alice_alone_optimizer.step()
    # val
    alice_alone_model.eval()
    output = alice_alone_model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))

    loss_train_list[1].append(loss_train.cpu().item())
    accuracy_train_list[1].append(acc_train.cpu().item())
    loss_val_list[1].append(loss_val.cpu().item())
    accuracy_val_list[1].append(acc_val.cpu().item())


def local_train(epoch):
    t = time.time()
    local_model.train()
    local_optimizer.zero_grad()
    output = local_model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    local_optimizer.step()
    # val
    local_model.eval()
    output = local_model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))

    loss_train_list[2].append(loss_train.cpu().item())
    accuracy_train_list[2].append(acc_train.cpu().item())
    loss_val_list[2].append(loss_val.cpu().item())
    accuracy_val_list[2].append(acc_val.cpu().item())


# define federated training process
def federated_train(epoch):
    for remote_index in range(len(compute_nodes)):
        adj_train, features_train, labels_train, index_train = remote_dataset[remote_index]
        models[remote_index], loss, acc = update(epoch, adj_train, features_train, labels_train, models[remote_index],
                                                 optimizers[remote_index], index_train)
        loss_train_list[remote_index + 3].append(loss.cpu().item())
        accuracy_train_list[remote_index + 3].append(acc.cpu().item())
    if (epoch + 1) % args.E == 0:
        # encrypt
        new_params = list()
        for param_i in range(len(params[0])):
            spdz_params = list()
            for remote_index in range(len(compute_nodes)):
                spdz_params.append(
                    params[remote_index][param_i].fix_precision().share(bob, alice, crypto_provider=james))
            new_param = (spdz_params[0]*1 + spdz_params[1]*3).get().float_precision() / 4
            new_params.append(new_param)
        # clean up
        with torch.no_grad():
            for x in params:
                for param in x:
                    param *= 0

            # for model in models:
            #     model.get()

            for remote_index in range(len(compute_nodes)):
                for param_index in range(len(params[remote_index])):
                    params[remote_index][param_index].set_(new_params[param_index].cuda())

    bob_fed_model.eval()
    output = bob_fed_model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_val_list[3].append(loss_val.cpu().item())
    accuracy_val_list[3].append(acc_val.cpu().item())


# plot
def plot():
    plt.figure()
    ax1 = plt.subplot(221)
    plt.plot(np.linspace(0, len(loss_train_list[3]), len(loss_train_list[3])), loss_train_list[3],
             label='bob federated', color='blue')
    plt.plot(np.linspace(0, len(loss_train_list[4]), len(loss_train_list[4])), loss_train_list[4],
             label='alice federated', color='red')
    plt.plot(np.linspace(0, len(loss_train_list[0]), len(loss_train_list[0])), loss_train_list[0], label='bob alone',
             color='yellow')
    plt.plot(np.linspace(0, len(loss_train_list[1]), len(loss_train_list[1])), loss_train_list[1], label='alice alone',
             color='green')
    plt.plot(np.linspace(0, len(loss_train_list[2]), len(loss_train_list[2])), loss_train_list[2], label='local train',
             color='black')
    plt.legend(ncol=2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train loss')
    ax2 = plt.subplot(222)
    plt.plot(np.linspace(0, len(accuracy_train_list[3]), len(accuracy_train_list[3])), accuracy_train_list[3],
             label='bob federated', color='blue')
    plt.plot(np.linspace(0, len(accuracy_train_list[4]), len(accuracy_train_list[4])), accuracy_train_list[4],
             label='alice federated', color='red')
    plt.plot(np.linspace(0, len(accuracy_train_list[0]), len(accuracy_train_list[0])), accuracy_train_list[0],
             label='bob alone', color='yellow')
    plt.plot(np.linspace(0, len(accuracy_train_list[1]), len(accuracy_train_list[1])), accuracy_train_list[1],
             label='alice alone', color='green')
    plt.plot(np.linspace(0, len(accuracy_train_list[2]), len(accuracy_train_list[2])), accuracy_train_list[2],
             label='local train', color='black')
    plt.legend(ncol=2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('train accuracy')
    ax3 = plt.subplot(223)
    plt.plot(np.linspace(0, len(loss_val_list[3]), len(loss_val_list[3])), loss_val_list[3], label='federated',
             color='blue')
    plt.plot(np.linspace(0, len(loss_val_list[0]), len(loss_val_list[0])), loss_val_list[0], label='bob alone',
             color='red')
    plt.plot(np.linspace(0, len(loss_val_list[1]), len(loss_val_list[1])), loss_val_list[1], label='alice alone',
             color='green')
    plt.plot(np.linspace(0, len(loss_val_list[2]), len(loss_val_list[2])), loss_val_list[2], label='local train',
             color='black')
    plt.legend(ncol=2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('val loss')
    ax4 = plt.subplot(224)
    plt.plot(np.linspace(0, len(accuracy_val_list[3]), len(accuracy_val_list[3])), accuracy_val_list[3],
             label='federated', color='blue')
    plt.plot(np.linspace(0, len(accuracy_val_list[0]), len(accuracy_val_list[0])), accuracy_val_list[0],
             label='bob alone', color='red')
    plt.plot(np.linspace(0, len(accuracy_val_list[1]), len(accuracy_val_list[1])), accuracy_val_list[1],
             label='alice alone', color='green')
    plt.plot(np.linspace(0, len(accuracy_val_list[2]), len(accuracy_val_list[2])), accuracy_val_list[2],
             label='local train', color='black')
    plt.legend(ncol=2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('val accuracy')
    plt.show()
    # plt.savefig(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\image\2_1_' + str(args.zubie) + '.png')


# final test
def final_test(ls):
    bob_alone_model.eval()
    output_test_bob_alone = bob_alone_model(features, adj)
    loss_test_bob_alone = F.nll_loss(output_test_bob_alone[idx_test], labels[idx_test])
    acc_test_bob_alone = accuracy(output_test_bob_alone[idx_test], labels[idx_test])
    print('bob test:\nacc:{} loss:{}'.format(acc_test_bob_alone.item(), loss_test_bob_alone.item()))

    alice_alone_model.eval()
    output_test_alice_alone = alice_alone_model(features, adj)
    loss_test_alice_alone = F.nll_loss(output_test_alice_alone[idx_test], labels[idx_test])
    acc_test_alice_alone = accuracy(output_test_alice_alone[idx_test], labels[idx_test])
    print('alice test:\nacc:{} loss:{}'.format(acc_test_alice_alone.item(), loss_test_alice_alone.item()))

    local_model.eval()
    output_test_local = local_model(features, adj)
    loss_test_local = F.nll_loss(output_test_local[idx_test], labels[idx_test])
    acc_test_local = accuracy(output_test_local[idx_test], labels[idx_test])
    print('local test:\nacc:{} loss:{}'.format(acc_test_local.item(), loss_test_local.item()))

    bob_fed_model.eval()
    output_test_bob_fed = bob_fed_model(features, adj)
    loss_test_bob_fed = F.nll_loss(output_test_bob_fed[idx_test], labels[idx_test])
    acc_test_bob_fed = accuracy(output_test_bob_fed[idx_test], labels[idx_test])
    print('federated test:\nacc:{} loss:{}'.format(acc_test_bob_fed.item(), loss_test_bob_fed.item()))

    df_test = pd.DataFrame(data=[[acc_test_bob_alone.item(), loss_test_bob_alone.item(), ls[0]],
                                 [acc_test_alice_alone.item(), loss_test_alice_alone.item(), ls[1]],
                                 [acc_test_local.item(), loss_test_local.item(), ls[2]],
                                 [acc_test_bob_fed.item(), loss_test_bob_fed.item(), ls[3]]],
                           columns=['acc', 'loss', 'epoch'], index=['bob', 'alice', 'local', 'fed'])
    with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\citeseer-results\testing\2_1_' + str(
            args.zubie) + '.pkl', 'wb') as f5:
        pkl.dump(df_test, f5)
    f5.close()
    print(df_test)


# train model
epoch_list = []
epoch_list.append(0)
epoch_list.append(0)
t_total1 = time.time()
# print('bob train:')
# for epoch in range(args.epoch):
#     bob_alone_train(epoch)
#     if epoch > 10:
#         if train_ending(loss_val_list[0], epoch):
#             break
# print("bob Optimization Finished!\n")
# print('epoch:{}'.format(epoch))
# epoch_list.append(epoch)
# print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total1))

# t_total2 = time.time()
# print('alice train:')
# for epoch in range(args.epoch):
#     alice_alone_train(epoch)
#     if epoch > 10:
#         if train_ending(loss_val_list[1], epoch):
#             break
# print("alice Optimization Finished!\n")
# print('epoch:{}'.format(epoch))
# epoch_list.append(epoch)
# print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total2))
# 
t_total3 = time.time()
print('local train:')
for epoch in range(args.epoch):
    local_train(epoch)
    if epoch > 40:
        if train_ending(loss_val_list[2], epoch):
            break
print("local Optimization Finished!\n")
print('epoch:{}'.format(epoch))
epoch_list.append(epoch)
print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total3))

t_total4 = time.time()
print('Federated train:')
for epoch in range(args.epoch):
    federated_train(epoch)
    if epoch > 40:
        if train_ending(loss_val_list[3], epoch):
            break
print("Federated Optimization Finished!")
print('epoch:{}'.format(epoch))
epoch_list.append(epoch)
print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total4))

print("\nTotal time elapsed: {:.4f}s\n".format(time.time() - t_total1))
print(epoch_list)
final_test(epoch_list)
# 画图
plot()

# data preprocess and save
val_info_dict = {'bob_val_acc': accuracy_val_list[0], 'alice_val_acc': accuracy_val_list[1],
                 'local_val_acc': accuracy_val_list[2], 'fed_val_acc': accuracy_val_list[3],
                 'bob_val_loss': loss_val_list[0], 'alice_val_acc': loss_val_list[1],
                 'local_val_acc': loss_val_list[2], 'fed_val_acc': loss_val_list[3]}
with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\citeseer-results\validation\2_1_' + str(
        args.zubie) + '.pkl', 'wb') as f:
    pkl.dump(val_info_dict, f)
f.close()

# model saving
# with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\model\bob_alone_2_1_' + str(
#         args.zubie) + '.pkl', 'wb') as f1:
#     torch.save(bob_alone_model, f1)
# f1.close()
# with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\model\alice_alone_2_1_' + str(
#         args.zubie) + '.pkl', 'wb') as f2:
#     torch.save(alice_alone_model, f2)
# f2.close()
# with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\model\local_2_1_' + str(args.zubie) + '.pkl',
#           'wb') as f3:
#     torch.save(local_model, f3)
# f3.close()
# with open(r'C:\Users\lzl_z\Desktop\Fed_GCN_Experiment_two\model\fed_2_1_' + str(args.zubie) + '.pkl',
#           'wb') as f4:
#     torch.save(bob_fed_model, f4)
# f4.close()







