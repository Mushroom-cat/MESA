import os
import time
import torch
import argparse
import pickle

from original_utils import *
from utils import *

from options import args
from models import model_factory


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'



if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset, args.existing_split)

    [user_train, user_valid, user_test, usernum, itemnum, inter_mat, exist_user, eval_user] = dataset

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    with open("data/ml-1m/m_user_dict.pkl", "rb") as f1:
        user_dict = pickle.load(f1)

    sampler = WarpSampler(user_train, usernum, itemnum, user_dict, exist_user, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    model = model_factory(inter_mat, usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)


    model.train()
    epoch_start_idx = 1


    if args.state_dict_path is not None:
        # try:
        #     model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        #     tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        #     epoch_start_idx = int(tail[:tail.find('.')]) + 1
        # except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        #     print('failed loading state_dicts, pls check file path: ', end="")
        #     print(args.state_dict_path)
        #     print('pdb enabled for your quick check, pls type exit() if you do not need it')
        #     import pdb; pdb.set_trace()

        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]



    if args.inference_only:
        time_start = time.time()
        model.eval()
        t_test = evaluate(model, dataset, user_dict, args, eval_user)
        print('test (NDCG@1: %.4f, HR@1: %.4f, NDCG@5: %.4f, HR@5: %.4f, NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
        print('test time:', time.time() - time_start)

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # meta_params = list(map(id, model.meta_layers.parameters()))
    # base_params = filter(lambda p: id(p) not in meta_params, model.parameters())
    # params = [{'params': base_params, 'lr': args.lr},
    #           {'params': model.meta_layers.parameters(), 'lr': args.meta_lr}]
    # adam_optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    start_time = time.time()
    t0 = time.time()
    loss_list = []
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = torch.tensor(u).to(args.device), torch.tensor(np.array(seq)).to(args.device), np.array(pos), np.array(neg)
            pos_logits, neg_logits, balance_loss = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss += args.balance_loss_weight * balance_loss

            if step == num_batch-1:
                loss_list.append(loss.detach().item())

            loss.backward()
            adam_optimizer.step()

            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 200 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, user_dict, args)
            t_valid = evaluate_valid(model, dataset, user_dict, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'BERT4Rec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    #np.save('ml-1m_default/loss.npy', np.stack(loss_list))
    print('train time:', time.time() - start_time)

    f.close()
    #sampler.close()
    print("Done")
