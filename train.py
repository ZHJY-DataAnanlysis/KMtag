import random
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import os
import datasets
from tqdm import tqdm
import argparse
import wandb

from eval import evaluate

import utils


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--data', type=str, default='WebOfScience')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--early-stop', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--arch', type=str, default='bert-base-uncased')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)
    return parser


class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    if args.wandb:
        wandb.init(config=args, project='HPT')
    print(args)
    utils.seed_torch(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'), weights_only=False)
    label_dict = {i: v for i, v in label_dict.items()}

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'), weights_only=False)
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1


    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth


    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

    max_node_id = num_class - 1
    
    for node_id in range(num_class):
        parent_id = value2slot.get(node_id, -1)
        if parent_id != -1 and parent_id < num_class:
            path_list.append((node_id, parent_id))

    if not path_list:
        for depth in range(max_depth):
            nodes_at_depth = depth2label.get(depth, [])
            for i, node in enumerate(nodes_at_depth):
                if i > 0:
                    path_list.append((node, nodes_at_depth[i-1]))
                path_list.append((node, node))

    if args.model == 'prompt':
        if os.path.exists(os.path.join(data_path, args.model)):
            dataset = datasets.load_from_disk(os.path.join(data_path, args.model))
        else:
            dataset = datasets.load_dataset('json',
                                            data_files={'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                        'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                        'test': 'data/{}/{}_test.json'.format(args.data, args.data), })

            new_tokens = []
            for i in range(max_depth + 1):
                new_tokens.append(f'<depth_{i}>')

            num_added_tokens = tokenizer.add_tokens(new_tokens)
            print(f"添加了 {num_added_tokens} 个新token，词汇表大小从 {tokenizer.vocab_size - num_added_tokens} 扩展到 {tokenizer.vocab_size}")

            prefix = []
            for i in range(max_depth):
                depth_token_id = tokenizer.convert_tokens_to_ids(f'<depth_{i}>')
                separator_token_id = tokenizer.convert_tokens_to_ids(f'<depth_{max_depth}>')
                prefix.append(depth_token_id)
                prefix.append(separator_token_id)
            
            # 关键修复：添加multiclass token (这是模型查找的token)
            # 根据GraphEmbedding的结构计算：
            # total_size = original_vocab + num_class + depth + 1
            # multiclass_token_id = total_size - 1 = original_vocab + num_class + depth
            multiclass_token_id = tokenizer.vocab_size + num_class + max_depth
            prefix.append(multiclass_token_id)

            prefix.append(tokenizer.sep_token_id)


            def data_map_function(batch, tokenizer):
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                for l, t in zip(batch['label'], batch['token']):
                    new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)])
                    for d in range(max_depth):
                        for i in depth2label[d]:
                            new_batch['labels'][-1][d][i] = 0
                        for i in l:
                            if new_batch['labels'][-1][d][i] == 0:
                                new_batch['labels'][-1][d][i] = 1

                    new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y]

                    tokens = tokenizer(t, truncation=True)
                    new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix)
                    new_batch['input_ids'][-1].extend(
                        [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                    new_batch['attention_mask'].append(
                        tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                    new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                    new_batch['token_type_ids'].append([0] * 512)

                return new_batch


            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(os.path.join(data_path, args.model))
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        from models.prompt import Prompt

    else:
        raise NotImplementedError
    if args.low_res:
        if os.path.exists(os.path.join(data_path, 'low.json')):
            index = json.load(open(os.path.join(data_path, 'low.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(data_path, 'low.json'), 'w'))
        dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])
    total_nodes = num_class
    print(f"总节点数: {total_nodes}, 原始标签数: {len(label_dict)}, 最大深度: {max_depth}")
    
    model = Prompt.from_pretrained(args.arch, num_labels=num_class, path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label,
                                   use_safetensors=True)
    
    if hasattr(tokenizer, 'added_tokens_encoder') and len(tokenizer.added_tokens_encoder) > 0:
        print(f"调整模型嵌入层大小以适应扩展的词汇表: {tokenizer.vocab_size}")
        model.resize_token_embeddings(tokenizer.vocab_size)
    
    model.init_embedding()

    model.to('cuda')
    if args.wandb:
        wandb.watch(model)

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    model.to('cuda')
    optimizer = Adam(model.parameters(), lr=args.lr)

    save = Save(model, optimizer, None, args)
    best_score_macro = -1
    best_score_micro = -1
    early_stop_count = 0
    update_step = 0
    loss = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))

    for epoch in range(100):
        if early_stop_count >= args.early_stop:
            break

        model.train()
        with tqdm(train, desc=f"Epoch {epoch} Training") as p_bar:
            for batch in p_bar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(**batch)
                output['loss'].backward()
                loss += output['loss'].item()
                update_step += 1
                if update_step % args.update == 0:
                    if args.wandb:
                        wandb.log({'loss': loss, })
                    p_bar.set_description(f'Epoch {epoch} loss:{loss:.4f}')
                    optimizer.step()
                    optimizer.zero_grad()
                    loss = 0
                    update_step = 0
                    

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev, desc="Validating") as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, threshold=0.01)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    
                    # 正确解析标签: labels的形状是 [max_depth * num_class]
                    total_label_size = g.size(0)
                    max_depth_eval = total_label_size // num_class
                    g_reshaped = g.view(max_depth_eval, num_class)
                    
                    # 在任何深度有标签1的类别都视为正例
                    for depth_idx in range(max_depth_eval):
                        for class_idx in range(num_class):
                            if g_reshaped[depth_idx, class_idx] == 1:
                                if class_idx not in gold[-1]:
                                    gold[-1].append(class_idx)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)

        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1})
        early_stop_count += 1
        if macro_f1 >= best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 >= best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
        if args.wandb:
            wandb.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        torch.cuda.empty_cache()

    test = DataLoader(dataset['test'], batch_size=8, shuffle=False)
    model.eval()


    def test_function(extra):
        checkpoint_path = os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra))

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file {checkpoint_path} not found!")
            print("Using the last checkpoint instead...")
            checkpoint_path = os.path.join('checkpoints', args.name, 'checkpoint_last.pt')

            if not os.path.exists(checkpoint_path):
                print(f"Error: No checkpoint files found in {os.path.join('checkpoints', args.name)}")
                print("Skipping test...")
                return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['param'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Skipping test...")
            return
            
        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, threshold=0.01)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)
            prefix = 'test' + extra
        if args.wandb:
            wandb.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})


    test_function('_macro')
