import os, csv, json, random, torch
from collections import defaultdict

csv_path = 'data.csv'
dataset_name = 'MyCSV'
text_col = 'question'
level_cols = [
    'level_1_knowledge', 'level_2_knowledge', 'level_3_knowledge',
    'level_4_knowledge', 'level_5_knowledge', 'level_6_knowledge',
    'level_7_knowledge', 'level_8_knowledge', 'level_9_knowledge'
]
seed = 42
train_ratio, dev_ratio = 0.8, 0.1

random.seed(seed)
data_dir = os.path.join('data', dataset_name)
os.makedirs(data_dir, exist_ok=True)

node_key_to_id = {}
next_id = 0
slot = defaultdict(set)
root_children = set()
samples = []


def get_or_add_node(level_idx, name):
    global next_id
    key = f"L{level_idx + 1}/{name}"
    if key not in node_key_to_id:
        node_key_to_id[key] = next_id
        next_id += 1
    return node_key_to_id[key]


with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = (row.get(text_col) or '').strip()
        if not text:
            continue

        path_names = []
        for lvl, col in enumerate(level_cols):
            val = (row.get(col) or '').strip()
            if not val:
                break

            val = val.split(';')[0].split('|')[0].strip()
            if not val:
                break
            path_names.append((lvl, val))
        if not path_names:
            continue

        node_ids = []
        for i, (lvl, name) in enumerate(path_names):
            cid = get_or_add_node(lvl, name)
            node_ids.append(cid)
            if i == 0:
                root_children.add(cid)
            else:
                pid = get_or_add_node(path_names[i - 1][0], path_names[i - 1][1])
                slot[pid].add(cid)

        samples.append({'token': text, 'label': sorted(set(node_ids))})

id_to_name = {}
for key, idx in node_key_to_id.items():
    _, name = key.split('/', 1)
    id_to_name[idx] = name

slot[-1] = slot[-1].union(root_children)

torch.save(id_to_name, os.path.join(data_dir, 'value_dict.pt'))
torch.save(slot, os.path.join(data_dir, 'slot.pt'))

value2slot = {}
num_class = len(id_to_name)
for s in slot:
    for v in slot[s]:
        value2slot[v] = s


def get_depth(x):
    depth = 0
    while value2slot.get(x, -1) != -1:
        depth += 1
        x = value2slot[x]
    return depth


depth_dict = {i: get_depth(i) for i in range(num_class)}
max_depth = max(depth_dict.values()) + 1 if depth_dict else 1
depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

indices = list(range(len(samples)))
random.shuffle(indices)
n = len(indices)
n_train = int(n * train_ratio)
n_dev = int(n * dev_ratio)

splits = {
    'train': indices[:n_train],
    'dev': indices[n_train:n_train + n_dev],
    'test': indices[n_train + n_dev:]
}

for split, idxs in splits.items():
    out_path = os.path.join(data_dir, f'MyCSV_{split}.json')
    with open(out_path, 'w', encoding='utf-8') as w:
        for i in idxs:

            label_vector = [0] * num_class
            for node_id in samples[i]['label']:
                if node_id < num_class:
                    label_vector[node_id] = 1

            sample_data = {
                'token': samples[i]['token'],
                'label': label_vector
            }
            w.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
