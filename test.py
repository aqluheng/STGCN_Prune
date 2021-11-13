from MySTGCN import ST_GCN_18
from Dataset import *
from tqdm import tqdm

def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def load_myCheckpoint(model, path):
    load_dict = torch.load(path)["state_dict"]
    new_dict = {}
    for k, v in load_dict.items():
        name = k
        if len(k) > 20:
            if k[18:21] == "tcn":  # tcn.conv
                if k[22] == "2":
                    name = k[:22]+'3'+k[23:]
                elif k[22] == "3":  # tcn.bn
                    name = k[:22]+'5'+k[23:]
            if k[18:26] == "residual":
                if k[27] == "0":
                    name = k[:27]+'1'+k[28:]
                elif k[27] == "1":
                    name = k[:27]+'3'+k[28:]
        new_dict[name] = v
    model.load_state_dict(new_dict)
    return model

model = ST_GCN_18(in_channels=3, num_class=400, graph_cfg={
                  'layout': 'openpose', 'strategy': 'spatial'}, edge_importance_weighting=True)

old_dict = torch.load("modelpth/st_gcn.kinetics-6fa43f73.pth")
new_dict = {}
for k, v in old_dict.items():
    if "tcn" in k:
        if k[22] == '0':
            k = k.replace("tcn.0","tcnBN1")
        elif k[22] == '2':
            k = k.replace("tcn.2","tcnConv")
        elif k[22] == '3':
            k = k.replace("tcn.3","tcnBN2")
    new_dict[k] = v
model.load_state_dict(new_dict)
torch.save(model.state_dict(),"modelpth/st_gcn_kinetics_spilt.pth")
model.load_state_dict(torch.load("modelpth/st_gcn_kinetics_spilt.pth"))

exit()
model.eval()
model.cuda()

data_path = "./data/Kinetics/kinetics-skeleton/val_data.npy"
label_path = "./data/Kinetics/kinetics-skeleton/val_label.pkl"

dataset = SkeletonFeeder(data_path, label_path)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=24,
                                          shuffle=False,
                                          num_workers=4)


results = []
labels = []
with torch.no_grad():
    for data, label in tqdm(data_loader):
        output = model(data.cuda()).data.cpu().numpy()
        results.append(output)
        labels.append(label)
    results = np.concatenate(results)
    labels = np.concatenate(labels)

    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))
