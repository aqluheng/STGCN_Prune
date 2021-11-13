from MySTGCN import ST_GCN_18
from Dataset import *
from tqdm import tqdm

def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


model = ST_GCN_18(in_channels=3, num_class=400, graph_cfg={
                  'layout': 'openpose', 'strategy': 'spatial'}, edge_importance_weighting=True)
model.load_state_dict(torch.load("modelpth/st_gcn.kinetics-6fa43f73.pth"))
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
