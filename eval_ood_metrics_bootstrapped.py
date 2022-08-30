from operator import mod
import os
import pandas as pd
import torch
import numpy as np
import argparse
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

from datasets.cityscapes import Cityscapes
from datasets.bdd100k import BDD100KSeg
from datasets.road_anomaly import RoadAnomaly
from datasets.fishyscapes import FishyscapesLAF, FishyscapesStatic
from datasets.segment_me_if_you_can import RoadAnomaly21, RoadObstacle21
from train_net import Trainer, setup
from detectron2.checkpoint import DetectionCheckpointer

from easydict import EasyDict as edict

from analysis.max_logits import MaxLogitsAnalyzer, OODEvaluator


parser = argparse.ArgumentParser(description='OOD SemSeg Bootstrap Evaluation')

parser.add_argument('--ratio', type=float, default=0.9, 
            help='Ratio of the dataset to sample when bootstrapping')
parser.add_argument('--trials', type=int, default=20, help="number of times to repeat evaluation")
parser.add_argument('--batch_size', type=int, default=1, help="Batch Size used in evaluation")
parser.add_argument('--num_workers', type=int, default=15, help="Number of threads used in data loader")
parser.add_argument('--device', type=str, default='cuda', help="cpu or cuda, the device used for evaluation")



args = parser.parse_args()

bdd100k_config = edict({
    'seg_downsample_rate': 1,
    'train_file': 'train_paths.txt',
    'val_file': 'val_paths.txt',
    'val_image_strategy': 'no_change',
    'ignore_train_class': True,
    'dataset_root': '/home/user/datasets/bdd100k/seg'
})

cityscapes_config = edict({
    'dataset_root': '/home/user/datasets/cityscapes',
})

road_anomaly_config = edict({
    'dataset_root': '/home/user/datasets/RoadAnomaly/RoadAnomaly_jpg',
    'test_image_strategy': 'no_change'
})

fishyscapes_laf_config = edict({
    'dataset_root': '/home/user/datasets/Fishyscapes/',
})

fishyscapes_static_config = edict({
    'dataset_root': '/home/user/datasets/Fishyscapes/',
})

road_anomaly_21_config = edict({
    'dataset_root': '/home/user/datasets/SegmentMeIfYouCan/dataset_AnomalyTrack',
    'dataset_mode': 'all'
})

road_obstacle_21_config = edict({
    'dataset_root': '/home/user/datasets/SegmentMeIfYouCan/dataset_ObstacleTrack',
    'dataset_mode': 'all'
})

# Cityscapes
transform = A.Compose([
    ToTensorV2()
])
cityscapes_dataset = Cityscapes(cityscapes_config, transform=transform, split='val', 
                                target_type='semantic')
loader_cityscapes = DataLoader(cityscapes_dataset, batch_size=1, shuffle=False, num_workers=15)

# BDD100K
transform_bdd100k = A.Compose([
    ToTensorV2(),
])
bdd100k_dataset = BDD100KSeg(hparams=bdd100k_config, mode='val', transforms=transform_bdd100k, 
                             image_size=(720, 1280))
loader_bdd100k = DataLoader(bdd100k_dataset, batch_size=1, shuffle=False, num_workers=15)
# Road Anomaly
transform_road_anomaly = A.Compose([
    ToTensorV2()
])
road_anomaly_dataset = RoadAnomaly(hparams=road_anomaly_config, transforms=transform_road_anomaly)
loader_road_anomaly = DataLoader(road_anomaly_dataset, batch_size=1, shuffle=False, num_workers=15)

# Fishyscapes LaF
transform_fs_laf = A.Compose([
    ToTensorV2()
])
fs_laf_dataset = FishyscapesLAF(hparams=fishyscapes_laf_config, transforms=transform_fs_laf)
loader_fs_laf = DataLoader(fs_laf_dataset, batch_size=1, shuffle=False, num_workers=15)

# Fishyscapes Static
transform_fs_static = A.Compose([
    ToTensorV2()
])
fs_static_dataset = FishyscapesStatic(hparams=fishyscapes_static_config, transforms=transform_fs_static)
loader_fs_static = DataLoader(fs_static_dataset, batch_size=1, shuffle=False, num_workers=15)

# Road Anomaly 21
transform_ra_21 = A.Compose([
    A.Resize(height=720, width=1280),
    ToTensorV2()
])
road_anomaly_21_dataset = RoadAnomaly21(hparams=road_anomaly_21_config, transforms=transform_ra_21)
loader_road_anomaly_21 = DataLoader(road_anomaly_21_dataset, batch_size=1, shuffle=False, num_workers=15)


dataset_group = [
    ('road_anomaly', road_anomaly_dataset),
    ('fishyscapes_laf', fs_laf_dataset),
    ('fishyscapes_static', fs_static_dataset)
]


records = edict()
save_path = 'metrics/ood_metrics_bootstrap.pkl'
if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
        records = pickle.load(f)
print("Initial Records", records)

def get_model(config_path, model_path):
    
    args = edict({'config_file': config_path, 'eval-only':True, 'opts':[]})
    config = setup(args)

    model = Trainer.build_model(config)
    DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR).resume_or_load(
        model_path, resume=False
    )
    model.cuda()
    _ = model.eval()
    
    return model

def get_logits(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x[0].cuda()}])
        
    return out[0]['sem_seg'].unsqueeze(0)


def save_records():
    

    print(records)
    with open(save_path, 'wb') as f:
        pickle.dump(records, f)

def bootstrap_evaluation(evaluator, dataset):

    results = edict()

    dataset_size = len(dataset)
    sample_size = int(dataset_size * args.ratio)
    
    for i in range(args.trials):

        indices = np.random.choice(np.arange(dataset_size), sample_size, replace=False)
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.num_workers)

        anomaly_score, ood_gts, _ = evaluator.compute_max_logit_scores(
            loader=loader,
            device=torch.device(args.device),
            return_preds=True,
        )

        metrics = evaluator.evaluate_ood(
            anomaly_score=anomaly_score,
            ood_gts=ood_gts,
            verbose=False
        )

        for k, v in metrics.items():
            if k not in results:
                results[k] = []
            results[k].extend([v])

    means = edict()
    stds = edict()
    for k, v in results.items():
        
        values = np.array(v)
        means[k] = values.mean() * 100.0
        stds[k] = values.std() * 100.0
    
    return means, stds


def run_evaluations(model_name, config_path, model_path):

    model = get_model(config_path=config_path, model_path=model_path)
    evaluator = OODEvaluator(model, get_logits)
    records[model_name] = edict()
    
    for name, dataset in dataset_group:
        metrics_mean, metrics_std = bootstrap_evaluation(evaluator, dataset)
        records[model_name][name] = edict(
            mean=metrics_mean,
            std=metrics_std
        )

        save_records()




def main():

    models_list = os.listdir('model_logs')
    exclude = [ 
        "swin_l_official",
        "mask2former_swin_b_replication",
        "mask2former_per_pixel",
        "mask2former_forced_partition",
        "mask2former_dec_layers_4"
    ]
    for model_name in models_list:
        if model_name  in exclude:
            continue
        if model_name[-4:] == 'yaml':
            continue
        config_path = os.path.join('model_logs', model_name, 'config.yaml')
        
        model_path = os.path.join('model_logs', model_name, 'model_final.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join('model_logs', model_name, 'model_final.pkl')
       
        run_evaluations(model_name, config_path, model_path)



if __name__ == '__main__':
    main()