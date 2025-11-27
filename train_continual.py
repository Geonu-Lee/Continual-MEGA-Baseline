import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from CLIP.clip import create_model
from CLIP.adapter import CLIPAD
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from loss import FocalLoss, BinaryDiceLoss
from dataset.continual import ImageDataset
import csv
import wandb
import datetime
import logging
from CoOp import PromptMaker
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_logger(output_dir):
    # set log file
    log_file = f"{output_dir}/output.log"
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--img_size', type=int, default=336)
    parser.add_argument("--epoch", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--prompt_lr", type=float, default=0.0001, help="prompt learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--gpu', type=str, default="2")
    parser.add_argument("--noise_sigma", type=float, default=0.25, help="noise sigma")
    parser.add_argument("--test_per_epoch", type=int, default=20, help="test per epoch")
    parser.add_argument("--meta_file", type=str, default="meta_files/scenario1_5classes_tasks.json")
    parser.add_argument("--base_meta_file", type=str, default="meta_files/scenario1_base.json")    
    parser.add_argument("--num_tasks", type=int, default=12, help="number of tasks")
    parser.add_argument("--n_learnable_token", type=int, default=8, help="number of learnable token")
    parser.add_argument("--data_root", type=str, default="data")

    # load base folder
    parser.add_argument("--base_folder", type=str, default="results/scenario1")
    parser.add_argument("--task_id", type=int, default=1, help="current task id")

    parser.add_argument("--task_class_num", type=str, default="5classes", help="number of classes in each task")

    args = parser.parse_args()

    setup_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")

    # date = datetime.datetime.now()
    save_path = os.path.join(args.base_folder, f"{args.task_class_num}/task_{args.task_id}")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # for logging
    logger = get_logger(save_path)
    logger.info("start training meta: {}, task_id: {}".format(args.meta_file, args.task_id))
    logger.info(args)
    
    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    
    # prompt learner
    prompts = {
        "normal": [
        "This is an example of a normal object",
        "This is a typical appearance of the object",
        "This is what a normal object looks like",
        "A photo of a normal object",
        "This is not an anomaly",
        "This is an example of a standard object.",
        "This is the standard appearance of the object.",
        "This is what a standard object looks like.",
        "A photo of a standard object.",
        "This object meets standard characteristics."
    ],
        "abnormal": [
        "This is an example of an anomalous object",
        "This is not the typical appearance of the object",
        "This is what an anomaly looks like",
        "A photo of an anomalous object",
        "An anomaly detected in this object",
        "This is an example of an abnormal object.",
        "This is not the usual appearance of the object.",
        "This is what an abnormal object looks like.",
        "A photo of an abnormal object.",
        "An abnormality detected in this object."
    ]
    }
    clip_model.device = device
    clip_model.to(device)

    prompt_maker = PromptMaker(
        prompts=prompts,
        clip_model=clip_model,
        n_ctx= args.n_learnable_token,
        CSC = True,
        class_token_position=['end'],
    ).to(device)

    model = CLIPAD(clip_model=clip_model, features=args.features_list, noise_sigma=args.noise_sigma)
    model.to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True
        
    base_path = f"{args.base_folder}/checkpoint.pth"
    base_checkpoint = torch.load(base_path, map_location=device)

    
    prompt_maker.prompt_learner.load_state_dict(base_checkpoint['prompt_state_dict'])
    model.adapters.load_state_dict(base_checkpoint['adapters'])
    logger.info(f"load base model from {base_path}")

    ada_optimizer = torch.optim.AdamW(list(model.adapters.parameters()), lr=args.lr)

    # load dataset
    task_all_meta_info = json.load(open(args.meta_file, 'r'))
    task_meta = task_all_meta_info[f"task_{args.task_id}"]

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_dataset = ImageDataset(data_root=args.data_root, meta_file=task_meta, resize=args.img_size, mode="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    # save results
    num_tasks = args.num_tasks + 1
    results_image = np.full((num_tasks, num_tasks), np.nan)  # save for csv image-level
    results_pixel = np.full((num_tasks, num_tasks), np.nan)  # save for csv pixel-level

    # load saved_results
    if args.task_id == 1:
        csv_image = f"{args.base_folder}/results_image.csv"
        csv_pixel = f"{args.base_folder}/results_pixel.csv"
    else:
        csv_image = f"{args.base_folder}/{args.task_class_num}/task_{args.task_id-1}/results_image.csv"
        csv_pixel = f"{args.base_folder}/{args.task_class_num}/task_{args.task_id-1}/results_pixel.csv"

    with open(csv_image, mode="r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if not i == 0:
                results_image[i-1] = row
    with open(csv_pixel, mode="r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if not i == 0:
                results_pixel[i-1] = row

    prompt_maker.eval()
    model.eval()
    for epoch in range(args.epoch):
        logger.info('epoch: {}'.format(epoch+1))
        loss_list = []
        seg_loss_list = []
        det_loss_list = []
        contrast_loss_list = []
        anomaly_seg_loss_list = []
        for data in tqdm(train_loader, desc="Train {} epoch".format(epoch+1)):
            image, mask, cls_name, label = data['image'], data['mask'], data['cls_name'], data['anomaly']

            image = image.to(device)
            with torch.cuda.amp.autocast():
                _, ada_patch_tokens, anomaly_patch_tokens = model(image, True)
                ada_patch_tokens = [p[:, 1:, :] for p in ada_patch_tokens]
                anomaly_patch_tokens = [p[:, 1:, :] for p in anomaly_patch_tokens]

                text_features = prompt_maker()
                seg_loss = 0
                det_loss = 0
                anomaly_seg_loss = 0
                loss = 0
                contrast_loss = torch.tensor(0.).to(device)
                mask = mask.squeeze(1).to(device)
                mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

                label = label.to(device)
                for layer in range(len(ada_patch_tokens)):
                    ada_patch_tokens[layer] = ada_patch_tokens[layer] / ada_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_patch_tokens[layer] = anomaly_patch_tokens[layer] / anomaly_patch_tokens[layer].norm(dim=-1, keepdim=True)

                    anomaly_map = (100.0 * ada_patch_tokens[layer] @ text_features)
                    anomaly_map_fake = (100.0 * anomaly_patch_tokens[layer] @ text_features)
                    
                    # normal
                    if (label==0).sum() > 0:
                        masks_true = torch.zeros(anomaly_map[label==0].shape[0]*anomaly_map[label==0].shape[1]).to(device)
                        seg_loss += loss_ce(anomaly_map[label==0].reshape(-1, 2), masks_true.long())
                        
                        # anomaly fake
                        masks_fake = torch.ones(anomaly_map_fake[label==0].shape[0]*anomaly_map_fake[label==0].shape[1]).to(device)
                        anomaly_seg_loss += loss_ce(anomaly_map_fake[label==0].reshape(-1, 2), masks_fake.long())
                    else:
                        anomaly_seg_loss = torch.tensor(0.).to(device)

                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map_fake = F.interpolate(anomaly_map_fake.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_map_fake = torch.softmax(anomaly_map_fake, dim=1)

                    if (label==1).sum() > 0:
                        seg_loss += loss_focal(anomaly_map[label==1], mask[label==1])  # only train real anomaly mask
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :][label==1], mask[label==1]) # only train real anomaly mask
                
                loss = seg_loss + anomaly_seg_loss
                

                ada_optimizer.zero_grad()
                loss.backward()
                ada_optimizer.step()
                loss_list.append(loss.item())
                seg_loss_list.append(seg_loss.item())
                anomaly_seg_loss_list.append(anomaly_seg_loss.item())

        logger.info("Loss: {}".format(np.mean(loss_list)))
        logger.info("Seg Loss: {}".format(np.mean(seg_loss_list)))
        logger.info("Anomaly Seg Loss: {}".format(np.mean(anomaly_seg_loss_list)))

        if args.wandb:
            wandb.log({"Loss": np.mean(loss_list),
                       "Seg Loss": np.mean(seg_loss_list),
                        "Anomaly Seg Loss": np.mean(anomaly_seg_loss_list)
                       }, step=epoch+1)

        ckp_path = os.path.join(save_path, f'checkpoint.pth')
        torch.save({
                    "prompt_state_dict": prompt_maker.prompt_learner.state_dict(),
                    'adapters': model.adapters.state_dict(),
                    'epoch': epoch+1,
                    "ada_optimizer": ada_optimizer.state_dict(),
                    }, 
                    ckp_path)
    
    # load previous adapters weights and merge (sum)
    new_param = {}
    for name, param in base_checkpoint['adapters'].items():
        param = param.to(device)
        new_param[name] = param.clone()

    for id in range(1, args.task_id + 1):
        task_checkpoint = torch.load(f"{args.base_folder}/{args.task_class_num}/task_{id}/checkpoint.pth", map_location=device)
        logger.info(f"load task_{id} model")
        for name, param in task_checkpoint['adapters'].items():
            param = param.to(device)
            new_param[name] += param.clone()
    # average
    for name, param in new_param.items():
        new_param[name] = param / (args.task_id + 1)

    logger.info(f"merged adapters")
    
    model.adapters.load_state_dict(new_param)
    # save current task adapters
    ckp_path = os.path.join(save_path, f'checkpoint_merged.pth')
    torch.save({
                "prompt_state_dict": prompt_maker.prompt_learner.state_dict(),
                'adapters': model.adapters.state_dict(),
                }, 
                ckp_path)

    # test current task
    logging.info(f"start current task test")
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.eval()

    class_name_list = train_dataset.class_names
    test_dataset_list = [ImageDataset(data_root=args.data_root, meta_file=task_meta, resize=args.img_size, mode="test", test_class=class_name) for class_name in class_name_list]
    test_loader_list = [torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs) for test_dataset in test_dataset_list]

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():

        prompt_maker.eval()
        model.eval()
        text_features = prompt_maker()

        # test all class
        seg_ap_list = []
        img_auc_list = []
        for test_loader, class_name in zip(test_loader_list, class_name_list):
            logger.info(f"start test {class_name}")
            roc_auc_im, seg_ap = test(args, model, test_loader, text_features, device)
            logger.info(f'{class_name} P-AP : {round(seg_ap,4)}')
            logger.info(f'{class_name} I-AUC : {round(roc_auc_im, 4)}')
            seg_ap_list.append(seg_ap)
            img_auc_list.append(roc_auc_im)

        seg_ap_mean = np.mean(seg_ap_list)
        img_auc_mean = np.mean(img_auc_list)

        logger.info(f'Average P-AP : {round(seg_ap_mean,4)}')
        logger.info(f'Average I-AUC : {round(img_auc_mean, 4)}')

    # save results csv (current task)
    seg_ap_mean = round(seg_ap_mean,4)
    img_auc_mean = round(img_auc_mean, 4)
    results_image[args.task_id, args.task_id] = img_auc_mean
    results_pixel[args.task_id, args.task_id] = seg_ap_mean

    # test all previous tasks
    for i in range(args.task_id):
        if i == 0: # base classes
            task_meta = json.load(open(args.base_meta_file, 'r'))
            logging.info(f"start base task test")
        else:
            task_meta = task_all_meta_info[f"task_{i}"]
            logging.info(f"start previous task_{i} test")

        class_name_list = list(task_meta["test"].keys())
        test_dataset_list = [ImageDataset(data_root=args.data_root, meta_file=task_meta, resize=args.img_size, mode="test", test_class=class_name) for class_name in class_name_list]
        test_loader_list = [torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs) for test_dataset in test_dataset_list]

        logger.info(f"start test task {i}")

        with torch.cuda.amp.autocast(), torch.no_grad():
            # test all class
            seg_ap_list = []
            img_auc_list = []
            prompt_maker.eval()
            model.eval()
            text_features = prompt_maker()

            for test_loader, class_name in zip(test_loader_list, class_name_list):
                logger.info(f"start test {class_name}")
                roc_auc_im, seg_ap = test(args, model, test_loader, text_features, device)
                logger.info(f'{class_name} P-AP : {round(seg_ap,4)}')
                logger.info(f'{class_name} I-AUC : {round(roc_auc_im, 4)}')
                seg_ap_list.append(seg_ap)
                img_auc_list.append(roc_auc_im)

            seg_ap_mean = np.mean(seg_ap_list)
            img_auc_mean = np.mean(img_auc_list)

            logger.info(f'Average P-AP : {round(seg_ap_mean,4)}')
            logger.info(f'Average I-AUC : {round(img_auc_mean, 4)}')

        # save results csv (i task)
        seg_ap_mean = round(seg_ap_mean,4)
        img_auc_mean = round(img_auc_mean, 4)
        results_image[args.task_id, i] = img_auc_mean
        results_pixel[args.task_id, i] = seg_ap_mean

        logger.info(f"save results csv task {i}")


    csv_image = f"{save_path}/results_image.csv"
    csv_pixel = f"{save_path}/results_pixel.csv"

    with open(csv_image, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Base"] + ["Task " + str(i + 1) for i in range(num_tasks-1)])
        for row in results_image:
            writer.writerow(row)
    with open(csv_pixel, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Base"] + ["Task " + str(i + 1) for i in range(num_tasks-1)])
        for row in results_pixel:
            writer.writerow(row)

    logger.info(f"save results for csv file")



def test(args, model, test_loader, text_features, device):
    gt_list = []
    gt_mask_list = []

    seg_score_map_zero = []
    image_scores = []
    for data in tqdm(test_loader):
        image, mask, cls_name, label = data['image'], data['mask'], data['cls_name'], data['anomaly']
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ada_patch_tokens = model(image, False)
            ada_patch_tokens = [p[0, 1:, :] for p in ada_patch_tokens]

            anomaly_maps = []
            image_score = 0
            for layer in range(len(ada_patch_tokens)):
                ada_patch_tokens[layer] /= ada_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * ada_patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))

                # image
                anomaly_score = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                image_score += anomaly_score.max()
        
                anomaly_maps.append(anomaly_map)
            score_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            score_map = F.interpolate(score_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=args.img_size, mode='bilinear', align_corners=True)
            score_map = torch.softmax(score_map, dim=1)[:, 1, :, :]
            score_map = score_map.squeeze(0).cpu().numpy()
            seg_score_map_zero.append(score_map)
            image_scores.append(image_score.cpu() / len(ada_patch_tokens))
                        
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(label.cpu().detach().numpy())

            
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(seg_score_map_zero)
    image_scores = np.array(image_scores)

    roc_auc_im = roc_auc_score(gt_list, image_scores)
    seg_pr = average_precision_score(gt_mask_list.flatten(), segment_scores.flatten())

    return roc_auc_im, seg_pr


if __name__ == '__main__':
    main()