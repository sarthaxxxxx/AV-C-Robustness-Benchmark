import argparse
import os
# os.environ['MPLCONFIGDIR'] = './plt/'
import sys
import torch
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
import warnings
from tqdm import tqdm
from utilities import accuracy, seed_everything
from adaptation import Source


# TTA for the cav-mae-finetuned model
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='vggsound', choices=['vggsound', 'ks50', 'epic-kitchen'], help='dataset name')
parser.add_argument("--json-root", type=str, default='utils/vggsound', help="validation data json")
parser.add_argument("--label-csv", type=str, default='data/VGGSound/class_labels_indices_vgg.csv', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=309, help="number of classes")
parser.add_argument("--model", type=str, default='cav-mae-ft', help="the model used")
parser.add_argument("--dataset_mean", type=float, default=-5.081, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default=4.4849, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, default=1024, help="the input length in frames")
# parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
# parser.add_argument("--pretrain_path", type=str, default='/xlearning/mouxing/workspace/MM-TTA/egs/kinetics/exp/testmae02-k50-cav-mae-ft-1e-4-2-0.5-1-bs32-ldaFalse-multimodal-fzFalse-h10-a5/models/audio_model_wa.pth', help="pretrained model path")
parser.add_argument("--pretrain_path", type=str, default='ckpt/vgg_65.5.pth', help="pretrained model path")
parser.add_argument("--gpu", type=str, default='0', help="gpu device number")
parser.add_argument("--testmode", type=str, default='multimodal', help="how to test the model")
parser.add_argument('--corruption-modality', type=str, default='both', choices=['video', 'audio', 'none', 'both'], help='which modality to be corrupted')
# parser.add_argument('--data-val', type=str, default='/xlearning/mouxing/workspace/MM-TTA/audioset-processing/data/ks50_test_json_files/gaussian_noise/severity_5.json', help='path to the validation data json')
parser.add_argument('--severity-start', type=int, default=5, help='the start severity of the corruption')
parser.add_argument('--severity-end', type=int, default=5, help='the end severity of the corruption')


args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

if args.dataset == 'vggsound':
    args.n_class = 309


corruption_list = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'speckle_noise',
    'snow',
    'frost',
    'spatter',
    ]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


task_accs = []


for idx, corruption in enumerate(corruption_list):
    print('*'*50)
    print('Now handling: ', corruption)
    print('*'*50)
    for severity in range(args.severity_start, args.severity_end+1):
        print("Now handling severity: ", severity)
        if args.corruption_modality == 'both':
            # data_val = 'utils/vggsound/both/{}/severity_{}.json'.format(corruption, severity)
            data_val = '/home/jovyan/workspace/AV-C-Robustness-Benchmark/utils/epic-kitchen/both/{}/severity_{}.json'.format(corruption, severity)


        # all exp in this work is based on 224 * 224 image
        im_res = 224
        val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                            'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

        shuffle_p = False # originally True in READ
        data_loader = torch.utils.data.DataLoader(
            dataloader.AVDataset(data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size, shuffle=shuffle_p, num_workers=args.num_workers, pin_memory=True, drop_last=False) # shuffle True??????

        # if idx == 0:
        if args.model == 'cav-mae-ft':
            print('test a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
            va_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
        else:
            raise ValueError('model not supported')

        if args.pretrain_path == 'None':
            warnings.warn("Note no pre-trained models are specified.")
        else:
            # TTA based on a CAV-MAE finetuned model
            mdl_weight = torch.load(args.pretrain_path)
            if not isinstance(va_model, torch.nn.DataParallel):
                va_model = torch.nn.DataParallel(va_model)
            miss, unexpected = va_model.load_state_dict(mdl_weight, strict=False)
            print('now load cav-mae finetuned weights from ', args.pretrain_path)
            print(miss, unexpected)
        # exit()
        # evaluate with multiple frames
        if not isinstance(va_model, torch.nn.DataParallel):
            va_model = torch.nn.DataParallel(va_model)

        va_model.to(device)

        va_model = Source.configure_model(va_model)
        trainables = [p for p in va_model.parameters() if p.requires_grad]
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in va_model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

        params, param_names = Source.collect_params(va_model)

        inference_model = Source.Source(va_model, None, device, args, steps=1, episodic=False)
        inference_model.eval()

        total_acc = 0

        with torch.no_grad():
            data_bar = tqdm(data_loader)
            batch_accs = []

            for idx, (a, v, y) in enumerate(data_bar):
                a, v, y = a.to(device), v.to(device), y.to(device)
                x = (a, v)
                outputs, _ = inference_model(x)
                batch_accs.append(accuracy(outputs[0], y)[0].item())
                # data_bar.set_description(f'Batch#{idx} :  ACC#{accuracy(outputs[0], y):.2f}')

            total_acc = np.mean(batch_accs)
            print('Total accuracy: ', total_acc)


        task_accs.append(total_acc)


print('*'*50)
print('Task accuracies: ', task_accs)


print('===> Summary <===')
print('===> Mean accuracy: ', np.mean(task_accs))
print('===> Std accuracy: ', np.std(task_accs))





if __name__ == '__main__':
    pass