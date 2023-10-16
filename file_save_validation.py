import argparse
import os
import numpy as np
from tqdm import tqdm 

def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2

def get_loss(exp_path : str):
    total_loss = 0
    total_epe = 0
    total_acc3d = 0
    total_acc3d_2 = 0
    num_examples = 0

    tensors_path = sorted(os.listdir(os.path.join(exp_path,'out')))
    for path in tqdm(tensors_path):
        loaded_np = np.load(os.path.join(exp_path,'out',path))
        batch_size = loaded_np['pred'].shape[0]
        num_examples += batch_size
        flow_pred = loaded_np['pred']
        flow = loaded_np['flow']
        mask1 = loaded_np['mask1']
        loss = np.mean(mask1 * np.sum((flow_pred - flow) * (flow_pred - flow), -1) / 2.0)
        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(flow_pred, flow, mask1)
        total_epe += epe_3d * batch_size
        total_acc3d += acc_3d * batch_size
        total_acc3d_2+=acc_3d_2*batch_size
        total_loss += loss.item() * batch_size

    return total_loss * 1.0 / num_examples, total_epe * 1.0 / num_examples, total_acc3d * 1.0 / num_examples, total_acc3d_2 * 1.0 / num_examples

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--checkpoint_path', type=str, default='', metavar='N',
                        help='Path of the checkpoint')
    args = parser.parse_args()

    if args.checkpoint_path == '':
        print('Please specify the checkpoint path')
        exit(0)

    test_loss, epe, acc, acc_2 = get_loss(args.checkpoint_path)

    print('==FINAL TEST==')
    print(f'mean test loss: {test_loss :.6f}\tEPE 3D: {epe :.6f}\tACC 3D: {acc :.6f}\tACC 3D 2: {acc_2 :.6f}')




if __name__ == '__main__':
    main()