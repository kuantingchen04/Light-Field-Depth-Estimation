import os
import numpy as np

# revise from
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


min_depth = 1e-3
max_depth = 80
# Get GT
#gt_npy_dir = 'datasets/scene12_v2_400/val/A'  ## Correct GT
gt_npy_dir = 'datasets/scene12_v3_400/val/A'  ## Correct GT

test_npy_dir = 'test/npy'
data_npy = [os.path.join(gt_npy_dir, name) for name in os.listdir(gt_npy_dir)];
data_npy.sort()
test_npy = [os.path.join(test_npy_dir, name) for name in os.listdir(test_npy_dir)];
test_npy.sort()
num_samples = len(data_npy)
print num_samples

rms = np.zeros(num_samples, np.float32)
log_rms = np.zeros(num_samples, np.float32)
abs_rel = np.zeros(num_samples, np.float32)
sq_rel = np.zeros(num_samples, np.float32)
d1_all = np.zeros(num_samples, np.float32)
a1 = np.zeros(num_samples, np.float32)
a2 = np.zeros(num_samples, np.float32)
a3 = np.zeros(num_samples, np.float32)

for i in range(num_samples):
    gt_depth = np.load(data_npy[i]) / 1000
    pred_depth = np.load(test_npy[i]) / 1000

    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth

    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

    abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))

