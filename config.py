import argparse


def config():
    parser = argparse.ArgumentParser('MUSIC')
    parser.add_argument('--trial', type=int, default=0,
                        help='Exp number.')
    parser.add_argument('--folder', type=str, default='',
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--ckpt', type=str, default='',
                        help='Path to the trained model.')
    parser.add_argument('--model', type=str, default='resnet12',
                        help='Model name.')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gpu', '-g', type=str, default='0')
    parser.add_argument('--mode', type=str, default='evaluate',
                        help='evaluate')
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--num_shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num_test_ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--tmp_model', type=str, default='',
                        help='Intermediate store model name.')
    parser.add_argument('--output-folder', type=str, default='./ckpt',
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--num-batches', type=int, default=600,
                        help='Number of batches for test(default: 600).')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4).')
    parser.add_argument('--unlabel', type=int, default=50,
                        help='Number of unlabeled examples per class, 0 means TFSL setting.')
    parser.add_argument('--nl_thres', type=float, default=0.2)
    parser.add_argument('--img_size', type=int, default=84)
    args = parser.parse_args()
    return args
