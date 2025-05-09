import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="ABL Defense Configuration")

    # Paths
    parser.add_argument('--isolation_model_root', type=str, default='/content/drive/MyDrive/ABL/weight/ABL_results',
                        help='Path to save isolated models')
    parser.add_argument('--unlearning_root', type=str, default='/content/drive/MyDrive/ABL/weight/ABL_results/WRN-16-1-unlearning_epoch5.tar',
                        help='Path to save unlearned models')
    parser.add_argument('--log_root', type=str, default='/content/drive/MyDrive/ABL/logs', help='Path for logging results')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'GTSRB', 'ImageNet'])
    parser.add_argument('--model_name', type=str, default='WRN-16-1', help='Model architecture')
    parser.add_argument('--isolate_data_root', type=str, default='/content/drive/MyDrive/ABL/isolation_data/',
                        help='Path to store isolated samples')

    # Training Hyperparameters
    parser.add_argument('--print_freq', type=int, default=200, help='Print frequency during training')
    parser.add_argument('--tuning_epochs', type=int, default=60, help='Number of tuning epochs')
    parser.add_argument('--finetuning_ascent_model', type=bool, default=False, help='Enable fine-tuning')
    parser.add_argument('--finetuning_epochs', type=int, default=60, help='Number of fine-tuning epochs')
    parser.add_argument('--unlearning_epochs', type=int, default=5, help='Number of unlearning epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--lr_finetuning_init', type=float, default=0.1, help='Fine-tuning learning rate')
    parser.add_argument('--lr_unlearning_init', type=float, default=5e-4, help='Unlearning learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    # Isolation Parameters
    parser.add_argument('--num_class', type=int, default=10, help='Number of classes')
    parser.add_argument('--isolation_ratio', type=float, default=0.01, help='Ratio of isolation data')
    parser.add_argument('--dynamic_isolation', action='store_true', default=True,
                        help='Enable dynamic isolation ratio based on loss variance')

    # Gradient Ascent Unlearning
    parser.add_argument('--gradient_ascent_type', type=str, default='Flooding', choices=['LGA', 'Flooding'],
                        help='Type of gradient ascent for unlearning')
    parser.add_argument('--gamma', type=float, default=0.5, help='Value for LGA method')
    parser.add_argument('--flooding', type=float, default=0.5, help='Value for Flooding method')
    parser.add_argument('--adv_training_weight', type=float, default=0.5,
                        help='Weight for adversarial loss during hybrid unlearning')
    parser.add_argument('--re_isolation_interval', type=int, default=5,
                        help='Epoch interval for re-isolation during unlearning')

    # Learning Rate Scheduler
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'step', 'constant'],
                        help='Learning rate scheduler type')

    # Model Saving
    parser.add_argument('--threshold_clean', type=float, default=70.0, help='Threshold for saving weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='Threshold for saving weight')
    parser.add_argument('--save', action='store_true', default=True, help='Save models and logs')
    parser.add_argument('--interval', type=int, default=5, help='Save model at every N epochs')

    # Miscellaneous
    parser.add_argument('--cuda', action='store_true', default=True, help='Enable CUDA')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--note', type=str, default='try', help='Note for this run')

    # Backdoor Attack Parameters
    parser.add_argument('--inject_portion', type=float, default=0.1, help='Ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=0, help='Class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='Type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='Type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='Width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='Height of trigger pattern')
    
    # Learning Rate Scheduler
    parser.add_argument('--lr_decay_epochs', type=list, default=[30, 60, 90], help='Epochs at which to decay LR')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor to decay LR')

    return parser