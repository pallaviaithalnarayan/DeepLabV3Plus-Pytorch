from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import time

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Mydata, Custompark
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter


from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'mydata', 'customdata'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    # parser.add_argument()

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs (default: 1)")
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    # parser.add_argument("--total_itrs", type=int, default=30e3,
    #                 help="epoch number (default: 30k)")
    parser.add_argument("--total_itrs", type=int, default=100,
                        help="epoch number (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    # parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    # parser.add_argument("--batch_size", type=int, default=16,
    #                     help='batch size (default: 16)')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    # parser.add_argument("--val_interval", type=int, default=100,
    #                     help="epoch interval for eval (default: 100)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 10)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    if opts.dataset == 'mydata':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(size=(768, 768)),  # Ensure all validation images are resized
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    

        # val_transform = et.ExtCompose([
        #     # et.ExtResize( 512 ),
        #     et.ExtToTensor(),
        #     et.ExtNormalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225]),
        # ])

        train_dst = Mydata(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Mydata(root=opts.data_root,
                             split='val', transform=val_transform)   
    
    if opts.dataset == 'customdata':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(size=(768, 768)),  # Ensure all validation images are resized
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Custompark(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Custompark(root=opts.data_root,
                             split='val', transform=val_transform)   

    return train_dst, val_dst



def validate(opts, model, loader, device, metrics, criterion, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    total_val_loss = 0
    correct_val_preds = 0
    total_val_samples = 0

    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)

            # Calculate loss for the batch and accumulate
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()  # Move labels to CPU as numpy array
            correct_val_preds += np.sum(preds == targets)

            # correct_val_preds += (preds == labels).sum().item()
            # total_val_samples += labels.size(0)
            total_val_samples += targets.size  # Total pixels in the batch
            # total_val_samples += labels.numel()  # Total pixels in the batch

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
                
        # Calculate average loss and accuracy over the validation dataset
        avg_val_loss = total_val_loss / len(loader)
        avg_val_acc = (correct_val_preds / total_val_samples) * 100  # Convert to percentage

        score = metrics.get_results()
    return score, ret_samples, avg_val_loss, avg_val_acc

def plot_loss_accuracies(train_losses, train_accuracies, val_losses, val_accuracies):
    # After training, print final losses and accuracies
    print(f"Final Training Losses: {train_losses}")
    print(f"Final Training Accuracies: {train_accuracies}")
    print(f"Final Validation Losses: {val_losses}")
    print(f"Final Validation Accuracies: {val_accuracies}")

    # Plotting the metrics
    plt.figure(figsize=(15, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.savefig("plot_metrics.png")  
    # plt.show()

def main():
    writer = SummaryWriter(log_dir="runs/segmentation")

    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'mydata' or opts.dataset.lower() == 'customdata':
        opts.num_classes = 7

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    image_shape = train_dst[0][0].shape
    print(f'{len(train_dst) = }, {image_shape = }')


    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)

    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    # print(f'{metrics.confusion_matrix=}')

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    if not hasattr(opts, 'loss_type') or opts.loss_type is None:
        opts.loss_type = 'cross_entropy'  # Default to cross-entropy if not set

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        # ***pnaray*** These changes are to exclude weights from last layer
        # Filter out the classifier layer's weights
        state_dict = checkpoint["model_state"]
        # Filter out the classifier layer's weights to avoid mismatched sizes
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.classifier.3")}
        # Load state_dict with strict=False to ignore the classifier layer
        model.load_state_dict(filtered_state_dict, strict=False)
        # Replace the classifier layer with a new one for UAVid's classes
        model.classifier.classifier[3] = nn.Conv2d(256, opts.num_classes, kernel_size=1)
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        else:
            print("Model weights loaded from %s for fine-tuning" % opts.ckpt)

        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples, avg_val_loss, avg_val_acc = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, criterion=criterion, ret_samples_ids=vis_sample_id)
        print(f'{val_score=}')
        print(metrics.to_str(val_score))
        return
    
    # Initialize metric lists

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    interval_loss = 0
    opts.total_itrs = len(train_loader)

    while cur_epochs < opts.num_epochs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        epoch_train_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        batch_times = []
        with tqdm(total=len(train_loader), desc=f"Epoch {cur_epochs}", unit="batch") as pbar:
            for images, labels in train_loader:
                # Start timing for the batch
                batch_start_time = time.time()

            # for (images, labels) in train_loader:
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # End timing for the batch
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time  # Time in seconds
                batch_times.append(batch_time)

                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss
                epoch_train_loss += np_loss

                # Calculate training accuracy for this batch
                preds = outputs.argmax(dim=1)
                correct_train_preds += (preds == labels).sum().item()
                total_train_samples += labels.numel()  # Total number of pixels in the batch
                # total_train_samples += labels.size(0)

                # Update tqdm progress bar
                pbar.set_postfix(loss=interval_loss / 10, batch_time=f"{batch_time:.2f}s")
                pbar.update(1)

                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss / 10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                        (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                    interval_loss = 0.0

        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"Average Batch Time for Epoch {cur_epochs}: {avg_batch_time:.4f} seconds")

        # Calculate epoch-level training loss and accuracy
        # train_loss = epoch_train_loss / len(train_loader)
        train_loss = round(epoch_train_loss / len(train_loader), 4)
        train_acc = round(((correct_train_preds / total_train_samples) * 100), 2)  # Convert to percentage
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        writer.add_scalar("Training Loss/train", train_loss, cur_epochs)
        writer.add_scalar("Training Accuracy/train", train_acc, cur_epochs)

        print(f"Epoch {cur_epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

        # ===== Validation =====
        if (cur_epochs) % opts.val_interval == 0:
            # Save best model based on Mean IoU
            # Perform validation at val_interval
            model.eval()
            val_score, ret_samples, avg_val_loss, avg_val_acc = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, criterion=criterion, ret_samples_ids=vis_sample_id)
            
            avg_val_loss = round(avg_val_loss, 4)
            avg_val_acc = round(avg_val_acc, 2)
            writer.add_scalar("Validation Loss/train", avg_val_loss, cur_epochs)
            writer.add_scalar("Validation Accuracy/train", avg_val_acc, cur_epochs)
            # breakpoint
            # print(metrics.to_str(val_score))
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)

            if val_score['Mean IoU'] > best_score:
                best_score = val_score['Mean IoU']
                save_ckpt(f'checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}.pth')
            
            model.train()
        scheduler.step()

    plot_loss_accuracies(train_losses, train_accuracies, val_losses, val_accuracies)
    writer.close()
    print('exiting')
    

if __name__ == '__main__':
    main()
