import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.8, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--model", type=str, default="EDSR")
    parser.add_argument("--dataset", type=str, default="DIV2K_SR")
    parser.add_argument("--scale", type=int, default=4) # SR
    # augmentations
    parser.add_argument("--use_moa", action="store_true")
    parser.add_argument("--augs", nargs="*", default=["none"])
    parser.add_argument("--prob", nargs="*", default=[1.0])
    parser.add_argument("--mix_p", nargs="*")
    parser.add_argument("--alpha", nargs="*", default=[1.0])
    parser.add_argument("--aux_prob", type=float, default=1.0)
    parser.add_argument("--aux_alpha", type=float, default=1.2)
    return parser.parse_args()

def make_template(opt):
    opt.strict_load = opt.test_only

    # model
    if "EDSR" in opt.model:
        opt.num_blocks = 32
        opt.num_channels = 256
        opt.res_scale = 0.1
    if "RCAN" in opt.model:
        opt.num_groups = 10
        opt.num_blocks = 20
        opt.num_channels = 64
        opt.reduction = 16
        opt.res_scale = 1.0
        opt.max_steps = 1000000
        opt.decay = "200-400-600-800"
        opt.gclip = 0.5 if opt.pretrain else opt.gclip
    if "CARN" in opt.model:
        opt.num_groups = 3
        opt.num_blocks = 3
        opt.num_channels = 64
        opt.res_scale = 1.0
        opt.batch_size = 64
        opt.decay = "400"

    # training setup
    if "DN" in opt.dataset or "JPEG" in opt.dataset:
        opt.max_steps = 1000000
        opt.decay = "300-550-800"
    if "RealSR" in opt.dataset:
        opt.patch_size *= opt.scale # identical (LR, HR) resolution

    # evaluation setup
    opt.crop = 6 if "DIV2K" in opt.dataset else 0
    opt.crop += opt.scale if "SR" in opt.dataset else 4

    # note: we tested on color DN task
    if "DIV2K" in opt.dataset or "DN" in opt.dataset:
        opt.eval_y_only = False
    else:
        opt.eval_y_only = True

    # default augmentation policies
    if opt.use_moa:
        opt.augs = ["none","blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
        opt.prob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        opt.alpha = [0.0, 0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        opt.aux_prob, opt.aux_alpha = 1.0, 1.2
        opt.mix_p = None

        if "RealSR" in opt.dataset:
            opt.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]

        if "DN" in opt.dataset or "JPEG" in opt.dataset:
            opt.prob = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        if "CARN" in opt.model and not "RealSR" in opt.dataset:
            opt.prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt