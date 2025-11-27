
from utils.train import restore_checkpoint
from utils.args import get_arguments
from utils.init import init_cuda, init_loss, init_model, init_optimizer, init_train_dataloader
from modules.runner import Runner


def main():
    print("Running training")

    args = get_arguments('train')

    init_cuda(args.cuda, args.cupy, args.seed)

    print("Building model...")

    model = init_model(args.model, args.backend,
                       args.inChannels, args.classes, args.model_depth, args)

    criterion = init_loss(args.loss, args.backend, args.classes, args)

    if args.cuda:
        model = model.cuda()
        if isinstance(criterion, list):
            for c in criterion:
                c = c.cuda()
        else:
            criterion = criterion.cuda()

    optimizer, scheduler = init_optimizer(
        args.opt, model, args.lr, args.weight_decay, args.no_scheduler)

    print("Loading train & valid dataset...")

    train_loader, valid_loader = init_train_dataloader(
        args.dataset_path, args.batchSz, args.seed, args.cuda, args.cupy, args.amp, args.test_fraction, args.z_size)

    print("Starting training...")

    start_epoch = 1
    if args.resume:
        start_epoch = restore_checkpoint(
            model,
            args.resume, optimizer=optimizer, scheduler=scheduler, ref_ckpt_file=args.resume_ref) + 1
    runner = Runner(args, model, criterion, optimizer, lr_scheduler=scheduler, train_data_loader=train_loader,
                    valid_data_loader=valid_loader, start_epoch=start_epoch)
    runner.train()

    print("Training finished")


if __name__ == '__main__':
    main()
