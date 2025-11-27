
from utils.train import restore_checkpoint
from utils.args import get_arguments
from utils.init import init_cuda, init_loss, init_model, init_infer_dataloader
from modules.runner import Runner


def main():
    print("Running inference")

    args = get_arguments('infer')

    init_cuda(args.cuda, args.cupy, args.seed, True)

    print("Building model...")

    model = init_model(args.model, args.backend,
                       args.inChannels, args.classes, args.model_depth, args)
    restore_checkpoint(model, args.ckpt)
    if args.cuda:
        model = model.cuda()

    criterion = init_loss(args.loss, args.backend, args.classes, args)

    print("Loading dataset...")

    infer_loader = init_infer_dataloader(
        args.dataset_path, args.seed, args.cuda, args.cupy, args.amp, args.z_size)

    print("Starting inference...")

    runner = Runner(args, model, criterion, valid_data_loader=infer_loader)
    if (args.timing):
        runner.timing()
    else:
        runner.infer()

    print("Inference finished")


if __name__ == '__main__':
    main()
