import ipdb
import numpy as np

from jsonargparse import ArgumentParser, ActionConfigFile

import torch
from torch.utils.tensorboard import SummaryWriter


def get_params():
    shell_parse = ArgumentParser()
    shell_parse.add_argument('--trainer', type=str, default='', help='trainer for the job')
    shell_parse.add_argument('--comments', type=str, default='', help='comments for the job')
    shell_parse.add_argument('--device', type=str, help='device to run job')
    shell_parse.add_argument('--eval', action='store_true', help='whether to go into eval mode directly')
    shell_parse.add_argument('--resume', action='store_true', help='resume training from last cp epoch')
    shell_parse.add_argument('--config_file', type=str, help='path to the json file containing params')
    shell_params = shell_parse.parse_args()
    return shell_params


def get_trainer(shell_params):
    parser = ArgumentParser()
    parser.add_argument('--trainer', type=str, default='', help='trainer for the job')
    parser.add_argument('--comments', type=str, default='', help='comments for the job')
    parser.add_argument('--device', type=str, help='device to run job')
    parser.add_argument('--eval', action='store_true', help='whether to go into eval mode directly')
    parser.add_argument('--resume', action='store_true', help='resume training from last cp epoch')
    parser.add_argument('--config_file', action=ActionConfigFile, help='path to the yaml file containing params')

    parser.add_argument('--loss', type=str, default='', help='loss function')
    parser.add_argument('--train_epochs', type=int, help='training epochs')

    match shell_params.trainer:
        case 'recursive_coding':
            from trainer.recursive_coding import RecursiveCoding
            parser = RecursiveCoding.get_parser(parser)
            params = parser.parse_args()
            trainer = RecursiveCoding(params.dataset.dataset, params.loss, params.staged_training, params, params.resume)
        case _:
            raise NotImplementedError

    return trainer, params


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    writer = None

    trainer, params = get_trainer(get_params())
    channel_params = params.channel
    print(params)
    print(str(trainer))

    if params.eval:
        test_snrs = np.arange(channel_params.test_snr[0], channel_params.test_snr[1]+1)
        # TODO add eval training loop; dataset etc
    else:
        test_snrs = [channel_params.eval_snr]

    while trainer.epoch < params.train_epochs and not params.eval:
        trainer.training()
        train_loss, _, train_aux = trainer(channel_params.train_snr)

        trainer.validate()
        val_loss, terminate, val_aux = trainer(channel_params.eval_snr)

        if writer is None:
            writer = SummaryWriter()

        writer.add_scalars('Train/Losses',
                           {f'{str(trainer)}': train_loss}, trainer.epoch)
        for data_key in train_aux:
            writer.add_scalars(f'Train/{data_key}',
                               {f'{str(trainer)}': train_aux[data_key]}, trainer.epoch)

        writer.add_scalars('Validate/Losses',
                           {f'{str(trainer)}': val_loss}, trainer.epoch)
        for data_key in val_aux:
            writer.add_scalars(f'Validate/{data_key}',
                               {f'{str(trainer)}': val_aux[data_key]}, trainer.epoch)

        if terminate or trainer.epoch >= params.train_epochs:
            print('Training complete'); break

    print('Evaluating...')
    # TODO potentially separate this from main
    for snr in test_snrs:
        trainer.evaluate()
        _, _, eval_aux = trainer(snr)

        if writer is None:
            writer = SummaryWriter()

        for data_key in eval_aux:
            writer.add_scalars(f'Evaluate/{data_key}',
                               {f'{str(trainer)}': eval_aux[data_key]}, snr)

    writer.close()
