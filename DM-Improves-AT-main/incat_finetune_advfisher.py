import csv
import json
import os
import time

import torch
import torch.nn as nn

from core.data import SEMISUP_DATASETS, get_data_info, load_data
from core.utils import Logger, Trainer, format_time, parser_train, seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_TO_SEMISUP = {
    'cifar10': 'cifar10s',
    'cifar100': 'cifar100s',
    'svhn': 'svhns',
    'tiny-imagenet': 'tiny-imagenets',
}


def resolve_fisher_data(base_data):
    if base_data in BASE_TO_SEMISUP:
        return BASE_TO_SEMISUP[base_data]
    return base_data


def load_target_loaders(args):
    data_dir = os.path.join(args.data_dir, args.data)
    loaded = load_data(
        data_dir,
        args.batch_size,
        args.batch_size_validation,
        use_augmentation=args.augment,
        use_consistency=False,
        shuffle_train=True,
        aux_data_filename=None,
        unsup_fraction=None,
        validation=False,
    )
    _, _, train_loader, test_loader = loaded
    eval_loader = test_loader
    eval_source = 'test'
    if not args.no_validation_split:
        val_data = resolve_fisher_data(args.data)
        if val_data in SEMISUP_DATASETS:
            val_dir = os.path.join(args.data_dir, val_data)
            val_loaded = load_data(
                val_dir,
                args.batch_size,
                args.batch_size_validation,
                use_augmentation='none',
                use_consistency=False,
                shuffle_train=False,
                aux_data_filename=None,
                unsup_fraction=0.0,
                validation=True,
            )
            _, _, _, _, _, eval_loader = val_loaded
            eval_source = 'validation'
    return data_dir, train_loader, test_loader, eval_loader, eval_source


def load_fisher_loader(args):
    if args.fisher_data == 'target':
        _, train_loader, _, _, _ = load_target_loaders(args)
        return train_loader

    fisher_data = resolve_fisher_data(args.data)
    data_dir = os.path.join(args.data_dir, fisher_data)
    validation = (fisher_data in SEMISUP_DATASETS) and (not args.no_validation_split)
    loaded = load_data(
        data_dir,
        args.batch_size,
        args.batch_size_validation,
        use_augmentation=args.augment,
        use_consistency=False,
        shuffle_train=True,
        aux_data_filename=args.fisher_aux_data_filename,
        unsup_fraction=args.fisher_unsup_fraction,
        validation=validation,
    )
    if validation:
        _, _, _, train_loader, _, _ = loaded
    else:
        _, _, train_loader, _ = loaded
    return train_loader


def compute_fisher_diagonal(model, dataloader, max_batches=100):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    fisher = {name: torch.zeros_like(p, device=device) for name, p in trainable}
    used_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        valid = y >= 0
        if valid.sum().item() == 0:
            continue
        x = x[valid]
        y = y[valid]

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        grads = torch.autograd.grad(loss, [p for _, p in trainable], retain_graph=False, create_graph=False)
        for (name, _), grad in zip(trainable, grads):
            fisher[name] += grad.detach() * grad.detach()
        used_batches += 1

    if used_batches == 0:
        raise RuntimeError('No labeled batch found while estimating Fisher.')

    for name in fisher:
        fisher[name] /= float(used_batches)
    return fisher


def fisher_penalty(model, reference_params, fisher_diag):
    penalty = torch.zeros((), device=device)
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in fisher_diag:
            continue
        delta = param - reference_params[name]
        penalty = penalty + (fisher_diag[name] * delta * delta).sum()
    return penalty


def main():
    parser = parser_train()
    parser.add_argument('--source-weights', type=str, required=True, help='Checkpoint path of diffusion-trained model (weights-best.pt).')
    parser.add_argument('--fisher-lambda', type=float, default=1e-4, help='Regularization strength for Fisher penalty.')
    parser.add_argument('--fisher-batches', type=int, default=100, help='Batches used to estimate Fisher diagonal.')
    parser.add_argument('--fisher-data', choices=['target', 'mixed'], default='target', help='Data used for Fisher estimation.')
    parser.add_argument('--fisher-aux-data-filename', type=str, default='', help='Aux NPZ path when --fisher-data mixed.')
    parser.add_argument('--fisher-unsup-fraction', type=float, default=0.7, help='Generated ratio for mixed Fisher data.')
    parser.add_argument('--no-validation-split', action='store_true', default=False, help='Skip semi-supervised validation split when loading Fisher mixed data.')
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.desc)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, 'log-incat.log'))
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    seed(args.seed)
    data_dir, train_loader, test_loader, eval_loader, eval_source = load_target_loaders(args)
    info = get_data_info(data_dir)

    trainer = Trainer(info, args)
    trainer.load_model(args.source_weights)
    trainer.init_optimizer(args.num_adv_epochs)

    logger.log('Using device: {}'.format(device))
    logger.log('Loaded source weights: {}'.format(args.source_weights))
    logger.log('Checkpoint selection source: {} adversarial accuracy'.format(eval_source))
    logger.log('INCAT objective: adversarial loss + Fisher (fisher_lambda={})'.format(args.fisher_lambda))

    fisher_loader = load_fisher_loader(args)
    logger.log('Estimating Fisher diagonal...')
    fisher_diag = compute_fisher_diagonal(trainer.model, fisher_loader, max_batches=args.fisher_batches)
    reference = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
    logger.log('Fisher estimation done.')

    rows = []
    best_eval_adv = -1.0
    best_path = os.path.join(log_dir, 'weights-best.pt')
    stats_path = os.path.join(log_dir, 'stats_incat.csv')

    for epoch in range(1, args.num_adv_epochs + 1):
        start_t = time.time()
        trainer.model.train()
        train_loss = 0.0
        train_clean_acc = 0.0
        train_adv_acc = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            base_loss, batch_metrics = trainer.adversarial_loss(x, y)
            reg = fisher_penalty(trainer.model, reference, fisher_diag)
            loss = base_loss + args.fisher_lambda * reg

            # trainer.adversarial_loss() clears grads internally.
            loss.backward()

            if args.clip_grad:
                nn.utils.clip_grad_norm_(trainer.model.parameters(), args.clip_grad)
            trainer.optimizer.step()
            if args.scheduler in ['cyclic']:
                trainer.scheduler.step()

            train_loss += loss.item()
            if 'clean_acc' in batch_metrics:
                train_clean_acc += batch_metrics['clean_acc']
            if 'adversarial_acc' in batch_metrics:
                train_adv_acc += batch_metrics['adversarial_acc']
            n_batches += 1

        if args.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            trainer.scheduler.step()

        test_clean = trainer.eval(test_loader)
        test_adv = ''
        if epoch % args.adv_eval_freq == 0 or epoch == args.num_adv_epochs:
            test_adv = trainer.eval(test_loader, adversarial=True)

        eval_adv = trainer.eval(eval_loader, adversarial=True)

        if eval_adv >= best_eval_adv:
            best_eval_adv = eval_adv
            trainer.save_model(best_path)

        row = {
            'epoch': epoch,
            'train_loss': train_loss / max(1, n_batches),
            'train_clean_acc': train_clean_acc / max(1, n_batches) if train_clean_acc > 0 else '',
            'train_adversarial_acc': train_adv_acc / max(1, n_batches) if train_adv_acc > 0 else '',
            'test_clean_acc': float(test_clean),
            'test_adversarial_acc': test_adv,
            'eval_adversarial_acc': float(eval_adv),
        }
        rows.append(row)
        with open(stats_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)

        logger.log(
            'Epoch {:03d} | loss={:.4f} | test_clean={:.2f}% | time={}'.format(
                epoch, row['train_loss'], test_clean * 100, format_time(time.time() - start_t)
            )
        )
        if test_adv != '':
            logger.log('Epoch {:03d} | test_adv={:.2f}%'.format(epoch, test_adv * 100))
        logger.log('Epoch {:03d} | eval_adv({})={:.2f}%'.format(epoch, eval_source, eval_adv * 100))

    logger.log('INCAT fine-tuning completed. Best eval_adv({}) {:.2f}%'.format(eval_source, best_eval_adv * 100))
    logger.log('Saved best checkpoint to {}'.format(best_path))


if __name__ == '__main__':
    main()