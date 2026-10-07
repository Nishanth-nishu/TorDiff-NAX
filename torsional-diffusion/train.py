import math, os, random, signal, sys, torch, yaml
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from rdkit import RDLogger
from utils.dataset import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file
from utils.boltzmann import BoltzmannResampler
from argparse import Namespace

RDLogger.DisableLog('rdApp.*')

"""
    Training procedures for both conformer generation and Botzmann generators
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""


_STOP = {'requested': False}


def _rng_state():
    st = {'py_rng': random.getstate(), 'np_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state()}
    if torch.cuda.is_available():
        st['cuda_rng'] = torch.cuda.get_rng_state_all()
    return st


def save_checkpoint(path, state):
    """[round2 resume] atomic write: a kill during torch.save can never leave a truncated last_model.pt"""
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume_state(args, model, optimizer, scheduler):
    """[round2 resume] returns (start_epoch, best_val_loss, best_epoch). Restores model, optimizer, scheduler and the
    python / numpy / torch (+cuda) RNG states. The training-L samplers (S2 jitter, S3 mix, S4 lambda) keep no state of
    their own: they draw from these RNGs in the main process, and DataLoader workers are re-seeded every epoch from the
    main torch RNG, so restoring the RNG states is enough. A resumed run is still not bit-identical to an uninterrupted
    one (CUDA / e3nn non-determinism, see IMPLEMENTATION.md §3)."""
    ck = os.path.join(args.log_dir, 'last_model.pt')
    if not os.path.exists(ck):
        print('RESUME requested but no last_model.pt: starting from epoch 0', flush=True)
        return 0, math.inf, 0
    s = torch.load(ck, map_location='cpu')
    model.load_state_dict(s['model'], strict=True)
    optimizer.load_state_dict(s['optimizer'])
    if scheduler is not None and s.get('scheduler') is not None:
        scheduler.load_state_dict(s['scheduler'])
    if 'py_rng' in s:
        random.setstate(s['py_rng'])
        np.random.set_state(s['np_rng'])
        torch.set_rng_state(s['torch_rng'])
        if 'cuda_rng' in s and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s['cuda_rng'])
    start = s['epoch'] + 1
    best_val_loss, best_epoch = s.get('best_val_loss', math.inf), s.get('best_epoch', 0)
    print(f'RESUMED from {ck}: last finished epoch {s["epoch"]}, continuing at epoch {start}; '
          f'best_val_loss {best_val_loss} (epoch {best_epoch}); rng restored: {"py_rng" in s}', flush=True)
    return start, best_val_loss, best_epoch


def train(args, model, optimizer, scheduler, train_loader, val_loader):
    best_val_loss = math.inf
    best_epoch = 0
    start_epoch = 0
    if getattr(args, 'resume', False):
        start_epoch, best_val_loss, best_epoch = load_resume_state(args, model, optimizer, scheduler)

    print("Starting training...")
    for epoch in range(start_epoch, args.n_epochs):

        stats = {}
        train_loss, base_train_loss = train_epoch(model, train_loader, optimizer, device, stats=stats,
                                                  fail_on_nan=getattr(args, 'fail_on_nan', False),
                                                  limit_iters=getattr(args, 'limit_train_iters', 0))
        print("Epoch {}: Training Loss {}  base loss {}".format(epoch, train_loss, base_train_loss))
        if getattr(args, 'log_timing', False):
            # [round2 D5] data-wait share of the epoch (plan 3 timer); decides 4 CPUs / 3 workers vs 10 / 8
            print("Epoch {}: TIMING epoch_time {:.1f}s data_wait {:.1f}s ({:.2f}%) iters {} ({:.2f} it/s)".format(
                epoch, stats['epoch_time'], stats['data_wait'], 100 * stats['data_wait'] / max(stats['epoch_time'], 1e-9),
                stats['n_iter'], stats['n_iter'] / max(stats['epoch_time'], 1e-9)), flush=True)

        val_loss, base_val_loss = test_epoch(model, val_loader, device)
        print("Epoch {}: Validation Loss {} base loss {}".format(epoch, val_loss, base_val_loss))

        if scheduler:
            scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))

        # [round2 resume] same keys as before + best val / RNG states for --resume; written atomically
        save_checkpoint(os.path.join(args.log_dir, 'last_model.pt'), {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            **_rng_state(),
        })
        if _STOP['requested'] and epoch < args.n_epochs - 1:
            print(f'STOP requested (SIGUSR1): checkpoint of epoch {epoch} written, exiting with 99 for requeue',
                  flush=True)
            sys.exit(99)

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))


def boltzmann_train(args, model, optimizer, train_loader, val_loader, resampler):
    print("Starting training...")

    val_ess = val_loader.dataset.resample_all(resampler, temperature=args.temp)
    print(f"Initial val ESS: Mean {np.mean(val_ess):.4f} Median {np.median(val_ess):.4f}")
    best_val = val_ess

    for epoch in range(args.n_epochs):
        if args.adjust_temp:
            train_loader.dataset.boltzmann_resampler.temp = (3000 - args.temp) / (epoch + 1) + args.temp

        train_loss, base_train_loss = train_epoch(model, train_loader, optimizer, device)
        print("Epoch {}: Training Loss {}  base loss {}".format(epoch, train_loss, base_train_loss))
        if epoch % 5 == 0:
            val_ess = val_loader.dataset.resample_all(resampler, temperature=args.temp)
            print(f"Epoch {epoch} val ESS: Mean {np.mean(val_ess):.4f} Median {np.median(val_ess):.4f}")

            if best_val > val_ess:
                best_val = val_ess
                torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(args.log_dir, 'last_model.pt'))


if __name__ == '__main__':
    args = parse_train_args()
    if args.resume:
        # [round2 resume] SIGUSR1 (sbatch --signal=B:USR1@900, forwarded by ablation_train_array.sbatch): finish the
        # current epoch and its checkpoint, then exit 99 so the batch script requeues the job
        signal.signal(signal.SIGUSR1, lambda *_: _STOP.update(requested=True))
    # [ablation-hooks] args.seed was parsed but never used upstream
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    if args.restart_dir:
        with open(f'{args.restart_dir}/model_parameters.yml') as f:
            args_old = Namespace(**yaml.full_load(f))

        model = get_model(args_old).to(device)
        state_dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)

    else:
        model = get_model(args).to(device)

    numel = sum([p.numel() for p in model.parameters()])

    # construct loader and set device
    if args.boltzmann_training:
        boltzmann_resampler = BoltzmannResampler(args, model)
    else:
        boltzmann_resampler = None
    train_loader, val_loader = construct_loader(args, boltzmann_resampler=boltzmann_resampler)

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    # record parameters
    yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    if args.boltzmann_training:
        boltzmann_train(args, model, optimizer, train_loader, val_loader, boltzmann_resampler)
    else:
        train(args, model, optimizer, scheduler, train_loader, val_loader)
