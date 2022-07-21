import torch
from .utils import init_scheduler, init_optimizer
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.optim.lr_scheduler as sched

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from .scheduler import GradualWarmupScheduler, GradualCooldownScheduler
# from torchviz import make_dot

import argparse, os, sys, pickle
from datetime import datetime
from math import sqrt, inf, ceil, exp
import logging
logger = logging.getLogger(__name__)


# These loss functions are used ONLY for logging (the last two loss values in the log string) and choosing the best model
# The actual loss used for optimization is passed as an argument to Trainer

def Loss(predict, targets):
    return torch.nn.CrossEntropyLoss()(predict, targets.long())      # Cross Entropy Loss (positive number). The closer to 0 the better.

def AltLoss(predict, targets):
    return (predict.argmax(dim=1) == targets.long()).float().mean()  # right now this is accuracy of classification

# AUC score for logging
def AUCScore(predict, targets):
    if torch.equal(targets, torch.ones_like(targets)) or torch.equal(targets, torch.zeros_like(targets)):
        return 0
    else:
        return roc_auc_score(targets, predict[:, 1])          # Area Under Curve score (between 0 and 1). The closer to 1 the better.

def ROC(predict, targets):
    if torch.equal(targets, torch.ones_like(targets)) or torch.equal(targets, torch.zeros_like(targets)):
        return None, 0., 0.
    else:
        curve = roc_curve(targets, predict[:, 1])
        idx = np.argmin(np.abs(curve[1]-0.3))
        if curve[0][idx]>0.: 
            eB, eS = curve[0][idx], curve[1][idx]
        else:
            idx = np.where(curve[0]>0)[0]
            if len(idx)>0:
                idx = idx[0]
                eB, eS = curve[0][idx], curve[1][idx]
            else:
                eB, eS = 1., 1.
        return curve, eB, eS


class Trainer:
    """
    Class to train network. Includes checkpoints, optimizer, scheduler,
    """
    def __init__(self, args, dataloaders, model, loss_fn, metrics_fn, minibatch_metrics_fn, minibatch_metrics_string_fn, 
                 optimizer, scheduler, restart_epochs, summarize, device, dtype):
        np.set_printoptions(precision=3)
        self.args = args
        self.dataloaders = dataloaders
        self.model = model
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.minibatch_metrics_fn = minibatch_metrics_fn
        self.minibatch_metrics_string_fn = minibatch_metrics_string_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        if args.lr_decay_type == 'warm':
            warmup_epochs = int(self.args.num_epoch/9)
            cooldown_epochs = int(self.args.num_epoch/12)
            self.scheduler = GradualWarmupScheduler(optimizer, multiplier=1, warmup_epochs=4*len(dataloaders['train']), after_scheduler=scheduler)
            coodlown_start = (self.args.num_epoch - warmup_epochs - cooldown_epochs)*len(dataloaders['train'])
            cooldown_length = cooldown_epochs*len(dataloaders['train'])
            self.scheduler = GradualCooldownScheduler(optimizer, args.lr_final, coodlown_start, cooldown_length, self.scheduler)
        self.restart_epochs = restart_epochs



        self.stats = None  # dataloaders['train'].dataset.stats

        # TODO: Fix this until TB summarize is implemented.
        self.summarize = summarize
        if summarize:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=args.logdir+args.prefix)

        self.epoch = 1
        self.minibatch = 0
        self.best_epoch = 0
        self.best_metrics = {'loss': inf}

        self.device = device
        self.dtype = dtype

    def _save_checkpoint(self, valid_metrics=None):
        if not self.args.save:
            return

        save_dict = {'args': self.args,
                     'model_state': self.model.state_dict(),
                     'optimizer_state': self.optimizer.state_dict(),
                     'scheduler_state': self.scheduler.state_dict(),
                     'epoch': self.epoch,
                     'minibatch': self.minibatch,
                     'best_metrics': self.best_metrics}

        if valid_metrics is None:
            logger.info('Saving model to checkpoint file: {}'.format(self.args.checkfile))
            torch.save(save_dict, self.args.checkfile)
        elif valid_metrics['loss'] < self.best_metrics['loss']:
            self.best_epoch = self.epoch
            self.best_metrics = save_dict['best_metrics'] = valid_metrics
            logger.info('Lowest loss achieved! Saving best model to file: {}'.format(self.args.bestfile))
            torch.save(save_dict, self.args.bestfile)


    def load_checkpoint(self):
        """
        Load checkpoint from previous file.
        """
        if not self.args.load:
            return
        elif os.path.exists(self.args.checkfile):
            logger.info('Loading previous model from checkpoint!')
            self.load_state(self.args.checkfile)
            self.epoch += 1
            # self.optimizer = init_optimizer(self.args, self.model)
            # self.scheduler, self.restart_epochs = init_scheduler(self.args, self.optimizer)
        else:
            logger.info('No checkpoint included! Starting fresh training program.')
            return

    def load_state(self, checkfile):
        logger.info('Loading from checkpoint!')

        checkpoint = torch.load(checkfile, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.best_metrics = checkpoint['best_metrics']
        self.minibatch = checkpoint['minibatch']

        logger.info(f'Best metrics from checkpoint is at epoch {self.epoch}:\n{self.best_metrics}')

    def evaluate(self, splits=['train', 'valid', 'test'], best=True, final=True):
        """
        Evaluate model on training/validation/testing splits.

        :splits: List of splits to include. Only valid splits are: 'train', 'valid', 'test'
        :best: Evaluate best model as determined by minimum validation error over evolution
        :final: Evaluate final model at end of training phase
        """
        if not self.args.save:
            logger.info('No model saved! Cannot give final status.')
            return

        # Evaluate final model (at end of training)
        if final:
            logger.info('Getting predictions for model in last checkpoint.')

            # Load checkpoint model to make predictions
            checkpoint = torch.load(self.args.checkfile, map_location=torch.device(self.device))
            final_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state'])

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                predict, targets = self.predict(set=split)
                best_metrics = self.log_predict(predict, targets, split, description='Final')

        # Evaluate best model as determined by validation error
        if best:
            # Load best model to make predictions
            checkpoint = torch.load(self.args.bestfile, map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint['model_state'])
            if (not final) or (final and not checkpoint['epoch'] == final_epoch):
                logger.info(f'Getting predictions for best model (epoch {checkpoint["epoch"]}).')
                # Loop over splits, predict, and output/log predictions
                for split in splits:
                    predict, targets = self.predict(split)
                    best_metrics = self.log_predict(predict, targets, split, description='Best')
            else:
                if checkpoint['epoch'] == final_epoch:
                    logger.info('BEST MODEL IS SAME AS FINAL')

        logger.info('Inference phase complete!\n')

        return best_metrics

    def _warm_restart(self, epoch):
        restart_epochs = self.restart_epochs

        if epoch in restart_epochs:
            logger.info('Warm learning rate restart at epoch {}!'.format(epoch))
            self.scheduler.last_epoch = 1
            idx = restart_epochs.index(epoch)
            self.scheduler.T_max = restart_epochs[idx + 1] - restart_epochs[idx]
            if self.args.lr_minibatch:
                self.scheduler.T_max *= ceil(self.args.num_train / (self.args.back_batch_size if self.args.back_batch_size is not None else self.args.batch_size))
            self.scheduler.step(0)

    def _log_minibatch(self, batch_idx, loss, targets, predict, batch_t, epoch_t):
        mini_batch_loss = loss.item()
        minibatch_metrics = self.minibatch_metrics_fn(predict, targets, mini_batch_loss)

        if batch_idx == 0:
            self.minibatch_metrics = minibatch_metrics
        else:
            """ 
            Exponential average of recent Loss/AltLoss on training set for more convenient logging.
            alpha must be a positive real number. 
            alpha = 0 corresponds to no smoothing ("average over 0 previous minibatchas")
            alpha > 0 will produce smoothing where the weight of the k-to-last minibatch is proportional to exp(-gamma * k)
                    where gamma = - log(alpha/(1 + alpha)). At large alpha the weight is approx. exp(- k/alpha).
                    The number of minibatches that contribute significantly at large alpha scales like alpha.
            """
            alpha = self.args.alpha
            assert alpha >= 0, "alpha must be a nonnegative real number"
            alpha = alpha / (1 + alpha)
            for i, metric in enumerate(minibatch_metrics):
                self.minibatch_metrics[i] = alpha * self.minibatch_metrics[i] + (1 - alpha) * metric

        dtb = (datetime.now() - batch_t).total_seconds()
        tepoch = (datetime.now() - epoch_t).total_seconds()
        self.batch_time += dtb
        tcollate = tepoch - self.batch_time

        if self.args.textlog:
            logstring = self.args.prefix + ' E:{:3}/{}, B: {:5}/{}'.format(self.epoch, self.args.num_epoch, batch_idx + 1, len(self.dataloaders['train']))
            logstring += self.minibatch_metrics_string_fn(self.minibatch_metrics)
            logstring += '  dt:{:> 6.3f}{:> 8.2f}{:> 8.2f}'.format(dtb, tepoch, tcollate)
            logstring += '  {:.2E}'.format(self.scheduler.get_last_lr()[0])
            if self.args.verbose:
                logger.info(logstring)
            else:
                print(logstring)


    def _step_lr_batch(self):
        if self.args.lr_minibatch:
            self.scheduler.step()

    def _step_lr_epoch(self):
        if not self.args.lr_minibatch:
            self.scheduler.step()

    def train(self, trial=None, metric_to_report='loss'):
        epoch0 = self.epoch
        for epoch in range(epoch0, self.args.num_epoch + 1):
            self.epoch = epoch
            epoch_time = datetime.now()
            logger.info('STARTING Epoch {}'.format(epoch))

            self._warm_restart(epoch)
            self._step_lr_epoch()

            train_predict, train_targets, epoch_t = self.train_epoch()
            train_metrics= self.log_predict(train_predict, train_targets, 'train', epoch=epoch, epoch_t=epoch_t)

            self._save_checkpoint()

            valid_predict, valid_targets = self.predict(set='valid')
            valid_metrics = self.log_predict(valid_predict, valid_targets, 'valid', epoch=epoch)

            self._save_checkpoint(valid_metrics)
            
            if trial:
                trial.set_user_attr("best_epoch", self.best_epoch)
                trial.set_user_attr("best_metrics", self.best_metrics)
                trial.report(min(valid_metrics[metric_to_report], 1), epoch - 1)
                if trial.should_prune():
                    import optuna
                    raise optuna.exceptions.TrialPruned()

            logger.info('FINISHED Epoch {}\n_________________________\n'.format(epoch))
            
        if self.summarize: self.writer.close()

        return self.best_epoch, self.best_metrics

    def _get_target(self, data, stats=None):
        """
        Get the learning target.
        If a stats dictionary is included, return a normalized learning target.
        """
        target_type = torch.long if self.args.target=='is_signal' else self.dtype
        targets = data[self.args.target].to(self.device, target_type)

        # if stats is not None:
        #     mu, sigma = stats[self.args.target]
        #     targets = (targets - mu) / sigma

        # print("TARGETS:", targets)
        return targets

    def train_epoch(self):
        dataloader = self.dataloaders['train']

        current_idx, num_data_pts = 0, len(dataloader.dataset)
        self.loss_val, self.alt_loss_val, self.batch_time = 0, 0, 0
        all_predict, all_targets = [], []

        self.model.train()
        epoch_t = datetime.now()

        total_loss = 0

        for batch_idx, data in enumerate(dataloader):
            batch_t = datetime.now()

            # Get targets and predictions
            targets = self._get_target(data, self.stats)
            predict = self.model(data)

            # predict_t = datetime.now()

            # Calculate loss and backprop
            loss = self.loss_fn(predict, targets)
            total_loss += loss
            if (batch_idx % self.args.batch_group_size == 0) or batch_idx + 1 == len(dataloader):
                ave_loss = total_loss / (self.args.batch_group_size if batch_idx % self.args.batch_group_size == 0 else len(dataloader) % self.args.batch_group_size)
                
                # Standard zero-gradient and backward propagation
                self.optimizer.zero_grad()
                total_loss.backward()

                total_loss = 0

                if not self.args.quiet and not all(param.grad is not None for param in dict(self.model.named_parameters()).values()):
                    print("WARNING: The following params have missing gradients at backward pass (they are probably not being used in output):\n", {key: '' for key, param in self.model.named_parameters() if param.grad is None})

                # Step optimizer and learning rate
                self.optimizer.step()
            self._step_lr_batch()

            targets, predict = targets.detach().cpu(), predict.detach().cpu()
            all_predict.append(predict)
            all_targets.append(targets)

            # print((predict_t-batch_t).total_seconds(), (datetime.now()-predict_t).total_seconds())

            self._log_minibatch(batch_idx, loss, targets, predict, batch_t, epoch_t)

            self.minibatch += 1

        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        return all_predict, all_targets, epoch_t

    def predict(self, set='valid'):
        dataloader = self.dataloaders[set]

        self.model.eval()
        all_predict, all_targets = [], []
        start_time = datetime.now()
        logger.info('Starting testing on {} set: '.format(set))

        for batch_idx, data in enumerate(dataloader):

            targets = self._get_target(data, self.stats)
            predict = self.model(data).detach()

            all_targets.append(targets)
            all_predict.append(predict)

        all_predict = torch.cat(all_predict)
        all_targets = torch.cat(all_targets)

        dt = (datetime.now() - start_time).total_seconds()
        logger.info('Total evaluation time: {}s'.format(dt))

        return all_predict, all_targets

    def log_predict(self, predict, targets, dataset, epoch=-1, epoch_t=None, description=''):
        predict = predict.cpu().double()
        targets = targets.cpu().double()

        datastrings = {'train': 'Training  ', 'test': 'Testing   ', 'valid': 'Validation'}

        if epoch >= 0:
            suffix = 'final'
        else:
            suffix = 'best'

        prefix = self.args.predictfile + '.' + 'final' + '.' + dataset
        metrics, logstring = self.metrics_fn(predict, targets, prefix, logger)
        if epoch >= 0:
            logger.info(f'Epoch {epoch} {description} {datastrings[dataset]}'+logstring)
        else:
            logger.info(f'{description} {datastrings[dataset]}'+logstring)

        if epoch_t:
            logger.info(f'Total epoch time:      {(datetime.now() - epoch_t).total_seconds()}s')

        metricsfile = self.args.predictfile + '.metrics.' + dataset + '.csv'   
        
        if epoch >= 0:
            with open(metricsfile, mode='a' if (self.args.load or epoch>1) else 'w') as file_:
                if epoch == 1:
                    file_.write(",".join(metrics.keys()))
                file_.write(",".join(map(str, metrics.values())))
                file_.write("\n")  # Next line.


        if self.args.predict:
            file = self.args.predictfile + '.' + suffix + '.' + dataset + '.pt'
            logger.info('Saving predictions to file: {}'.format(file))
            torch.save({'predict': predict, 'targets': targets}, file)

        if self.summarize:
            if description == 'Best': dataset = 'Best_' + dataset
            if description == 'Final': dataset = 'Final_' + dataset
            for name, metric in metrics.items():
                self.writer.add_scalar(dataset+'/'+name, metric, epoch)

        return metrics
