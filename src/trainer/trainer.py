import torch
from .utils import init_scheduler, init_optimizer, _max_norm
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import torch.optim.lr_scheduler as sched

import numpy as np
from .scheduler import GradualWarmupScheduler, GradualCooldownScheduler
# from torchviz import make_dot

import argparse, os, sys, pickle
from datetime import datetime
from math import sqrt, inf, ceil, exp
import logging
logger = logging.getLogger(__name__)

class Trainer:
    """
    Class to train network. Includes checkpoints, optimizer, scheduler,
    """
    def __init__(self, args, dataloaders, model, loss_fn, metrics_fn, minibatch_metrics_fn, minibatch_metrics_string_fn, 
                 optimizer, scheduler, restart_epochs, summarize_csv, summarize, device, dtype):
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
        warmup_epochs = 4 #int(self.args.num_epoch/8)
        if args.num_epoch > warmup_epochs:
            if warmup_epochs > 0:
                self.scheduler = GradualWarmupScheduler(optimizer, multiplier=1, warmup_epochs=len(dataloaders['train'])*warmup_epochs, after_scheduler=scheduler)
            if args.lr_decay_type == 'warm':
                cooldown_epochs = int(self.args.num_epoch/11)
                coodlown_start = (self.args.num_epoch - warmup_epochs - cooldown_epochs)*len(dataloaders['train'])
                cooldown_length = cooldown_epochs*len(dataloaders['train'])
                self.scheduler = GradualCooldownScheduler(optimizer, args.lr_final, coodlown_start, cooldown_length, self.scheduler)
            elif args.lr_decay_type == 'flat':
                cooldown_epochs = int(self.args.num_epoch/3)
                coodlown_start = (self.args.num_epoch - cooldown_epochs)*len(dataloaders['train'])
                cooldown_length = cooldown_epochs*len(dataloaders['train'])
                self.scheduler = GradualCooldownScheduler(optimizer, args.lr_final, coodlown_start, cooldown_length, self.scheduler)
            elif args.lr_decay_type == 'cos':
                cooldown_epochs = 3
                coodlown_start = (self.args.num_epoch - warmup_epochs - cooldown_epochs)*len(dataloaders['train'])
                cooldown_length = cooldown_epochs*len(dataloaders['train'])
                self.scheduler = GradualCooldownScheduler(optimizer, args.lr_final, coodlown_start, cooldown_length, self.scheduler)
        self.restart_epochs = restart_epochs



        self.stats = None  # dataloaders['train'].dataset.stats

        # TODO: Fix this until TB summarize is implemented.
        self.summarize_csv = summarize_csv
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
        elif valid_metrics['loss'] < self.best_metrics['loss']: # and self.epoch/self.args.num_epoch >= 0.5:
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
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.best_metrics = checkpoint['best_metrics']
        self.minibatch = checkpoint['minibatch']
        del checkpoint

        logger.info(f'Loaded checkpoint at epoch {self.epoch}.\nBest metrics from checkpoint are at epoch {self.epoch}:\n{self.best_metrics}')

    def evaluate(self, splits=['train', 'valid', 'test'], best=True, final=True, ir_data=None, c_data=None, expand_data=None):
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
            # Load checkpoint model to make predictions
            checkpoint = torch.load(self.args.checkfile, map_location=torch.device(self.device))
            final_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state'])
            logger.info(f'Getting predictions for final model {self.args.checkfile} (epoch {final_epoch}).')

            # Loop over splits, predict, and output/log predictions
            for split in splits:
                predict, targets = self.predict(set=split, ir_data=ir_data, c_data=c_data, expand_data=expand_data)
                best_metrics, logstring = self.log_predict(predict, targets, split, description='Final')

        # Evaluate best model as determined by validation error
        if best:
            # Load best model to make predictions
            checkpoint = torch.load(self.args.bestfile, map_location=torch.device(self.device))
            best_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state'])
            if (not final) or (final and not best_epoch == final_epoch):
                logger.info(f'Getting predictions for best model {self.args.bestfile} (epoch {best_epoch}, best validation metrics were {checkpoint["best_metrics"]}).')
                # Loop over splits, predict, and output/log predictions
                for split in splits:
                    predict, targets = self.predict(split, ir_data=ir_data, c_data=c_data, expand_data=expand_data)
                    best_metrics, logstring = self.log_predict(predict, targets, split, description='Best')
            elif best_epoch == final_epoch:
                logger.info('BEST MODEL IS SAME AS FINAL')
                self.log_predict(predict, targets, split, description='Best', repeat=[best_metrics, logstring])
        logger.info('Inference phase complete!\n')

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

    def _log_minibatch(self, batch_idx, loss, targets, predict, batch_t, fwd_t, bwd_t, epoch_t):
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

        dt_batch = (datetime.now() - batch_t).total_seconds()
        dt_fwd = (fwd_t - batch_t).total_seconds()
        dt_bwd = (bwd_t - fwd_t).total_seconds()
        tepoch = (datetime.now() - epoch_t).total_seconds()
        self.batch_time += dt_batch
        tcollate = tepoch - self.batch_time

        if self.args.textlog:
            logstring = self.args.prefix + ' E:{:3}/{}, B: {:5}/{}'.format(self.epoch, self.args.num_epoch, batch_idx + 1, len(self.dataloaders['train']))
            logstring += self.minibatch_metrics_string_fn(self.minibatch_metrics)
            logstring += '  dt:({:> 4.3f}+{:> 4.3f}={:> 4.3f}){:> 8.2f}{:> 8.2f}'.format(dt_fwd, dt_bwd, dt_batch, tepoch, tcollate)
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
            train_metrics,_ = self.log_predict(train_predict, train_targets, 'train', epoch=epoch, epoch_t=epoch_t)

            self._save_checkpoint()

            valid_predict, valid_targets = self.predict(set='valid')
            valid_metrics, _ = self.log_predict(valid_predict, valid_targets, 'valid', epoch=epoch)

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

        return targets

    def train_epoch(self):
        dataloader = self.dataloaders['train']

        current_idx, num_data_pts = 0, len(dataloader.dataset)
        self.loss_val, self.alt_loss_val, self.batch_time = 0, 0, 0
        all_predict, all_targets = {}, []

        self.model.train()
        epoch_t = datetime.now()

        total_loss = 0

        for batch_idx, data in enumerate(dataloader):
            batch_t = datetime.now()

            # Get targets and predictions
            targets = self._get_target(data, self.stats)
            predict = self.model(data)
            fwd_t = datetime.now()
            # predict_t = datetime.now()

            # Calculate loss and backprop
            loss = self.loss_fn(predict['predict'], targets)
            total_loss += loss
                
            # Standard zero-gradient and backward propagation
            self.optimizer.zero_grad()
            total_loss.backward()
            bwd_t = datetime.now()
            total_loss = 0
            if not self.args.quiet and not all(param.grad is not None for param in dict(self.model.named_parameters()).values()):
                print("WARNING: The following params have missing gradients at backward pass (they are probably not being used in output):\n", {key: '' for key, param in self.model.named_parameters() if param.grad is None})

            # Step optimizer and learning rate
            self.optimizer.step()
            # self.model.apply(_max_norm)

            self._step_lr_batch()


            targets = targets.detach().cpu()
            predict = {key: val.detach().cpu() for key, val in predict.items()}

            all_targets.append(targets)
            for key, val in predict.items(): all_predict.setdefault(key,[]).append(val)

            self._log_minibatch(batch_idx, loss, targets, predict['predict'], batch_t, fwd_t, bwd_t, epoch_t)

            self.minibatch += 1

        all_predict = {key: torch.cat(val) for key, val in all_predict.items()}
        all_targets = torch.cat(all_targets)

        return all_predict, all_targets, epoch_t

    def predict(self, set='valid', ir_data=None, c_data=None, expand_data=None):
        dataloader = self.dataloaders[set]

        self.model.eval()
        all_predict, all_targets = {}, []
        start_time = datetime.now()
        logger.info('Starting testing on {} set: '.format(set))

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                if expand_data:
                    data = expand_data(data)
                if ir_data is not None:
                    data = ir_data(data)
                if c_data is not None:
                    data = c_data(data)
                targets = self._get_target(data, self.stats)
                predict = {key: val for key, val in self.model(data).items()}

                all_targets.append(targets)
                for key, val in predict.items(): all_predict.setdefault(key, []).append(val)

        all_targets = torch.cat(all_targets)
        all_predict = {key: torch.cat(val) for key, val in all_predict.items()}

        dt = (datetime.now() - start_time).total_seconds()
        logger.info('Total evaluation time: {}s'.format(dt))

        return all_predict, all_targets

    def log_predict(self, predict, targets, dataset, epoch=-1, epoch_t=None, description='', repeat=None):
        predict = {key: val.cpu().double() for key, val in predict.items()}
        targets = targets.cpu().double()

        datastrings = {'train': 'Training  ', 'test': 'Testing   ', 'valid': 'Validation'}

        if epoch >= 0:
            suffix = 'final'
        else:
            suffix = 'best'

        prefix = self.args.predictfile + '.' + suffix + '.' + dataset
        metrics, logstring = None, 'metrics skipped!'

        if repeat is None:
            metrics, logstring = self.metrics_fn(predict['predict'], targets, self.loss_fn, prefix, logger)
        else:
            metrics, logstring = repeat[0], repeat[1]

        if epoch >= 0:
            logger.info(f'Epoch {epoch} {description} {datastrings[dataset]}'+logstring)
        else:
            logger.info(f'{description:<5} {datastrings[dataset]:<10}'+logstring)

        if epoch_t:
            logger.info(f'Total epoch time:      {(datetime.now() - epoch_t).total_seconds()}s')

        metricsfile = self.args.predictfile + '.metrics.' + dataset + '.csv'   
        testmetricsfile = os.path.join(self.args.workdir, self.args.logdir, self.args.prefix.split("-")[0]+'.'+description+'.metrics.csv')

        if epoch >= 0 and self.summarize_csv=='all':
            with open(metricsfile, mode='a' if (self.args.load or epoch>1) else 'w') as file_:
                if epoch == 1:
                    file_.write(",".join(metrics.keys()))
                file_.write(",".join(map(str, metrics.values())))
                file_.write("\n")  # Next line.
        if epoch < 0 and self.summarize_csv in ['test','all']:
            if not os.path.exists(testmetricsfile):
                with open(testmetricsfile, mode='a') as file_:
                    file_.write("prefix,timestamp,"+",".join(metrics.keys()))
                    file_.write("\n")  # Next line.
            with open(testmetricsfile, mode='a') as file_:
                file_.write(self.args.prefix+','+str(datetime.now())+','+",".join(map(str, metrics.values())))
                file_.write("\n")  # Next line.


        if self.args.predict and (repeat is None):
            file = self.args.predictfile + '.' + suffix + '.' + dataset + '.pt'
            logger.info('Saving predictions to file: {}'.format(file))
            predict.update({'targets': targets})
            torch.save(predict, file)
        
        if repeat is not None:
            logger.info('Predictions already saved above')

        if self.summarize:
            if description == 'Best': dataset = 'Best_' + dataset
            if description == 'Final': dataset = 'Final_' + dataset
            for name, metric in metrics.items():
                if not isinstance(metric, np.ndarray):
                    self.writer.add_scalar(dataset+'/'+name, metric, epoch)
                else:
                    if metric.size==1:
                        self.writer.add_scalar(dataset+'/'+name, metric, epoch)
                    else:
                        for i, m in enumerate(metric):
                            self.writer.add_scalar(dataset+'/'+name+'_'+str(i), m, epoch)

        return metrics, logstring
