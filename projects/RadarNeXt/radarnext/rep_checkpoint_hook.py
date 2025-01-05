import copy
import logging
import os.path as osp
from math import inf
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from collections import deque
from mmengine.registry import HOOKS
from mmengine.hooks.hook import Hook
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from mmengine.utils import is_list_of, is_seq_of
from mmengine.fileio import FileClient, get_file_backend 

from .mobileone_blocks import reparameterize_model

@HOOKS.register_module()
class Rep_Checkpoint_Hook(Hook):
    '''Save the weights after val_loop when the training at a specific epoch'''\
    
    priority='VERY_LOW'

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']
    
    def __init__(self, 
                 interval: int = -1,
                 reparams: bool = True,  # Whether to re-parameterized the model for inference and saving the re-parameterized weights
                 by_epoch: bool = True,
                 save_optimizer: bool = False,
                 save_param_scheduler: bool = False,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 rule: Union[str, List[str], None] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 file_client_args: Optional[dict] = None,
                 filename_tmpl: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 published_keys: Union[str, List[str], None] = None,
                 save_begin: int = 0,
                 **kwargs):
        super(Rep_Checkpoint_Hook, self).__init__()
        self.reparams = reparams
        self.save_reparams = reparams
        
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_param_scheduler = save_param_scheduler
        self.out_dir = out_dir  # type: ignore
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs

        if file_client_args is not None:
            print_log(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead',
                logger='current',
                level=logging.WARNING)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

        self.file_client_args = file_client_args
        self.backend_args = backend_args

        if filename_tmpl is None:
            if self.by_epoch:
                self.filename_tmpl = 'epoch_{}.pth'
            else:
                self.filename_tmpl = 'iter_{}.pth'
        else:
            self.filename_tmpl = filename_tmpl

        # save best logic
        assert (isinstance(save_best, str) or is_list_of(save_best, str)
                or (save_best is None)), (
                    '"save_best" should be a str or list of str or None, '
                    f'but got {type(save_best)}')

        if isinstance(save_best, list):
            if 'auto' in save_best:
                assert len(save_best) == 1, (
                    'Only support one "auto" in "save_best" list.')
            assert len(save_best) == len(
                set(save_best)), ('Find duplicate element in "save_best".')
        else:
            # convert str to list[str]
            if save_best is not None:
                save_best = [save_best]  # type: ignore # noqa: F401
        self.save_best = save_best

        # rule logic
        assert (isinstance(rule, str) or is_list_of(rule, str)
                or (rule is None)), (
                    '"rule" should be a str or list of str or None, '
                    f'but got {type(rule)}')
        if isinstance(rule, list):
            # check the length of rule list
            assert len(rule) in [
                1,
                len(self.save_best)  # type: ignore
            ], ('Number of "rule" must be 1 or the same as number of '
                f'"save_best", but got {len(rule)}.')
        else:
            # convert str/None to list
            rule = [rule]  # type: ignore # noqa: F401

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys, )  # type: ignore
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys  # type: ignore

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys, )  # type: ignore
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys  # type: ignore

        if self.save_best is not None:
            self.is_better_than: Dict[str, Callable] = dict()
            self._init_rule(rule, self.save_best)
            if len(self.key_indicators) == 1:
                self.best_ckpt_path: Optional[str] = None
            else:
                self.best_ckpt_path_dict: Dict = dict()

        # published keys
        if not (isinstance(published_keys, str)
                or is_seq_of(published_keys, str) or published_keys is None):
            raise TypeError(
                '"published_keys" should be a str or a sequence of str or '
                f'None, but got {type(published_keys)}')

        if isinstance(published_keys, str):
            published_keys = [published_keys]
        elif isinstance(published_keys, (list, tuple)):
            assert len(published_keys) == len(set(published_keys)), (
                'Find duplicate elements in "published_keys".')
        self.published_keys = published_keys

        self.last_ckpt = None
        if save_begin < 0:
            raise ValueError(
                'save_begin should not be less than 0, but got {save_begin}')
        self.save_begin = save_begin

    def before_test(self, runner, **kwargs):
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        # if arguing to save the re-parameterized weights into 'reparams_' + '{loading_filename}.pth'
        if 'reparams_' in self.filename_tmpl and self.reparams:
            if self.out_dir is None:
                self.out_dir = runner.work_dir

            # If self.file_client_args is None, self.file_client will not
            # used in CheckpointHook. To avoid breaking backward compatibility,
            # it will not be removed util the release of MMEngine1.0
            self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

            if self.file_client_args is None:
                self.file_backend = get_file_backend(
                    self.out_dir, backend_args=self.backend_args)
            else:
                self.file_backend = self.file_client

            # if `self.out_dir` is not equal to `runner.work_dir`, it means that
            # `self.out_dir` is set so the final `self.out_dir` is the
            # concatenation of `self.out_dir` and the last level directory of
            # `runner.work_dir`
            if self.out_dir != runner.work_dir:
                basename = osp.basename(runner.work_dir.rstrip(osp.sep))
                self.out_dir = self.file_backend.join_path(
                    self.out_dir, basename)  # type: ignore  # noqa: E501

            runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

            if self.save_best is not None:
                if len(self.key_indicators) == 1:
                    if 'best_ckpt' not in runner.message_hub.runtime_info:
                        self.best_ckpt_path = None
                    else:
                        self.best_ckpt_path = runner.message_hub.get_info(
                            'best_ckpt')
                else:
                    for key_indicator in self.key_indicators:
                        best_ckpt_name = f'best_ckpt_{key_indicator}'
                        if best_ckpt_name not in runner.message_hub.runtime_info:
                            self.best_ckpt_path_dict[key_indicator] = None
                        else:
                            self.best_ckpt_path_dict[
                                key_indicator] = runner.message_hub.get_info(
                                    best_ckpt_name)

            if self.max_keep_ckpts > 0:
                keep_ckpt_ids = []
                if 'keep_ckpt_ids' in runner.message_hub.runtime_info:
                    keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids')

                    while len(keep_ckpt_ids) > self.max_keep_ckpts:
                        step = keep_ckpt_ids.pop(0)
                        if is_main_process():
                            path = self.file_backend.join_path(
                                self.out_dir, self.filename_tmpl.format(step))
                            if self.file_backend.isfile(path):
                                self.file_backend.remove(path)
                            elif self.file_backend.isdir(path):
                                # checkpoints saved by deepspeed are directories
                                self.file_backend.rmtree(path)

                self.keep_ckpt_ids: deque = deque(keep_ckpt_ids,
                                                  self.max_keep_ckpts)

            # re-parameterize the model into the graph for validations
            runner.logger.info('Re-parameterize the multi-branch model into the single-path graph.')
            runner.model = copy.deepcopy(reparameterize_model(runner.model))
            
            # save the re-parameterized model graphs and weights
            if self.save_reparams:
                self._save_rep_model(runner)
        else:
            pass  # if not requiring to save re-parameterized weights (straightly define the single-path model for inferences)

    def _save_rep_model(self, runner):
        # the filename that the weights read from
        pth_filename = self.filename_tmpl[9:]
        
        # save checkpoint
        runner.logger.info(f'Saving re-parameterized checkpoints into {self.filename_tmpl} derived from {pth_filename}')
        self._save_checkpoint(runner)
    
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            step = runner.epoch + 1
            meta = dict(epoch=step, iter=runner.iter)
        else:
            step = runner.iter + 1
            meta = dict(epoch=runner.epoch, iter=step)

        self._save_checkpoint_with_step(runner, step, meta=meta)
    
    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step))

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info('keep_ckpt_ids',
                                               list(self.keep_ckpt_ids))

        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(self.out_dir,
                                                     ckpt_filename)
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args)

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return
        
