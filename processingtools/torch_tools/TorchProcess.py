import torch
import torch.utils.data
import processingtools.ProgressBar
import processingtools.functions
import torchvision
import cv2
import typing
import warnings
import os


class Trainer(torch.nn.Module):
    """
    Basic trainer.
    """

    def __init__(self, model, train_loader, test_loader, optimizer, criterion, epoch, save_path,
                recoder=None, validation_loader=None, scheduler=None, save_interval: int = 5, valid_interval: int = 5,
                start_epoch: int = 0, best_acc: int = 0, compile=True):
        super().__init__()

        self.model = torch.compile(model) if compile else model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader if validation_loader is not None else test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.epoch = epoch
        self.save_path = save_path
        self.save_interval = save_interval
        self.valid_interval = valid_interval
        self.start_epoch = start_epoch
        self.best_acc = best_acc
        self.recoder = recoder

        self.valid_evaluator = Evaluator(model, self.validation_loader, save_path, recoder)
        self.test_evaluator = Evaluator(model, test_loader, save_path, recoder)

    def forward(self):
        self.model.train()

        best_acc = 0
        for epoch in range(self.start_epoch, self.epoch):
            loss = None
            loss_sum = 0

            for inputs, targets in processingtools.ProgressBar(self.train_loader, total=len(self.train_loader), detail_func=lambda _: f'loss: {loss:0.3f}' if loss is not None else '...', finish_mark=None):
                inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                loss_sum = loss_sum + float(loss)

            print('\r', end='')
            print_recoder(self.recoder, f'[{epoch + 1}/{self.epoch}] loss: {loss_sum / len(self.train_loader): 0.3f}')

            if self.scheduler is not None:
                self.scheduler.step()
                if (epoch + 1) % self.save_interval == 0:
                    torch.save({'epoch': epoch, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'best_acc': best_acc},
                               f'{self.save_path}/model_{processingtools.functions.zero_padding(self.epoch, epoch + 1)}.pth.tar')

                if (epoch + 1) % self.valid_interval == 0:
                    present_acc = self.valid_evaluator()
                    if best_acc < present_acc:
                        best_acc = present_acc
                        torch.save(
                            {'epoch': epoch, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                             'scheduler': self.scheduler.state_dict(), 'best_acc': best_acc},
                            f'{self.save_path}/model_best.pth.tar')
                    self.recoder.put_space() if self.recoder is not None else print()

            else:
                if (epoch + 1) % self.save_interval == 0:
                    torch.save({'epoch': epoch, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'best_acc': best_acc},
                               f'{self.save_path}/model_{processingtools.functions.zero_padding(self.epoch, epoch + 1)}.pth.tar')

                if (epoch + 1) % self.valid_interval == 0:
                    present_acc = self.valid_evaluator()
                    if best_acc < present_acc:
                        best_acc = present_acc
                        torch.save(
                            {'epoch': epoch, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                             'best_acc': best_acc},
                            f'{self.save_path}/model_best.pth.tar')
                    self.recoder.put_space() if self.recoder is not None else print()

        self.test_evaluator()

        return self.model


class Evaluator(torch.nn.Module):
    """
    Basic evaluator.
    """

    def __init__(self, model, test_loader, save_path, recoder=None):
        super().__init__()

        self.model = model
        self.test_loader = test_loader
        self.save_path = save_path
        self.recoder = recoder

    @torch.no_grad()
    def forward(self):
        self.model.eval()

        correct = 0
        data_size = 0

        for inputs, targets in processingtools.ProgressBar(self.test_loader, total=len(self.test_loader), finish_mark=None, detail_func=lambda _: f'acc: {correct / data_size: 0.3f} %' if data_size != 0 else 'acc: 0 %'):
            inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            data_size = data_size + targets.shape[0]

            outputs = self.model(inputs)
            correct = correct + int(torch.sum(torch.argmax(outputs, dim=1) == targets))

        print('\r', end='')
        print_recoder(self.recoder, f'accuracy: {(correct / data_size) * 100: 0.2f} %')

        self.model.train()
        return correct


class DDPTrainer:
    """
    Basic trainer for distributed data parallel (DDP) training.
    """

    def __init__(self, model, train_loader, test_loader, optimizer, train_func, epoch, save_path,
                 recoder=None, validation_loader=None, valid_func=None, scheduler=None, save_interval: int = 5,
                 valid_interval: int = 1, start_epoch: int = 0, model_compile: bool = True, best_metrics: float = 0):
        super().__init__()

        self.model = torch.compile(model) if model_compile else model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader if validation_loader is not None else test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_func = train_func

        self.epoch = epoch
        self.save_path = save_path
        self.save_interval = save_interval
        self.valid_interval = valid_interval
        self.start_epoch = start_epoch
        self.recoder = recoder

        self.valid_func = valid_func
        self.best_metrics = best_metrics

        self.check_shuffle_conflict()

    @staticmethod
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    @staticmethod
    def cleanup():
        torch.distributed.destroy_process_group()

    def check_shuffle_conflict(self):
        """Raise warning or error if train_loader has shuffle=True while using DistributedSampler."""
        if hasattr(self.train_loader, 'shuffle') and self.train_loader.shuffle:
            raise ValueError("train_loader has shuffle=True. You must disable shuffle when using DistributedSampler.")
        elif getattr(self.train_loader, 'sampler', None) is None and getattr(self.train_loader, 'shuffle', False):
            warnings.warn(
                "train_loader has shuffle=True but no sampler is set. "
                "This may cause issues with DistributedSampler. "
                "Consider using DistributedSampler with shuffle=False."
            )

    def train(self, rank: int, world_size: int):
        self.setup(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        self.model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )

        # Train DataLoader with DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            sampler=train_sampler,
            num_workers=self.train_loader.num_workers,
            pin_memory=True,
            shuffle=False  # Must be False when sampler is used
        )

        # Validation DataLoader and sampler
        valid_loader = None
        valid_sampler = None
        if self.valid_func:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                self.validation_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            valid_loader = torch.utils.data.DataLoader(
                self.validation_loader.dataset,
                batch_size=self.validation_loader.batch_size,
                sampler=valid_sampler,
                num_workers=self.validation_loader.num_workers,
                pin_memory=True,
                shuffle=False
            )

        for epoch in range(self.start_epoch, self.epoch):
            model.train()
            train_sampler.set_epoch(epoch)
            if valid_sampler:
                valid_sampler.set_epoch(epoch)

            loss = self.train_func(model, train_loader, self.optimizer, epoch, device, self.recoder)

            if self.scheduler:
                self.scheduler.step()

            # Validation and checkpointing
            if (epoch + 1) % self.valid_interval == 0 and self.valid_func:
                model.eval()
                with torch.no_grad():
                    local_metric = self.valid_func(model, valid_loader, epoch, device, self.recoder)

                metric_tensor = torch.tensor(local_metric, device=device)
                torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
                metrics = metric_tensor.float().item() / world_size

                if rank == 0:
                    if metrics > self.best_metrics:
                        self.best_metrics = metrics
                        self.save_checkpoint(os.path.join(self.save_path, 'best.pt'), model)

                    if (epoch + 1) % self.save_interval == 0:
                        self.save_checkpoint(os.path.join(self.save_path, f'epoch_{epoch + 1}.pt'), model)

        self.cleanup()
        return model.module  # Return the original model (unwrapped from DDP)


    @staticmethod
    def save_checkpoint(save_path: str, model):
        """
        Save the model's state dictionary to the specified path.

        This method saves only the model weights (state_dict of the underlying model
        wrapped by DistributedDataParallel). If you need to save additional information
        such as optimizer state, scheduler, epoch, or metrics, you can override this method:

            def extended_save_checkpoint(save_path, model, optimizer, epoch):
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, save_path)

            trainer = DDPTrainer(...)
            trainer.save_checkpoint = staticmethod(extended_save_checkpoint)

        :param save_path: The path to save the model checkpoint file.
        :param model: The DDP-wrapped model (i.e., torch.nn.parallel.DistributedDataParallel).
        """
        checkpoint = {
            'model_state_dict': model.module.state_dict()
        }

        torch.save(checkpoint, save_path)


class BundleLoss(torch.nn.Module):
    """
    Bundle input criterions
    """

    def __init__(self, criterions: list, reduction: str = 'Sum', weights=None):
        super().__init__()
        self.criterions = criterions
        self.reduction = reduction
        self.weights = [1 for _ in self.criterions] if weights is None else weights

    def forward(self, predictions, targets):
        loss_result = 0 if self.reduction == 'Sum' else []

        for criterion, weight in zip(self.criterions, self.weights):
            loss = criterion(predictions, targets) * weight
            if self.reduction == 'Sum':
                loss_result = loss_result + loss
            elif self.reduction == 'None':
                loss_result.append(loss)

        return loss_result


def print_recoder(recoder, string: str):
    recoder.print(string) if recoder is not None else print(string)


class AutoInputModel(torch.nn.Module):
    """
    A PyTorch module for automatically processing and normalizing input images.

    This class wraps a given model and provides functionality to read, preprocess, and forward images through the model.
    It supports custom transformers and normalization parameters.
    """

    @processingtools.functions.custom_warning_format
    def __init__(self, model, size: typing.Union[tuple, list, None] = None, mean: typing.Union[float, list, torch.Tensor, None] = None, std: typing.Union[float, list, torch.Tensor, None] = None, transformer=None):
        """
        initialize the AutoInputModel
        :param model: model to be used
        :param size: size to which images will be resized
        :param mean: mean for normalization
        :param std: standard deviation for normalization
        :param transformer: custom transformer for image preprocessing
        """

        if transformer is None and (mean is None or std is None):
            raise ValueError('Either transformer must be provided, or both mean and std must be specified.')

        super().__init__()

        self.model = model

        if transformer is not None and (mean is not None or std is not None):
            warnings.warn('NormalizeModel uses transformer for normalizing not (mean, std).')

        if transformer is not None:
            self.transform = transformer
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])

        self.device = None

    def image_read(self, path: str) -> torch.Tensor:
        """
        read and preprocess an image from the given path
        :param path: image file path
        :return: normalized image tensor
        """

        try:
            image = cv2.cvtColor(processingtools.functions.imread(path), cv2.COLOR_BGR2RGB)
            return torch.unsqueeze(self.transform(image), dim=0)
        except Exception as e:
            raise ValueError(f'Error reading image from path {path}: {e}')

    def get_device(self, logging: bool) -> None:
        """
        get the device to run the model on
        :param logging: whether to log the device information
        :return: device
        """

        self.device = next(self.model.parameters()).device
        if logging:
            print(f'run on {processingtools.functions.s_text(f"{self.device}", styles=("bold",))}')

    @processingtools.functions.custom_warning_format
    @torch.no_grad()
    def forward(self, inputs: typing.Union[str, list], batch_size: int = 1, num_workers: int = 0, logging: bool = True) -> typing.Union[torch.Tensor, dict]:
        """
        forward pass of the model.
        :param inputs: string or list of strings representing image paths.
        :param batch_size: batch size if input is a path list.
        :param num_workers: number of workers for DataLoader if input is a path list.
        :param logging: whether to log progress.
        :return: model outputs.
        """

        if self.model.training:
            warnings.warn('Model is in training mode! (If you want to change to eval mode, use model.eval())')

        self.get_device(logging)

        if isinstance(inputs, list):
            return self._process_batch(inputs, batch_size, num_workers, logging)

        if isinstance(inputs, str):
            return self.model(self.image_read(inputs))

        raise TypeError('Inputs must be a string or a list of strings')

    def _process_batch(self, inputs: list, batch_size: int, num_workers: int, logging: bool = True) -> typing.Dict[str, typing.Union[typing.Dict, typing.Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        process batch inputs.
        :param inputs: list of image paths.
        :param batch_size: batch size for DataLoader.
        :param num_workers: number of workers for DataLoader.
        :param logging: whether to log progress.
        :return: a dictionary of model outputs.
        """

        outputs = []
        out_dict: typing.Dict[str, typing.Union[typing.Dict, typing.Tuple[torch.Tensor, ...], torch.Tensor]] = {'results': {}}

        # Initialize dataset and dataloader
        dataset = AutoInputDataset(inputs, transformer=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        interator = processingtools.ProgressBar(dataloader, total=len(dataloader), finish_mark=None) if logging else dataloader
        for data in interator:
            image, paths = data
            image = image.to(self.device)
            output = self.model(image)
            outputs.append(output)

            self._store_results(out_dict, output, paths)

        if logging:
            print('\rInference done.')

        # Combine outputs into 'total outputs'
        if isinstance(outputs[0], (tuple, list)):
            out_dict['total outputs'] = self._combine_tuple_outputs(outputs)
        else:
            out_dict['total outputs'] = torch.cat(outputs, dim=0)

        return out_dict

    @staticmethod
    def _store_results(out_dict: dict, output: typing.Union[tuple, list, torch.Tensor], paths: list) -> None:
        """
        store model outputs in the results dictionary.
        :param out_dict: dictionary to store results.
        :param output: model output for the current batch.
        :param paths: corresponding paths for the batch.
        """

        if isinstance(output, (tuple, list)):
            for out, path in zip(zip(*output), paths):
                out_dict['results'][path] = out
        else:
            for out, path in zip(output, paths):
                out_dict['results'][path] = out

    @staticmethod
    def _combine_tuple_outputs(outputs: list) -> typing.Tuple[torch.Tensor, ...]:
        """
        combine outputs when the model returns tuples or lists.
        :param outputs: list of model outputs.
        :return: combined tuple of outputs.
        """

        total_output = []
        for _ in zip(*outputs):
            total_output.append(torch.cat(_, dim=0))
        return tuple(total_output)


class AutoInputDataset(torch.utils.data.Dataset):
    def __init__(self, image_list: list, size: typing.Union[tuple, list, None] = None, mean: typing.Union[float, list, torch.Tensor, None] = None, std: typing.Union[float, list, torch.Tensor, None] = None, transformer=None):
        """
        initialize the AutoInputDataset
        :param image_list:
        :param size: size to which images will be resized
        :param mean: mean for normalization
        :param std: standard deviation for normalization
        :param transformer: custom transformer for image preprocessing
        """

        if transformer is not None and (mean is not None or std is not None):
            warnings.warn('NormalizeModel uses transformer for normalizing not (mean, std).')

        if transformer is not None:
            self.transform = transformer
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ])

        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx]
        image = cv2.cvtColor(processingtools.functions.imread(path), cv2.COLOR_BGR2RGB)
        return self.transform(image), path
