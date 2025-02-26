import torch
import torch.utils.data
import processingtools.ProgressBar
import processingtools.functions
import torchvision
import cv2
import typing
import warnings


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
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            return torch.unsqueeze(self.transform(image), dim=0)
        except Exception as e:
            raise ValueError(f'Error reading image from path {path}: {e}')

    def get_device(self) -> None:
        """
        get the device to run the model on
        :return: device
        """

        self.device = next(self.model.parameters()).device
        print(f'run on {processingtools.functions.s_text(f"{self.device}", styles=("bold",))}')

    @processingtools.functions.custom_warning_format
    @torch.no_grad()
    def forward(self, inputs: typing.Union[str, list], batch_size: int = 1, num_workers: int = 0) -> typing.Union[torch.Tensor, dict]:
        """
        forward pass of the model
        :param inputs: string or list of strings representing image paths
        :param batch_size: batch size if input is path list
        :param num_workers: workers num for dataloader if input is path list
        :return: model outputs
        """

        if self.model.training:
            warnings.warn('model is training mode now! (if you want to change eval mode, add model.eval())')

        self.get_device()

        if isinstance(inputs, list):
            outputs = []
            out_dict = {'results': {}}

            dataset = AutoInputDataset(inputs, transformer=self.transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            for data in processingtools.ProgressBar(dataloader, total=len(dataloader), finish_mark=None):
                image, paths = data
                image = image.to(self.device)
                output = self.model(image)

                outputs.append(output)

                for out, path in zip(output, paths):
                    out_dict['results'][path] = out

            print('\rInference done.')
            out_dict['total outputs'] = torch.cat(outputs, dim=0)

            return out_dict

        elif isinstance(inputs, str):
            return self.model(self.image_read(inputs))
        else:
            raise TypeError('inputs must be a string or a list of strings')


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
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.transform(image), path
