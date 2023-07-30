import torch
import processingtools.PrgressBar
import processingtools.functions


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

            for inputs, targets in processingtools.PrgressBar.ProgressBar(self.train_loader, total=len(self.train_loader), detail_func=lambda _: f'loss: {loss:0.3f}' if loss is not None else '...', finish_mark=None):
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

        for inputs, targets in processingtools.PrgressBar.ProgressBar(self.test_loader, total=len(self.test_loader), finish_mark=None, detail_func=lambda _: f'acc: {correct / data_size: 0.3f} %' if data_size != 0 else 'acc: 0 %'):
            inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            data_size = data_size + targets.shape[0]

            outputs = self.model(inputs)
            correct = correct + int(torch.sum(torch.argmax(outputs, dim=1) == targets))

        print('\r', end='')
        print_recoder(self.recoder, f'accuracy: {(correct / data_size) * 100: 0.2f} %')

        return correct


class BundleLoss(torch.nn.Module):
    """
    Bundle input criterions
    """

    def __init__(self, criterions: list, reduction: str = 'Sum'):
        super().__init__()
        self.criterions = criterions
        self.reduction = reduction

    def forward(self, predictions, targets):
        loss = None
        if self.reduction == 'Sum':
            loss = 0
            for criterion in self.criterions:
                loss = loss + criterion(predictions, targets)

        elif self.reduction == 'None':
            loss = list()
            for criterion in self.criterions:
                loss.append(criterion(predictions, targets))

        return loss


def print_recoder(recoder, string: str):
    recoder.print(string) if recoder is not None else print(string)

