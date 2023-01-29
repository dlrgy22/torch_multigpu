import torch
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer():
    def __init__(self, model, train_loader, valid_loader, test_loader, train_sampler, cfg):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_loader = test_loader
        self.train_sampler = train_sampler

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), cfg.lr, weight_decay=1e-2)

        self.cfg = cfg

        self.scaler = GradScaler()
    
    def train_epoch(self, epoch):
        losses = AverageMeter()
        accuracy_scores = AverageMeter()

        if self.cfg.local_rank != -1:
            self.train_sampler.set_epoch(epoch)

        self.model.train()

        if self.cfg.rank == 0:
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), position=0, leave=True)
        else:
            pbar = enumerate(self.train_loader)

        for step, (image, label) in pbar:
            image, label = image.to(self.cfg.device), label.to(self.cfg.device)
            batch_size = image.size(0)

            # with autocast():
            preds = self.model(image)
            loss = self.loss_fn(preds, label)
            
            preds = torch.argmax(preds, 1).detach().cpu().numpy()
            acc = accuracy_score(preds, label.detach().cpu())

            losses.update(loss.item(), batch_size)
            accuracy_scores.update(acc, batch_size)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            if self.cfg.rank == 0:
                description = f"TRAIN  EPOCH {epoch} loss: {losses.avg: .4f} acc: {accuracy_scores.avg: .4f}"
                pbar.set_description(description)

        return losses.avg
    
    def test_epoch(self, epoch):
        losses = AverageMeter()
        accuracy_scores = AverageMeter()

        self.model.eval()
        
        if self.cfg.rank == 0:
            pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), position=0, leave=True)
        else:
            pbar = enumerate(self.valid_loader)

        for step, (image, label) in pbar:
            image, label = image.to(self.cfg.device), label.to(self.cfg.device)
            batch_size = image.size(0)

            with torch.no_grad():
                preds = self.model(image)
                loss = self.loss_fn(preds, label)

            preds = torch.argmax(preds, 1).detach().cpu().numpy()
            acc = accuracy_score(preds, label.detach().cpu())

            losses.update(loss.item(), batch_size)
            accuracy_scores.update(acc, batch_size)

            if self.cfg.rank == 0:
                description = f"Test  EPOCH {epoch} loss: {losses.avg: .4f} acc: {accuracy_scores.avg: .4f}"
                pbar.set_description(description)

        return losses.avg
    
    def train(self):
        best_loss = 1e8
        best_score = 0

        for epoch in range(self.cfg.epoch):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.test_epoch(epoch)

            if best_score >= best_score:
                torch.save(self.model.state_dict(), self.cfg.save_path)
                best_score = best_score

        return best_score