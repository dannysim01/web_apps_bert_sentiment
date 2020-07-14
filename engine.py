import torch
import torch.nn as nn
from tqdm import tqdm   # monitor progress

def loss_fn(outputs, target):
    return nn.BCEWithLogitsLoss()(outputs, target.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        target = d["target"]

        ids = ids.to(device, dtype=torch.long)              # send to cuda device
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, target)        # find loss
        loss.backward()                         # backward propagation

        optimizer.step()
        scheduler.step()

        """ stop the optimizer only after a certain number of accumulation steps """

        # if (bi + 1) % accumulation_steps == 0:
        #     optimizer.step()
        #     scheduler.step

def eval_fn(data_loader, model, device):
    model.eval()
    fin_target = []                         # final targets
    fin_outputs = []                        # final outputs
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_type_ids"]
            target = d["target"]

            ids = ids.to(device, dtype=torch.long)              # send to cuda device
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            # loss = loss_fn(outputs, targets)        # find loss, its bettwer to evaluate loss in eval fn

            fin_target.extend(target.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_target


