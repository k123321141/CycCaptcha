import torch
import os
import json
import argparse
import shutil
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from torchvision.transforms import v2
from data_utils import CaptchaDataset
from models import CLIPClassifier, CNNClassifier
from transformers import AutoProcessor
from tqdm import tqdm
from tensorboardX import SummaryWriter


def _generator_fn(it_obj):
    while True:
        for x in it_obj:
            yield x


def logits2str(logits, inv_vocab):
    logits = logits.detach().cpu().numpy()
    pred = np.argmax(logits, axis=2)
    str_list = [''.join(map(inv_vocab.get, idx_list)) for idx_list in pred.tolist()]
    return str_list


def label2str(label, inv_vocab):
    label = label.detach().cpu().numpy()
    str_list = [''.join(map(inv_vocab.get, idx_list)) for idx_list in label.tolist()]
    return str_list


def accuracy(logits, label, inv_vocab):
    pred_list = logits2str(logits, inv_vocab)
    true_list = label2str(label, inv_vocab)

    hit = 0
    char_hit = 0

    assert len(pred_list) == len(true_list)

    for pred, true in zip(pred_list, true_list):
        if pred == true:
            hit += 1
            char_hit += len(true)
        else:
            for p, t in zip(pred, true):
                if p == t:
                    char_hit += 1
    char_count = sum([len(true) for true in true_list])

    acc = hit / len(true_list)
    char_acc = char_hit / char_count

    return acc, char_acc, pred_list, true_list


def calc_loss(criterion, model, X, label, device):
    X = X.to(device)
    label = label.to(device)
    outputs = model(X)
    loss = criterion(outputs.swapdims(1, 2), label)
    logits = F.softmax(outputs, dim=2)
    return logits, loss


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', default='/tmp/vocab.json', type=str, required=False)
    parser.add_argument('--data_dir', default='/tmp/train/', type=str, required=False)
    parser.add_argument('--eval_data_dir', default='/tmp/val/', type=str, required=False)
    parser.add_argument('--test_data_dir', default='/tmp/test/', type=str, required=False)
    parser.add_argument('--steps', default=int(1e6), type=int, required=False)
    parser.add_argument('--batch_size', default=512, type=int, required=False)
    parser.add_argument('--eval_batch_size', default=512, type=int, required=False)
    parser.add_argument('--lr', default=1e-4, type=float, required=False)
    parser.add_argument('--log_step', default=5, type=int, required=False)
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False)
    parser.add_argument('-o', '--output', default='./save_models/', type=str, required=False, help='model output path.')
    parser.add_argument('--logdir', default='./logs/dev', type=str, required=False)
    parser.add_argument('--seed', default=1337, type=int, required=False)
    parser.add_argument('-w', '--worker', type=int, default=0)
    parser.add_argument('--cnn', action='store_true', default=False)
    args = parser.parse_args()
    print(f'args: {args.__repr__()}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    #
    if os.path.isdir(args.logdir) and len(os.listdir(args.logdir)) > 0:
        for sub_dir in os.listdir(args.logdir):
            if os.path.isdir(os.path.join(args.logdir, sub_dir)):
                shutil.rmtree(os.path.join(args.logdir, sub_dir))
            else:
                os.remove(os.path.join(args.logdir, sub_dir))
    tb_writer = SummaryWriter(logdir=args.logdir)

    train_dataset = CaptchaDataset(args.data_dir, not args.cnn)
    val_dataset = CaptchaDataset(args.eval_data_dir, not args.cnn)
    test_dataset = CaptchaDataset(args.test_data_dir, not args.cnn)

    # processor = AutoProcessor.from_pretrained("./ckpt")
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    inv_vocab = {i + 1: w for w, i in vocab.items()}
    inv_vocab[0] = ''

    if args.cnn:
        model = CNNClassifier(len(vocab) + 1, 20)
        processor = T.Compose([
            v2.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0)),
            T.Grayscale(),
            T.ToTensor(),
            ThresholdTransform(thr_255=127),
        ])

        def _collate_fn(batch):
            label_list = []
            tensor_list = []
            for label, img in batch:
                label_list.append([vocab[c.lower()] + 1 for c in label] + [0] * (20 - len(label)))
                img_tensor = processor(img)
                tensor_list.append(img_tensor)

            # to Tensor
            label_tensor = torch.LongTensor(label_list)
            inputs = torch.stack(tensor_list)
            return label_tensor, inputs, np.array(img).swapaxes(0, 2).swapaxes(1, 2)
    else:
        model = CLIPClassifier(len(vocab) + 1, 20)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def _collate_fn(batch):
            label_list = []
            img_list = []
            for label, img in batch:
                label_list.append([int(c) for c in label])
                img_list.append(img)

            inputs = processor(images=img_list, return_tensors="pt")

            # to Tensor
            label_tensor = torch.LongTensor(label_list)
            return label_tensor, inputs, np.array(img_list[0]).swapaxes(0, 2).swapaxes(1, 2)

    model.to(device)
    st = torch.load('./ckpt/iter_93759_acc_6.5.model')
    model.load_state_dict(st)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.worker,
        collate_fn=_collate_fn,
    )
    train_G = _generator_fn(dataloader)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.worker,
        collate_fn=_collate_fn,
    )
    val_G = _generator_fn(val_dataloader)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.worker,
        collate_fn=_collate_fn,
    )
    test_G = _generator_fn(test_dataloader)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('starting training')

    # default value
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    test_char_acc = test_acc = 0
    test_loss = torch.Tensor([999])
    current_lr = optimizer.param_groups[0]['lr']
    with tqdm(total=args.steps) as pbar:
        for overall_step in range(93759, args.steps):
            label, X, X_img = next(train_G)
            model.train()
            current_lr = optimizer.param_groups[0]['lr']

            # loss
            logits, loss = calc_loss(criterion, model, X, label, device)
            grad_loss = loss / args.gradient_accumulation
            grad_loss.backward()

            #  optimizer step
            if (overall_step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (overall_step + 1) % args.log_step == 0:
                #  validation
                with torch.no_grad():
                    model.eval()

                    logits, loss = calc_loss(criterion, model, X, label, device)
                    train_acc, train_char_acc, train_pred, train_true = accuracy(logits, label, inv_vocab)
                    tb_writer.add_image('train_pred', X_img, overall_step)
                    tb_writer.add_text('train', f'{train_true[0]} --> [{train_pred[0]}]', overall_step)

                    eval_label, eval_X, eval_X_img = next(val_G)
                    eval_logits, eval_loss = calc_loss(criterion, model, eval_X, eval_label, device)
                    eval_acc, eval_char_acc, eval_pred, eval_true = accuracy(eval_logits, eval_label, inv_vocab)
                    tb_writer.add_image('eval_pred', eval_X_img, overall_step)
                    tb_writer.add_text('eval', f'{eval_true[0]} --> [{eval_pred[0]}]', overall_step)

                    test_label, test_X, test_X_img = next(test_G)
                    test_logits, test_loss = calc_loss(criterion, model, test_X, test_label, device)
                    test_acc, test_char_acc, test_pred, test_true = accuracy(test_logits, test_label, inv_vocab)
                    tb_writer.add_image('test_pred', test_X_img, overall_step)
                    tb_writer.add_text('test', f'{test_true[0]} --> [{test_pred[0]}]', overall_step)

                    tb_writer.add_scalars('loss', {
                        'train': loss.item(),
                        'val': eval_loss.item(),
                        'test': test_loss.item(),
                    }, overall_step)
                    tb_writer.add_scalars('accuracy', {
                        'train': train_acc,
                        'val': eval_acc,
                        'test': test_acc,
                    }, overall_step)
                    tb_writer.add_scalars('char_accuracy', {
                        'train': train_char_acc,
                        'val': eval_char_acc,
                        'test': test_char_acc,
                    }, overall_step)

                    tb_writer.add_scalars('lr', {
                        'train': current_lr,
                    }, overall_step)

                    if eval_acc > best_acc:
                        best_acc = eval_acc
                        if os.path.isdir(args.output):
                            shutil.rmtree(args.output)
                        os.makedirs(args.output)
                        filename = os.path.join(args.output, f'iter_{overall_step}_acc_{best_acc*100:.1f}.model')
                        print(f'save file to {filename}')
                        torch.save(model.state_dict(), filename)
                    tb_writer.flush()
            pbar.set_postfix(loss=f'{loss.item():.2f}', lr=f'{current_lr:.2e}, test_loss={test_loss.item():.2f}, test_char_acc={test_char_acc*100:.1f}, test_acc={test_acc*100:.1f}')
            pbar.update(1)
    tb_writer.close()

    print('training finished')


if __name__ == '__main__':
    main()
