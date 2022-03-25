import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KoBARTSummaryDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import textwrap
import string
import re
from pytorch_lightning.loggers import WandbLogger
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

parser = argparse.ArgumentParser(description='KoBART Seq2Seq')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--wandb_project',
                    type=str,
                    help='Name of the wandb project')

parser.add_argument('--run_name',
                    type=str,
                    help='Name of the wandb run')

parser.add_argument('--gpu_nums',
                    type=str,
                    help='A list of gpus that are usable')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/nl2url_v2.0.0_train.tsv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='data/nl2url_v2.0.0_validation.tsv',
                            help='test file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

class KobartSummaryModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok,
                 max_len=512,
                 batch_size=8,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        if tok is None:
            self.tok = get_kobart_tokenizer()
        else:
            self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KoBARTSummaryDataset(self.train_file_path,
                                 self.tok,
                                 self.max_len)
        self.test = KoBARTSummaryDataset(self.test_file_path,
                                self.tok,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()

    def forward(self, inputs):

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        #decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['labels'].ne(self.pad_token_id).float()
        #print(attention_mask)
        #print(decoder_attention_mask)
        #exit()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          #decoder_input_ids=inputs['decoder_input_ids'],
                          #decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)

    def bleu(self, gen, ref):
        ref = [ref.split()]
        gen = gen.split()
        score_bleu = sentence_bleu(ref,gen)
        return score_bleu

    def clean_up(self, text):
        '''
        text = text.replace(".", '')
        text = text.replace(',', '')
        text = text.replace("'", '')
        text = text.replace('"', '')
        '''
        text = text.replace("<s>", "")
        text =text.replace('<pad>', '')
        text = text.replace('</s>', '')
        text = text.replace('<usr>', '')
        return text  

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        '''
        def lower(text):
            return text.lower()
        '''
        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text
        return rid_of_specials(white_space_fix(remove_articles(remove_punc(s))))

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def approx_match_score(self, prediction, ground_truth):
        answer = self.normalize_answer(prediction) 
        gt = self.normalize_answer(ground_truth)
        match = 0
        gt_words = gt.split(" ")
        for word in gt_words:
            if word in answer:
                match = 1
                return match
        return match 

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _generative_step(self, batch):
        attention_mask = batch['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = batch['decoder_input_ids'].ne(self.pad_token_id).float()

        outs = self.model.generate(
            batch["input_ids"].cuda(),
            attention_mask=attention_mask.cuda(),
            use_cache=True,
            #decoder_attention_mask=decoder_attention_mask.cuda(),
            max_length=100,
            num_beams=5,
            eos_token_id=1,
            #decoder_start_token_id=self.pad_token_id
            #early_stopping=True,
            #no_repeat_ngram_size=3
        )
        target2 = []
        for ids in batch['labels']:
            new_ids = [0 if x == -100 else x for x in ids]
            target2.append(new_ids)

        dec = [self.tokenizer.decode(ids) for ids in outs]
        texts = [self.tokenizer.decode(ids) for ids in batch['input_ids']]
        #targets = [tokenizer.decode(ids, for ids in batch['labels']]
        targets = [self.tokenizer.decode(ids) for ids in target2]
        batch_len = len(batch['input_ids'])
        em_correct_num = 0
        subset_correct_num = 0
        bleu_score = 0
        for i in range(len(batch['input_ids'])):
            lines = textwrap.wrap("\n%s\n" % texts[i], width=3000)
            lines = self.clean_up(lines[0])
            ground_truth = self.clean_up(targets[i])
            predicted = self.clean_up(dec[i])
            em = self.exact_match_score(predicted, ground_truth)
            subset = self.approx_match_score(predicted, ground_truth)  
            if i == 0:      
                print(f'INPUT : {lines}')
                print(f'GROUD TRUTH: {ground_truth}, MODEL OUTPUT: {predicted}')
            if em == 1:
                em_correct_num+=1
            if subset == 1:
                subset_correct_num+=1
            bleu_score+=self.bleu(predicted, ground_truth)
        bleu_score = bleu_score / batch_len
        em_score = em_correct_num / batch_len
        subset_score = subset_correct_num / batch_len
        self.log('em_score', em_score, prog_bar=True, logger=True)
        self.log('subset_score', subset_score, prog_bar=True, logger=True)
        self.log('bleu_score', bleu_score, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        self._generative_step(batch)
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_nums
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.run_name)

    model = KoBARTConditionalGeneration(args)

    dm = KobartSummaryModule(args.train_file,
                        args.test_file,
                        None,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=args.num_workers)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
