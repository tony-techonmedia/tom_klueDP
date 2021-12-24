#!/usr/bin/env python
# coding: utf-8


from klue_baseline.models import DPTransformer
from klue_baseline.data import KlueDPProcessor, KlueDataModule
from klue_baseline.data.klue_dp import KlueDPDataModule
from transformers import AutoTokenizer
import torch, csv
from tqdm import tqdm


class DPResult:
    """Result object for Dependency Parsing"""

    def __init__(self, heads: torch.Tensor, types: torch.Tensor) -> None:
        self.heads = heads
        self.types = types


def dp(args):
    model = DPTransformer(args)
    device = torch.device('cuda')
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    processor = KlueDPProcessor(args, tokenizer)
    data_module = KlueDPDataModule(args, processor)
    
    dataloader = data_module.get_dataloader('test', batch_size=32)
    
    prs = []
    gts = []
    with torch.no_grad():
        model.eval()
        for i, data in tqdm(enumerate(dataloader)):

            input_ids, masks, ids, max_word_length = data
            attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = masks
            head_ids, type_ids, pos_ids = ids
            inputs = {"input_ids": input_ids.cuda(), "attention_mask": attention_mask.cuda()}

            batch_index = torch.arange(0, head_ids.size()[0]).long().cuda()

            out_arc, out_type = model.forward(
                bpe_head_mask.cuda(),
                bpe_tail_mask.cuda(),
                pos_ids.cuda(),
                head_ids.cuda(),
                max_word_length,
                mask_e.cuda(),
                mask_d.cuda(),
                batch_index.cuda(),
                is_training=False,
                **inputs,
            )

            # predict arc and its type
            heads = torch.argmax(out_arc, dim=2)
            types = torch.argmax(out_type, dim=2)

            preds = DPResult(heads, types)
            labels = DPResult(head_ids, type_ids)

            prs.append(preds)
            gts.append(labels)

        model.write_prediction_file(prs,gts)
        
    