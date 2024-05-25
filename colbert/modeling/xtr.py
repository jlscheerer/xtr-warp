import torch

from transformers import AutoTokenizer, AutoModel, logging
from huggingface_hub import hf_hub_download

from colbert.parameters import DEVICE

TOKEN_EMBED_DIM = 128
QUERY_MAXLEN = 32
DOC_MAXLEN = 512
TOKEN_EMBED_DIM = 128


class XTRTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts, length=QUERY_MAXLEN):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenizer([text.lower() for text in texts], return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=length)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        return input_ids, attention_mask

# Source: https://huggingface.co/google/xtr-base-en/resolve/main/2_Dense/config.json
# Activation function is torch.nn.modules.linear.Identity
class XTRLinear(torch.nn.Module):
    def __init__(self, in_features=768, out_features=128, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, x):
        return self.linear(x)

class XTR(torch.nn.Module):
    def __init__(self, tokenizer, encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.linear = XTRLinear().to(DEVICE)
        self.encoder = encoder.to(DEVICE)

    @property
    def device(self):
        return DEVICE

    def forward(self, input_ids, attention_mask):
        Q = self.encoder(input_ids, attention_mask).last_hidden_state
        Q = self.linear(Q)
        mask = (input_ids != 0).unsqueeze(2).float()
        Q = Q * mask
        Q = torch.nn.functional.normalize(Q, dim=2)
        return Q

def build_xtr_model():
    # NOTE Warning about unitialized decoder weights is to be expected.
    #      We only make use of the encoder anyways.
    logging.set_verbosity_error()
    model = AutoModel.from_pretrained("google/xtr-base-en", use_safetensors=True)

    tokenizer = XTRTokenizer(AutoTokenizer.from_pretrained("google/xtr-base-en"))
    xtr = XTR(tokenizer, model.encoder)

    # Source: https://huggingface.co/google/xtr-base-en/
    to_dense_path = hf_hub_download(repo_id="google/xtr-base-en", filename="2_Dense/pytorch_model.bin")
    xtr.linear.load_state_dict(torch.load(to_dense_path))

    logging.set_verbosity_warning()
    return xtr


class XTRCheckpoint:
    def __init__(self, xtr, config, query_maxlen=QUERY_MAXLEN):
        self.xtr = xtr
        self.config = config
        self.query_maxlen = query_maxlen

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert not to_cpu
        assert keep_dims == "flatten"
        assert not showprogress
        assert not return_tokens
        assert self.config.doc_maxlen == DOC_MAXLEN
        assert bsize is not None

        input_ids, attention_mask = self.xtr.tokenizer(docs, self.config.doc_maxlen)
        
        text_batches = self._split_into_batches(input_ids, attention_mask, bsize)
        total_length = sum([torch.sum(attention_mask) for _, attention_mask in text_batches])

        batch_lengths = [torch.sum(attention_mask, dim=1) for _, attention_mask in text_batches]
        batches = [self.xtr(input_ids.to(DEVICE), attention_mask.to(DEVICE))
                   for input_ids, attention_mask in text_batches]
    
        flatten_embeddings = torch.zeros((total_length, TOKEN_EMBED_DIM), dtype=torch.float32)

        num_tokens = 0
        for batch_embeds, batch_length in zip(batches, batch_lengths):
            for _, (embeddings, length) in enumerate(zip(batch_embeds, batch_length)):
                flatten_embeddings[num_tokens:num_tokens+length] = embeddings[:int(length)].detach()
                num_tokens += int(length)

        assert num_tokens == flatten_embeddings.shape[0]
        return flatten_embeddings.half(), [x.item() for y in batch_lengths for x in y]

    def _split_into_batches(self, ids, mask, bsize):
        batches = []
        for offset in range(0, ids.size(0), bsize):
            batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))
    
        return batches
    
    def cuda(self):
        self.xtr = self.xtr.cuda()
        return self

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False):
        assert context is None
        assert full_length_search == False

        input_ids, attention_mask = self.xtr.tokenizer(queries, self.query_maxlen)
        if bsize is not None:
            batches = self._split_into_batches(input_ids, attention_mask, bsize)
            with torch.no_grad():
                if to_cpu:
                    return torch.cat([
                        self.xtr(input_ids.to(self.xtr.device), attention_mask.to(self.xtr.device)).cpu()
                        for input_ids, attention_mask in batches
                    ])
                return torch.cat([
                    self.xtr(input_ids.to(self.xtr.device), attention_mask.to(self.xtr.device)).cpu()
                    for input_ids, attention_mask in batches
                ])
        
        with torch.no_grad():
            encodings = self.xtr(input_ids.to(self.xtr.device), attention_mask.to(self.xtr.device))

        if to_cpu:
            encodings = encodings.cpu()
        return encodings

    # NOTE "Hack" to support self.checkpoint.query_tokenizer.query_maxlen assignment
    @property
    def query_tokenizer(self):
        class XTRQueryTokenizerPlaceholder:
            query_maxlen: int
        return XTRQueryTokenizerPlaceholder()