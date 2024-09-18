TOKEN_EMBED_DIM = 128
QUERY_MAXLEN = 32
DOC_MAXLEN = 512
TOKEN_EMBED_DIM = 128

class TPrimePolicy:
    def __init__(self, value):
        self.value = value    
    def __getitem__(self, document_top_k):
        return self.value

class TPrimeMaxPolicy:
    def __getitem__(self, document_top_k):
        if document_top_k > 100:
            return 100_000
        return 50_000
T_PRIME_MAX = TPrimeMaxPolicy()