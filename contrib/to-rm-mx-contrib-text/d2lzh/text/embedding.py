import os
from mxnet import nd, gluon
import tarfile
import zipfile

DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
PRETRAINED_FILE = {
    'glove':{},
    'fasttext':{}
}
PRETRAINED_FILE['glove']['glove.6b.50d.txt'] = (DATA_URL + 'glove.6B.50d.zip',
                                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')
PRETRAINED_FILE['glove']['glove.6b.100d.txt'] = (DATA_URL + 'glove.6B.100d.zip',
                                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
PRETRAINED_FILE['glove']['glove.42b.300d.txt'] = (DATA_URL + 'glove.42B.300d.zip',
                                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')
PRETRAINED_FILE['fasttext']['wiki.en'] = (DATA_URL + 'wiki.en.zip',
                                          'c1816da3821ae9f43899be655002f6c723e91b88')

def mkdir_if_not_exist(path):
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)

def download(embedding_name, pretrained_file_name, cache_dir=os.path.join('..', 'data')):
    url, sha1 = PRETRAINED_FILE[embedding_name][pretrained_file_name]
    mkdir_if_not_exist(cache_dir)
    return gluon.utils.download(url, cache_dir, sha1_hash=sha1)

def download_extract(embedding_name, pretrained_file_name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(embedding_name, pretrained_file_name)
    base_dir = os.path.dirname(fname) 
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted'
    fp.extractall(base_dir)
    if folder:
        return os.path.join(base_dir, folder)
    else:
        return data_dir
    
def get_pretrained_file_names(embedding_name=None):
    if embedding_name is not None:
        return PRETRAINED_FILE[embedding_name].keys()
    else:
        return PRETRAINED_FILE
    
def create(embedding_name, pretrained_file_name, vocabulary=None):
    return TokenEmbedding(embedding_name, pretrained_file_name.lower(), vocabulary)
    
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name, pretrained_file_name, vocabulary=None):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name, pretrained_file_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}
        if vocabulary is not None:
            indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in vocabulary.idx_to_token]
            self.idx_to_vec = self.idx_to_vec[nd.array(indices)]
            self.token_to_idx = vocabulary.token_to_idx
            self.idx_to_token = vocabulary.idx_to_token
            
    def _load_embedding(self, embedding_name, pretrained_file_name):
        """
        Load a pretrained embedding from a file.

        Parameters
        ----------
        embedding_name : str, default 'glove'
            The name of the pretrained word embedding to
        load. Can be 'word2vec', 'glove', or 'fasttext'.
        pretrained_file_name : str, default None
            The name of the local file containing pre-trained word
        vectors to load. If not specified, will use the `embedding_name` instead (which is usually fine).

        Returns
        -------    
         idx_to_token : list of strs
        Shape: [n_tokens]. token for each row in idxToVec matrix  (row 0 is for padding & unknown words)   # no longer needed?       # TODO: check if this is
        still necessary/used - probably not!         # TODO: remove this line after checking         vocab = Vocab(counter=None)          vocab._idxToToken =
        ['<unk>'] + [e[0] for e in sorted(vocab._tokenToIdx.items(), key=lambda e: e[1])]      return
        """
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = download_extract(embedding_name, pretrained_file_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, nd.array(idx_to_vec)

    def get_vecs_by_tokens(self, tokens):
        """
        Get the vector representation of each token in a list of tokens.

        Parameters
        ----------
        tokens : List[str] or List[int]  # Note that this is *not* a
        numpy array! (It's important for doing batch lookup)

            A list of tokens to be converted into vectors.

            If the input is an integer, it will just
        return the corresponding row from `idx_to_vec`.

            Otherwise, if it's a string, then it'll first look up which row in `token_to_idx` corresponds to
        each token and then do a vector lookup using those indices. This means that if you have new unknown words in test data but not training data, you can
        still use GloVe vectors rather than having to add them as zero vectors at index 0 (which slows down computation). You can also pass in integers
        directly instead of strings too; they'll just get passed through as-is. The reason we need both `token_to_idx` and `idx_to_vec` is because we want
        fast lookups for both training and test time: during training time we construct these dictionaries by iterating over all documents once so that they
        contain all relevant
        """
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[nd.array(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)