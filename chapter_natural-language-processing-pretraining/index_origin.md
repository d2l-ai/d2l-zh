# Natural Language Processing: Pretraining
:label:`chap_nlp_pretrain`


Humans need to communicate.
Out of this basic need of the human condition, a vast amount of written text has been generated on an everyday basis.
Given rich text in social media, chat apps, emails, product reviews, news articles,  research papers, and books, it becomes vital to enable computers to understand them to offer assistance or make decisions based on human languages.

*Natural language processing* studies interactions between computers and humans using natural languages.
In practice, it is very common to use natural language processing techniques to process and analyze text (human natural language) data, such as language models in :numref:`sec_language_model` and machine translation models in :numref:`sec_machine_translation`.

To understand text, we can begin by learning
its representations.
Leveraging the existing text sequences
from large corpora,
*self-supervised learning*
has been extensively
used to pretrain text representations,
such as by predicting some hidden part of the text
using some other part of their surrounding text.
In this way,
models learn through supervision
from *massive* text data
without *expensive* labeling efforts!


As we will see in this chapter,
when treating each word or subword as an individual token,
the representation of each token can be pretrained
using word2vec, GloVe, or subword embedding models
on large corpora.
After pretraining, representation of each token can be a vector,
however, it remains the same no matter what the context is.
For instance, the vector representation of "bank" is the same
in both
"go to the bank to deposit some money"
and
"go to the bank to sit down".
Thus, many more recent pretraining models adapt representation of the same token
to different contexts.
Among them is BERT, a much deeper self-supervised model based on the transformer encoder.
In this chapter, we will focus on how to pretrain such representations for text,
as highlighted in :numref:`fig_nlp-map-pretrain`.

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on the upstream text representation pretraining.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`


For sight of the big picture,
:numref:`fig_nlp-map-pretrain` shows that
the pretrained text representations can be fed to
a variety of deep learning architectures for different downstream natural language processing applications.
We will cover them in :numref:`chap_nlp_app`.

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining

```

