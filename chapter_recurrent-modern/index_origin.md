# Modern Recurrent Neural Networks
:label:`chap_modern_rnn`

We have introduced the basics of RNNs,
which can better handle sequence data.
For demonstration,
we implemented RNN-based
language models on text data.
However, 
such techniques may not be sufficient
for practitioners when they face
a wide range of sequence learning problems nowadays.

For instance,
a notable issue in practice
is the numerical instability of RNNs.
Although we have applied implementation tricks
such as gradient clipping,
this issue can be alleviated further
with more sophisticated designs of sequence models.
Specifically,
gated RNNs are much more common in practice.
We will begin by introducing two of such widely-used networks,
namely *gated recurrent units* (GRUs) and *long short-term memory* (LSTM).
Furthermore, we will expand the RNN architecture
with a single undirectional hidden layer
that has been discussed so far.
We will describe deep architectures with
multiple hidden layers,
and discuss the bidirectional design
with both forward and backward recurrent computations.
Such expansions are frequently adopted
in modern recurrent networks.
When explaining these RNN variants,
we continue to consider
the same language modeling problem introduced in :numref:`chap_rnn`.

In fact, language modeling
reveals only a small fraction of what 
sequence learning is capable of.
In a variety of sequence learning problems,
such as automatic speech recognition, text to speech, and machine translation,
both inputs and outputs are sequences of arbitrary length.
To explain how to fit this type of data,
we will take machine translation as an example,
and introduce the encoder-decoder architecture based on
RNNs and beam search for sequence generation.

```toc
:maxdepth: 2

gru
lstm
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```

