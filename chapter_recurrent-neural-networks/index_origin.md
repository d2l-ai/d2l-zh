# Recurrent Neural Networks
:label:`chap_rnn`

So far we encountered two types of data: tabular data and image data.
For the latter we designed specialized layers to take advantage of the regularity in them.
In other words, if we were to permute the pixels in an image, it would be much more difficult to reason about its content of something that would look much like the background of a test pattern in the times of analog TV.

Most importantly, so far we tacitly assumed that our data are all drawn from some distribution,
and all the examples are independently and identically distributed (i.i.d.).
Unfortunately, this is not true for most data. For instance, the words in this paragraph are written in sequence, and it would be quite difficult to decipher its meaning if they were permuted randomly.
Likewise, image frames in a video, the audio signal in a conversation, and the browsing behavior on a website, all follow sequential order.
It is thus reasonable to assume that specialized models for such data will do better at describing them.

Another issue arises from the fact that we might not only receive a sequence as an input but rather might be expected to continue the sequence.
For instance, the task could be to continue the series $2, 4, 6, 8, 10, \ldots$ This is quite common in time series analysis, to predict the stock market, the fever curve of a patient, or the acceleration needed for a race car. Again we want to have models that can handle such data.

In short, while CNNs can efficiently process spatial information, *recurrent neural networks* (RNNs) are designed to better handle sequential information.
RNNs introduce state variables to store past information, together with the current inputs, to determine the current outputs.

Many of the examples for using recurrent networks are based on text data. Hence, we will emphasize language models in this chapter. After a more formal review of sequence data we introduce practical techniques for preprocessing text data.
Next, we discuss basic concepts of a language model and use this discussion as the inspiration for the design of RNNs.
In the end, we describe the gradient calculation method for RNNs to explore problems that may be encountered when training such networks.

```toc
:maxdepth: 2

sequence
text-preprocessing
language-models-and-dataset
rnn
rnn-scratch
rnn-concise
bptt
```

