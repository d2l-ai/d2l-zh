# Natural Language Processing: Applications
:label:`chap_nlp_app`

We have seen how to represent tokens in text sequences and train their representations in :numref:`chap_nlp_pretrain`.
Such pretrained text representations can be fed to various models for different downstream natural language processing tasks.

In fact,
earlier chapters have already discussed some natural language processing applications
*without pretraining*,
just for explaining deep learning architectures.
For instance, in :numref:`chap_rnn`,
we have relied on RNNs to design language models to generate novella-like text.
In :numref:`chap_modern_rnn` and :numref:`chap_attention`,
we have also designed models based on RNNs and attention mechanisms for machine translation.

However, this book does not intend to cover all such applications in a comprehensive manner.
Instead,
our focus is on *how to apply (deep) representation learning of languages to addressing natural language processing problems*.
Given pretrained text representations,
this chapter will explore two 
popular and representative
downstream natural language processing tasks:
sentiment analysis and natural language inference,
which analyze single text and relationships of text pairs, respectively.

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on how to design models for different downstream natural language processing applications.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

As depicted in :numref:`fig_nlp-map-app`,
this chapter focuses on describing the basic ideas of designing natural language processing models using different types of deep learning architectures, such as MLPs, CNNs, RNNs, and attention.
Though it is possible to combine any pretrained text representations with any architecture for either application in :numref:`fig_nlp-map-app`,
we select a few representative combinations.
Specifically, we will explore popular architectures based on RNNs and CNNs for sentiment analysis.
For natural language inference, we choose attention and MLPs to demonstrate how to analyze text pairs.
In the end, we introduce how to fine-tune a pretrained BERT model
for a wide range of natural language processing applications,
such as on a sequence level (single text classification and text pair classification)
and a token level (text tagging and question answering).
As a concrete empirical case,
we will fine-tune BERT for natural language inference.

As we have introduced in :numref:`sec_bert`,
BERT requires minimal architecture changes
for a wide range of natural language processing applications.
However, this benefit comes at the cost of fine-tuning
a huge number of BERT parameters for the downstream applications.
When space or time is limited,
those crafted models based on MLPs, CNNs, RNNs, and attention
are more feasible.
In the following, we start by the sentiment analysis application
and illustrate the model design based on RNNs and CNNs, respectively.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```

