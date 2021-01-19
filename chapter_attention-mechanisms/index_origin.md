# Attention Mechanisms
:label:`chap_attention`

The optic nerve of a primate's visual system
receives massive sensory input,
far exceeding what the brain can fully process.
Fortunately,
not all stimuli are created equal.
Focalization and concentration of consciousness 
have enabled primates to direct attention
to objects of interest,
such as preys and predators, 
in the complex visual environment.
The ability of paying attention to 
only a small fraction of the information
has evolutionary significance,
allowing human beings 
to live and succeed.

Scientists have been studying attention 
in the cognitive neuroscience field
since the 19th century.
In this chapter,
we will begin by reviewing a popular framework
explaining how attention is deployed in a visual scene.
Inspired by the attention cues in this framework,
we will design models
that leverage such attention cues.
Notably, the Nadaraya-Waston kernel regression
in 1964 is a simple demonstration of machine learning with *attention mechanisms*.

Next, we will go on to introduce attention functions 
that have been extensively used in 
the design of attention models in deep learning.
Specifically,
we will show how to use these functions
to design the *Bahdanau attention*,
a groundbreaking attention model in deep learning
that can align bidirectionally and is differentiable.

In the end,
equipped with 
the more recent
*multi-head attention*
and *self-attention* designs,
we will describe the *Transformer* architecture
based solely on attention mechanisms.
Since their proposal in 2017,
Transformers
have been pervasive in modern 
deep learning applications,
such as in areas of
language,
vision, speech,
and reinforcement learning.

```toc
:maxdepth: 2

attention-cues
nadaraya-waston
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
```

