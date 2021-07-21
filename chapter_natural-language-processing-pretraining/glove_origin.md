# Word Embedding with Global Vectors (GloVe)
:label:`sec_glove`


Word-word co-occurrences 
within context windows
may carry rich semantic information.
For example,
in a large corpus
word "solid" is
more likely to co-occur
with "ice" than "steam",
but word "gas"
probably co-occurs with "steam"
more frequently than "ice".
Besides,
global corpus statistics
of such co-occurrences
can be precomputed:
this can lead to more efficient training.
To leverage statistical
information in the entire corpus
for word embedding,
let us first revisit
the skip-gram model in :numref:`subsec_skip-gram`,
but interpreting it
using global corpus statistics
such as co-occurrence counts.

## Skip-Gram with Global Corpus Statistics
:label:`subsec_skipgram-global`

Denoting by $q_{ij}$
the conditional probability
$P(w_j\mid w_i)$
of word $w_j$ given word $w_i$
in the skip-gram model,
we have

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

where 
for any index $i$
vectors $\mathbf{v}_i$ and $\mathbf{u}_i$
represent word $w_i$
as the center word and context word,
respectively, and $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ 
is the index set of the vocabulary.

Consider word $w_i$
that may occur multiple times
in the corpus.
In the entire corpus,
all the context words
wherever $w_i$ is taken as their center word
form a *multiset* $\mathcal{C}_i$
of word indices
that *allows for multiple instances of the same element*.
For any element,
its number of instances is called its *multiplicity*.
To illustrate with an example,
suppose that word $w_i$ occurs twice in the corpus
and indices of the context words
that take $w_i$ as their center word
in the two context windows
are 
$k, j, m, k$ and $k, l, k, j$.
Thus, multiset $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$, where 
multiplicities of elements $j, k, l, m$
are 2, 4, 1, 1, respectively.

Now let us denote the multiplicity of element $j$ in
multiset $\mathcal{C}_i$ as $x_{ij}$.
This is the global co-occurrence count 
of word $w_j$ (as the context word)
and word $w_i$ (as the center word)
in the same context window
in the entire corpus.
Using such global corpus statistics,
the loss function of the skip-gram model 
is equivalent to

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

We further denote by
$x_i$
the number of all the context words
in the context windows
where $w_i$ occurs as their center word,
which is equivalent to $|\mathcal{C}_i|$.
Letting $p_{ij}$
be the conditional probability
$x_{ij}/x_i$ for generating
context word $w_j$ given center word $w_i$,
:eqref:`eq_skipgram-x_ij`
can be rewritten as

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

In :eqref:`eq_skipgram-p_ij`, $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ calculates
the cross-entropy 
of
the conditional distribution $p_{ij}$
of global corpus statistics
and
the
conditional distribution $q_{ij}$
of model predictions.
This loss
is also weighted by $x_i$ as explained above.
Minimizing the loss function in 
:eqref:`eq_skipgram-p_ij`
will allow
the predicted conditional distribution
to get close to
the conditional distribution
from the global corpus statistics.


Though being commonly used
for measuring the distance
between probability distributions,
the cross-entropy loss function may not be a good choice here. 
On one hand, as we mentioned in :numref:`sec_approx_train`, 
the cost of properly normalizing $q_{ij}$
results in the sum over the entire vocabulary,
which can be computationally expensive.
On the other hand, 
a large number of rare 
events from a large corpus
are often modeled by the cross-entropy loss
to be assigned with
too much weight.

## The GloVe Model

In view of this,
the *GloVe* model makes three changes
to the skip-gram model based on squared loss :cite:`Pennington.Socher.Manning.2014`:

1. Use variables $p'_{ij}=x_{ij}$ and $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ 
that are not probability distributions
and take the logarithm of both, so the squared loss term is $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Add two scalar model parameters for each word $w_i$: the center word bias $b_i$ and the context word bias $c_i$.
3. Replace the weight of each loss term with the weight function $h(x_{ij})$, where $h(x)$ is increasing in the interval of $[0, 1]$.

Putting all things together, training GloVe is to minimize the following loss function:

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

For the weight function, a suggested choice is: 
$h(x) = (x/c) ^\alpha$ (e.g $\alpha = 0.75$) if $x < c$ (e.g., $c = 100$); otherwise $h(x) = 1$.
In this case,
because $h(0)=0$,
the squared loss term for any $x_{ij}=0$ can be omitted
for computational efficiency.
For example,
when using minibatch stochastic gradient descent for training, 
at each iteration
we randomly sample a minibatch of *non-zero* $x_{ij}$ 
to calculate gradients
and update the model parameters. 
Note that these non-zero $x_{ij}$ are precomputed 
global corpus statistics;
thus, the model is called GloVe
for *Global Vectors*.

It should be emphasized that
if word $w_i$ appears in the context window of 
word $w_j$, then *vice versa*. 
Therefore, $x_{ij}=x_{ji}$. 
Unlike word2vec
that fits the asymmetric conditional probability
$p_{ij}$,
GloVe fits the symmetric $\log \, x_{ij}$.
Therefore, the center word vector and
the context word vector of any word are mathematically equivalent in the GloVe model. 
However in practice, owing to different initialization values,
the same word may still get different values
in these two vectors after training:
GloVe sums them up as the output vector.



## Interpreting GloVe from the Ratio of Co-occurrence Probabilities


We can also interpret the GloVe model from another perspective. 
Using the same notation in 
:numref:`subsec_skipgram-global`,
let $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$ be the conditional probability of generating the context word $w_j$ given $w_i$ as the center word in the corpus. 
:numref:`tab_glove`
lists several co-occurrence probabilities
given words "ice" and "steam"
and their ratios based on  statistics from a large corpus.


:Word-word co-occurrence probabilities and their ratios from a large corpus (adapted from Table 1 in :cite:`Pennington.Socher.Manning.2014`:)


|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`


We can observe the following from :numref:`tab_glove`:

* For a word $w_k$ that is related to "ice" but unrelated to "steam", such as $w_k=\text{solid}$, we expect a larger ratio of co-occurence probabilities, such as 8.9.
* For a word $w_k$ that is related to "steam" but unrelated to "ice", such as $w_k=\text{gas}$, we expect a smaller ratio of co-occurence probabilities, such as 0.085.
* For a word $w_k$ that is related to both "ice" and "steam", such as $w_k=\text{water}$, we expect a ratio of co-occurence probabilities that is close to 1, such as 1.36.
* For a word $w_k$ that is unrelated to both "ice" and "steam", such as $w_k=\text{fashion}$, we expect a ratio of co-occurence probabilities that is close to 1, such as 0.96.




It can be seen that the ratio
of co-occurrence probabilities
can intuitively express
the relationship between words. 
Thus, we can design a function
of three word vectors
to fit this ratio.
For the ratio of co-occurrence probabilities
${p_{ij}}/{p_{ik}}$
with $w_i$ being the center word
and $w_j$ and $w_k$ being the context words,
we want to fit this ratio
using some function $f$:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

Among many possible designs for $f$,
we only pick a reasonable choice in the following.
Since the ratio of co-occurrence probabilities
is a scalar,
we require that
$f$ be a scalar function, such as
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$. 
Switching word indices
$j$ and $k$ in :eqref:`eq_glove-f`,
it must hold that
$f(x)f(-x)=1$,
so one possibility is $f(x)=\exp(x)$,
i.e., 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Now let us pick
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$,
where $\alpha$ is a constant.
Since $p_{ij}=x_{ij}/x_i$, after taking the logarithm on both sides we get $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$. 
We may use additional bias terms to fit $- \log\, \alpha + \log\, x_i$, such as the center word bias $b_i$ and the context word bias $c_j$:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

Measuring the squared error of
:eqref:`eq_glove-square` with weights,
the GloVe loss function in
:eqref:`eq_glove-loss` is obtained.



## Summary

* The skip-gram model can be interpreted using global corpus statistics such as word-word co-occurrence counts.
* The cross-entropy loss may not be a good choice for measuring the difference of two probability distributions, especially for a large corpus. GloVe uses squared loss to fit precomputed global corpus statistics.
* The center word vector and the context word vector are mathematically equivalent for any word in GloVe.
* GloVe can be interpreted from the ratio of word-word co-occurrence probabilities.


## Exercises

1. If words $w_i$ and $w_j$ co-occur in the same context window, how can we use their   distance in the text sequence to redesign the method for  calculating the conditional probability $p_{ij}$? Hint: see Section 4.2 of the GloVe paper :cite:`Pennington.Socher.Manning.2014`.
1. For any word, are its center word bias  and context word bias mathematically equivalent in GloVe? Why?


[Discussions](https://discuss.d2l.ai/t/385)
