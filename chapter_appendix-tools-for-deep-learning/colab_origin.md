# Using Google Colab
:label:`sec_colab`

We introduced how to run this book on AWS in :numref:`sec_sagemaker` and :numref:`sec_aws`. Another option is running this book on [Google Colab](https://colab.research.google.com/), which provides free GPU if you have a Google account.

To run a section on Colab, you can simply click the `Colab` button to the right of the title of that section, such as in :numref:`fig_colab`. 

![Open a section on Colab](../img/colab.png)
:width:`300px`
:label:`fig_colab`


When it is the first time you execute a code cell, you will receive a warning message as shown in :numref:`fig_colab2`. You may click "RUN ANYWAY" to ignore it.

![The warning message for running a section on Colab](../img/colab-2.png)
:width:`300px`
:label:`fig_colab2`

Next, Colab will connect you to an instance to run this notebook. Specifically, if GPU is needed, such as when invoking the `d2l.try_gpu()` function, we will request Colab to connect to a GPU instance automatically.


## Summary

* You can use Google Colab to run each section of this book with GPUs.


## Exercises

1. Try to edit and run the code in this book using Google Colab.


[Discussions](https://discuss.d2l.ai/t/424)
