# Introduction to Deep Learning

In 1950, Turing put forward the question of “Can machines think?” in his famous paper "Computing machinery and intelligence", and claimed to believe this is possible. In what he described as the Turing test, a machine can be considered intelligent if it is difficult for a human evaluator to distinguish between the replies from a machine and a human being through textual interactions. To this day, the development of intelligent machines is changing rapidly and continuously.

## Artificial Intelligence

In layman's terms, the intelligence displayed by the machine is called artificial intelligence. After the 1950s, the field of artificial intelligence experienced continuous flourishment. Today, artificial intelligence has approached or reached human intelligence in various applications such as image recognition, speech recognition, and autonomous driving, and surpassed human intelligence in applications such as board games. To give a few examples,

* In 2016, Go program AlphaGo, which was developed by Google's DeepMind, defeated world champion Lee Sedol by 4-1 in five rounds. The later upgraded version of AlphaGo defeated world champion Ke Jie, who is also ranked first in the world, by 3-0.

* In 2017, the visual program developed by Momenta from China achieved a top-5 error rate of 2.25% in the ImageNet Large Visual Recognition Challenge for the task of 1000-class image classification. That is, for 97.75% of all test images, the correct answer is one of the five most likely categories given by the program.

* In 2018, an assistant program developed by Google can schedule appointments by talking to a barber's clerk on the phone, while the clerks were unaware that the other end of the phone is a machine during the oral conversation which lasted for nearly a minute.

So, how exactly should artificial intelligence be implemented? In the early days of artificial intelligence, many people thought that humans could make machines intelligent by setting enough rules in computer programs, which led to the birth of expert systems. Expert systems solve complex problems through knowledge bases and inference engines, where the knowledge base includes a large number of human-defined facts and rules, and the inference engine applies the rules to existing facts in the knowledge base to infer new facts. As a branch of artificial intelligence, the development of expert systems reached its peak in the 1980s.

Although expert systems are very effective for tasks that involve complex reasoning, it is difficult for them to solve problems that are easy for humans such as image recognition and speech recognition. We need another approach to artificial intelligence: machine learning.

## Machine Learning

Machine learning studies how to make computer systems use experiences to improve performance. Usually, these experiences exist in the form of data. For a certain task, we want the computer to learn the rules of getting output from the input from the provided data without explicit specifications of them. The input here can be a picture containing a cat or a dog, and the output can be either "cat" or "dog". Therefore, we can also consider machine learning as a data-based programming paradigm. In machine learning, the rules that a computer learns from data to transform input into output are also called models. With the ever-increasing data scale and continuous improvement of hardware performance, machine learning has gradually become the most attractive branch of artificial intelligence. Today, machine learning is almost ubiquitous whether it is recommendation of personalized merchandise or news or automatic interception of spam e-mails.

However, traditional machine learning methods also have their limitations, especially when the input is raw data like images and sounds. We have mentioned that machine learning models transform the input into output, if we can represent these input data in a better way, the model will transform them into the correct output more easily. For many years, people have relied on specialized knowledge about specific tasks to carefully transform raw data into appropriate representations first, and then transform these representations into outputs using machine learning methods.

Among the many branches of machine learning, representation learning focuses on automatically finding the right way to represent data in order to better transform the inputs into the correct outputs. Deep learning, the focus of this book, is one of methods for representation learning.

## Deep Learning

Deep learning are representation-learning methods with multiple levels of representation. At each level (starting from the raw data), deep learning transforms the representation of the level into a more advanced representation with a simple function. Therefore, deep learning models can also be regarded as functions composed of many simple functions. With the composition of enough such simple transformations, deep learning models can express very complex transformations.

Deep learning can progressively represent increasingly abstract concepts or patterns. Take pictures as an example, a picture is a bunch of pixel values. In deep learning models, the first level of representation of a picture is usually whether there is an edge at a particular position and angle. The second-level representation can usually combine these edges into interesting patterns, such as patterns. At the third level, the patterns of the previous level may be further combined into a pattern corresponding to a specific part of some object. In this way, the model can be represented in a step-by-step fashion. In the end, the model can easily complete the given task such as picture classification according to representation at the final level. It is worth mentioning that, as a type of representational learning, deep learning will automatically find the proper way to represent data at each level.

Deep learning models are usually based on neural network models. The output of each layer in a neural network model corresponds to the representation of each level of deep learning. Though various neural network models have been proposed during the past century, deep learning has only recently achieved a series of exciting achievements in applications such as image recognition, speech recognition, and games. This is mainly due to the advance of hardware such as the emergence of GPUs for general-purpose computing and growth of data in recent years. In addition, open source deep learning libraries have also rapidly enabled deep learning technology to be widely adopted in the academic world and the industry. Early deep learning practitioners need to be proficient in CUDA and C++, but as you read this book, you will find that you only need to understand basic Python programming to use deep learning.

In short, deep learning as a class of machine learning methods is an important way toward artificial intelligence. Actually, deep learning is applied to everything described at the beginning of this section: AlphaGo, the classification of images with a very low error rate, and the assistant programs that make phone calls like humans. In the coming years, we believe that the field of deep learning will still be full of opportunities and challenges.

## Summary

* The intelligence displayed by the machine is called artificial intelligence.
* Machine learning studies how to make computer systems use experience to improve performance. It is a branch of artificial intelligence and one way of implementing artificial intelligence.
* As a category of machine learning, representation learning focuses on how to automatically find the correct way to represent data.
* Deep learning is a representation learning method with multiple levels of representations. It can progressively represent increasingly abstract concepts or patterns.

## Exercise

* If we see the development of artificial intelligence as a new industrial revolution, then is the relationship between deep learning and data similar to the relationship between steam engines and coal? Why?
* What aspects of your life may be changed by deep learning?

## Scan QR code to get to [forum](https://discuss.gluon.ai/t/topic/746)


![](../img/qr_deep-learning-intro.svg)
