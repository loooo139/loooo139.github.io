---
layout:     post
title:     Illustrated Guide to LSTM’s and GRU’s
subtitle:   A step by step explanation
date:       2018-10-23
author:     Frank
header-img: img/post-bg-ios8-web.jpg
catalog: true
tags:
    - LSTM
    - deep learning
---

# Illustrated Guide to LSTM’s and GRU’s: A step by step explanation

![img](https://cdn-images-1.medium.com/max/800/1*n-IgHZM5baBUjq0T7RYDBw.gif)

In this post,we will focus on intuition behind LSTM and GRU.

## The problem ,short-term memory

RNN suffer from short-term memory.if a sequence is too long ,it is very difficult to carry information from earlier time steps to later one. so it is unstable for processing a paragraph of text.

During the back propagation, **RNN** suffer from the gradient vanishing problem.from later layers to earlier layers the gradient become smaller and smaller even come to **zero**.so the earlier layer could not learn anything if vanishing gradient happens(smaller gradient, smaller contributions). it demonstrates RNN will forget the earlier information if the sequence is long enough.

![Gradient update rule](https://cdn-images-1.medium.com/max/800/1*PYiQa_bNzM8ugYz_D1yvgw.png)

##  LSTM’s and **GRU’s as a solution**



LSTM ’s and GRU’s were created as the solution to short-term memory. They have internal mechanisms called gates that can regulate the flow of information.

![img](https://cdn-images-1.medium.com/max/800/1*yBXV9o5q7L_CvY7quJt3WQ.png)

These gates can learn which data in a sequence is important to keep or throw away.By doing this ,it can pass relevant information down the long chain of sequences to make predictions.Almost all state of art results based on **RNN** are achieved with these two networks.LSTM’s and GRU’s can be found in speech recognition, speech synthesis, and text generation. You can even use them to generate captions for videos.

## Intuition



Ok, Let’s start with a thought experiment. Let’s say you’re looking at reviews online to determine if you want to buy Life cereal (don’t ask me why). You’ll first read the review then determine if someone thought it was good or if it was bad.



![img](https://cdn-images-1.medium.com/max/800/1*YHjfAgozQaghcsEvsBEu2g.png)

When you read the review, your brain subconsciously only remembers important keywords. You pick up words like “amazing” and “perfectly balanced breakfast”. You don’t care much for words like “this”, “gave“, “all”, “should”, etc. If a friend asks you the next day what the review said, you probably wouldn’t remember it word for word. You might remember the main points though like “will definitely be buying again”. If you’re a lot like me, the other words will fade away from memory.



![img](https://cdn-images-1.medium.com/max/800/1*ygAgowqTZjR6ABzZHd8Bqg.gif)

And that is essentially what an LSTM or GRU does. It can learn to keep only relevant information to make predictions, and forget non relevant data. In this case, the words you remembered made you judge that it was good.

### Review of Recurrent Neural Networks

To understand how LSTM’s or GRU’s achieves this, let’s review the recurrent neural network. An RNN works like this; First words get transformed into machine-readable vectors. Then the RNN processes the sequence of vectors one by one.



![img](https://cdn-images-1.medium.com/max/800/1*AQ52bwW55GsJt6HTxPDuMA.gif)

Processing sequence one by one

While processing, it passes the previous hidden state to the next step of the sequence. The hidden state acts as the neural networks memory. It holds information on previous data the network has seen before.



![img](https://cdn-images-1.medium.com/max/800/1*o-Cq5U8-tfa1_ve2Pf3nfg.gif)

Passing hidden state to next time step

Let’s look at a cell of the RNN to see how you would calculate the hidden state. First, the input and previous hidden state are combined to form a vector. That vector now has information on the current input and previous inputs. The vector goes through the tanh activation, and the output is the new hidden state, or the memory of the network.



![img](https://cdn-images-1.medium.com/max/800/1*WMnFSJHzOloFlJHU6fVN-g.gif)

RNN Cell

## Tanh activation

The tanh activation is used to help regulate the values flowing through the network. The tanh function squishes values to always be between -1 and 1.



![img](https://cdn-images-1.medium.com/max/800/1*iRlEg1GBKRzGTre5aOQUCg.gif)

Tanh squishes values to be between -1 and 1

When vectors are flowing through a neural network, it undergoes many transformations due to various math operations. So imagine a value that continues to be multiplied by let’s say **3**. You can see how some values can explode and become astronomical, causing other values to seem insignificant.



![img](https://cdn-images-1.medium.com/max/800/1*LgbEFcGiUpseZ--M7wuZhg.gif)

vector transformations without tanh

A tanh function ensures that the values stay between -1 and 1, thus regulating the output of the neural network. You can see how the same values from above remain between the boundaries allowed by the tanh function.



![img](https://cdn-images-1.medium.com/max/800/1*gFC2bTg3uihp1klknWU0qg.gif)

vector transformations with tanh

So that’s an RNN. It has very few operations internally but works pretty well given the right circumstances (like short sequences). RNN’s uses a lot less computational resources than it’s evolved variants, LSTM’s and GRU’s.

### LSTM

An LSTM has a similar control flow as a recurrent neural network. It processes data passing on information as it propagates forward. The differences are the operations within the LSTM’s cells.



![img](https://cdn-images-1.medium.com/max/800/1*0f8r3Vd-i4ueYND1CUrhMA.png)

LSTM Cell and It’s Operations

These operations are used to allow the LSTM to keep or forget information. Now looking at these operations can get a little overwhelming so we’ll go over this step by step.

#### Core Concept

The core concept of LSTM’s are the cell state, and it’s various gates. The cell state act as a transport highway that transfers relative information all the way down the sequence chain. You can think of it as the “memory” of the network. The cell state, in theory, can carry relevant information throughout the processing of the sequence. So even information from the earlier time steps can make it’s way to later time steps, reducing the effects of short-term memory. As the cell state goes on its journey, information get’s added or removed to the cell state via gates. The gates are different neural networks that decide which information is allowed on the cell state. The gates can learn what information is relevant to keep or forget during training.

#### Sigmoid

Gates contains sigmoid activations. A sigmoid activation is similar to the tanh activation. Instead of squishing values between -1 and 1, it squishes values between 0 and 1. That is helpful to update or forget data because any number getting multiplied by 0 is 0, causing values to disappears or be “forgotten.” Any number multiplied by 1 is the same value therefore that value stay’s the same or is “kept.” The network can learn which data is not important therefore can be forgotten or which data is important to keep.



![img](https://cdn-images-1.medium.com/max/800/1*rOFozAke2DX5BmsX2ubovw.gif)

Sigmoid squishes values to be between 0 and 1

Let’s dig a little deeper into what the various gates are doing, shall we? So we have three different gates that regulate information flow in an LSTM cell. A forget gate, input gate, and output gate.

#### Forget gate

First, we have the forget gate. This gate decides what information should be thrown away or kept. Information from the previous hidden state and information from the current input is passed through the sigmoid function. Values come out between 0 and 1. The closer to 0 means to forget, and the closer to 1 means to keep.



![img](https://cdn-images-1.medium.com/max/800/1*GjehOa513_BgpDDP6Vkw2Q.gif)

Forget gate operations

#### Input Gate

To update the cell state, we have the input gate. First, we pass the previous hidden state and current input into a sigmoid function. That decides which values will be updated by transforming the values to be between 0 and 1. 0 means not important, and 1 means important. You also pass the hidden state and current input into the tanh function to squish values between -1 and 1 to help regulate the network. Then you multiply the tanh output with the sigmoid output. The sigmoid output will decide which information is important to keep from the tanh output.



![img](https://cdn-images-1.medium.com/max/800/1*TTmYy7Sy8uUXxUXfzmoKbA.gif)

Input gate operations

#### Cell State

Now we should have enough information to calculate the cell state. First, the cell state gets pointwise multiplied by the forget vector. This has a possibility of dropping values in the cell state if it gets multiplied by values near 0. Then we take the output from the input gate and do a pointwise addition which updates the cell state to new values that the neural network finds relevant. That gives us our new cell state.



![img](https://cdn-images-1.medium.com/max/800/1*S0rXIeO_VoUVOyrYHckUWg.gif)

Calculating cell state

#### Output Gate

Last we have the output gate. The output gate decides what the next hidden state should be. Remember that the hidden state contains information on previous inputs. The hidden state is also used for predictions. First, we pass the previous hidden state and the current input into a sigmoid function. Then we pass the newly modified cell state to the tanh function. We multiply the tanh output with the sigmoid output to decide what information the hidden state should carry. The output is the hidden state. The new cell state and the new hidden is then carried over to the next time step.



![img](https://cdn-images-1.medium.com/max/800/1*VOXRGhOShoWWks6ouoDN3Q.gif)

output gate operations

To review, the Forget gate decides what is relevant to keep from prior steps. The input gate decides what information is relevant to add from the current step. The output gate determines what the next hidden state should be.

#### Code Demo

For those of you who understand better through seeing the code, here is an example using python pseudo code.



![img](https://cdn-images-1.medium.com/max/800/1*p2yXhtxmYflEUrTC1rCoUA.png)

python pseudo code

1. First, the previous hidden state and the current input get concatenated. We’ll call it combine.
2. Combine get’s fed into the forget layer. This layer removes non-relevant data.
3. A candidate layer is created using combine. The candidate holds possible values to add to the cell state.
4. Combine also get’s fed into the input layer. This layer decides what data from the candidate should be added to the new cell state.
5. After computing the forget layer, candidate layer, and the input layer, the cell state is calculated using those vectors and the previous cell state.
6. The output is then computed.
7. Pointwise multiplying the output and the new cell state gives us the new hidden state.

That’s it! The control flow of an LSTM network are a few tensor operations and a for loop. You can use the hidden states for predictions. Combining all those mechanisms, an LSTM can choose which information is relevant to remember or forget during sequence processing.

### GRU

So now we know how an LSTM work, let’s briefly look at the GRU. The GRU is the newer generation of Recurrent Neural networks and is pretty similar to an LSTM. GRU’s got rid of the cell state and used the hidden state to transfer information. It also only has two gates, a reset gate and update gate.



![img](https://cdn-images-1.medium.com/max/800/1*jhi5uOm9PvZfmxvfaCektw.png)

GRU cell and it’s gates

#### Update Gate

The update gate acts similar to the forget and input gate of an LSTM. It decides what information to throw away and what new information to add.

#### Reset Gate

The reset gate is another gate is used to decide how much past information to forget.

And that’s a GRU. GRU’s has fewer tensor operations; therefore, they are a little speedier to train then LSTM’s. There isn’t a clear winner which one is better. Researchers and engineers usually try both to determine which one works better for their use case.

### So That’s it

To sum this up, RNN’s are good for processing sequence data for predictions but suffers from short-term memory. LSTM’s and GRU’s were created as a method to mitigate short-term memory using mechanisms called gates. Gates are just neural networks that regulate the flow of information flowing through the sequence chain. LSTM’s and GRU’s are used in state of the art deep learning applications like speech recognition, speech synthesis, natural language understanding, etc.

If you’re interested in going deeper, here are links of some fantastic resources that can give you a different perspective in understanding LSTM’s and GRU’s. This post was heavily inspired by them.

[http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

<http://colah.github.io/posts/2015-08-Understanding-LSTMs/>

<https://www.youtube.com/watch?v=WCUNPb-5EYI>

I had a lot of fun making this post so let me know in the comments if this was helpful or what you would like to see in the next one. And as always, thanks for reading!
