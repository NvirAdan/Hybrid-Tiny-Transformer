# HYBRID TINY TRANSFORMER

## DESCRIPTION

A Hybrid Tiny Transformer trained with the dataset of all Shakespeare.\
**Combining** the use of the automatic gradient descent of my previous project (**Minitorch**) and **manual** back-propagation using **only NumPy** for functions like the **Multi-Head Attention**, **Positional Encoding**, **Layer Norm**, etc.


## WHY?

This project is like a "final exam" for me. Previously I made projects like the **Minitorch** which have achieved a parity with PyTorch of 1e-6 Atol and this is a good way of challenge its capacity in a **"real environment"**.Also, I made projects like the **CNN-from scratch**(CIFAR-10) or the **NN-from scratch**(MNIST) so the next logical step is to do the same thing but now with a more advanced architecture like the Transformer.\
As a project, the final objetive isn't that it becomes a "bard" but the decreasing of the loss and the acquisition of a complete control of the architecture.

>[!NOTE]
>That is the reason why it is a Hybrid Tiny Transformer, because I'll use the minitorch for the parts that I already know, like the FFN (I've learned it in the NN of MNIST), and the rest of the parts that are new to me (like the Multi-Head Attention and its Back Propagation) will be written using only NumPy.

## SAMPLINGS & METRICS

The Model has been trained for **46 seconds** with a total of **5000** iterations reaching a peak velocity of **108.45 iterations per second**.\
In the following images you can see the progress of the training of the model at the same time as the sampling of each checkpoint saved, letting you see the quality of the predictions and how they get better through the training. 


![FIRST TRAIN](./assets/firsttrainingpart1.png)
*First screenshot of the output cell of the training function*

![FIRST TRAIN](./assets/firsttrainingpart2.png)
*Second screenshot of the output cell of the training function*


In the time training, the loss has decreased from an entropy of **4.5** (total chaos) to a loss close to **2.1**, showing clear signs of improvement.\
The graph shows a noisy training (because of the difference of the batches) but with an overall tendency to decrease.

![Loss Curve](./assets/losscurve.png)
*Raw loss curve showing a decrease in the loss of the model*


In the next image the number of tokens generated is expanded, from a previous 30 tokens(checkpoint samplings) to a 500 tokens generation to show its ability to generate longer text after the training.

![SHOWCASE](./assets/Showcasefirsttraining1.png)
*Text showcasing the model trying to predict the correct next token after the training*



## FEATURES

- Hybrid Architecture with a Custom Engine.
- Manual Back-Propagation.
- High-Efficiency Training(**CPU Only**).
- Memory-Lean Design.
- Learned-Positional Encoding.
- Checkpoint System.
- Numerical Stability.
- Automated Data Pipeline.


## ARCHITECTURE 🔶

The **Hybrid Tiny Transformer** implements an architecture that splits the data flow and the responsabilities between an autograd engine and manual mathematical operations.

### The Hybrid Engine  

#### Minitorch(Autograd Engine): 
Manages the Dense Layers(**FFN**) and gradient computation for all the trainables weights.
#### Manual Numpy Ops
Implements the **Multi-Head Attention** and **LayerNorm** logic. The backpropagation for these components was derived and implemented from scratch.

### Block Diagram & Data Flow

#### Token & Positional Encoding: 
Tokens are projected into vector and fused with a learned parameter to encode sequence order.
#### Transformer Block:
- **Layer Norm**(Manual): Normalize activations to stabilize the training process.
- **Multi-Head Attention**(Manual): Compute the relevance of different tokens in the context window.
- **Layer Norm**(Manual): Another Layer Norm because I wanted to see what happens if I use two and test it.
- **Feed-Foward Network**(Minitorch): Processes the second layer norm output through non-linear transformations.
#### Cross Entropy Loss:
Use of a **Custom Loss Function** to be able to process the inputs of NumPy Tensors(if the model isn't taining) and the inputs of Minitorch's Tensors(if the model is in training phase).

### Hyperparameters (The "Tiny" Config)
- **Batch size**: 4
- **Block Size & Context Windows**: 8
- **Number of Embeddings**: 128
- **Number of Heads**: 4
- **Vocabulary Size**: ~65 Characters
- **Optimizer**: Adam (integrated within the MiniTorch Engine)

>[!NOTE]
> Unlike standard black-boxes the **Scaled Dot-Product** and the **Causal Triangular Mask** were implemented bit-by-bit in NumPy.Preventing the model to be able to cheat seeing the future tokens.


## PROBLEMS & SOLUTIONS



## HOW TO RUN
