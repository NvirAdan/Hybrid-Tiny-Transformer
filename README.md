# HYBRID TINY TRANSFORMER

## DESCRIPTION

A Hybrid Tiny Transformer trained with the dataset of all Shakespeare.
**Combining** the use of the automatic gradient descent of my previous project (**Minitorch**) and **manual** back-propagation using **only NumPy** for functions like the **Multi-Head Attention**, **Positional Encoding**, **Layer Norm**, etc.


## WHY?

This project is like a "final exam" for me. Previously I made projects like the **Minitorch** which have achieved a parity with PyTorch of 1e-6 Atol and this is a good way of challenge its capacity in a **"real environment"**.Also, I made projects like the **CNN-from scratch**(CIFAR-10) or the **NN-from scratch**(MNIST) so the next logical step is to do the same thing but now with a more advanced architecture like the Transformer.
As a project, the final objetive isn't that it becomes a "bard" but the decreasing of the loss and acquire the complete control of the architecture.

>[!NOTE]
>That is the reason why it is a Hybrid Tiny Transformer, because I'll use the minitorch for the parts that I already know, like the FFN (I've learned it in the NN of MNIST), and the rest of the parts that are new to me (like the Multi-Head Attention and its Back Propagation) will be written using only NumPy.

## SAMPLINGS & METRICS

The model has been trained for **46 seconds** 


![FIRST TRAIN](assets/firsttrainingpart1.png)
*First screenshot of the output cell of the training function*

![FIRST TRAIN](assets/firsttrainingpart2.png)
*Second screenshot of the output cell of the training function*

![Loss Curve](assets/losscurve.png)
*Raw loss curve showing a decrease in the loss of the model*

![SHOWCASE](assets/Showcasefirsttraining1.png)
*Text showcasing the model trying to predict the correct next token after the training*



## FEATURES


## ARCHITECTURE


## PROBLEMS & SOLUTIONS


## HOW TO RUN