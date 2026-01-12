import { Question } from '../types';

export const RAW_DATA = `
# Advanced Machine Learning

Collection of questions asked during the course, which can have some mistakes.

## Other questions from previous years

### Question 1  
Which of these AI systems is based on hard-coded rules?

- A) Deep Blue
- B) AlphaGo
- C) ADALINE

**Correct Answer:**  **A) Deep Blue**

---

### Question 2  
What has been enabled by Deep Learning?

- A) Multi-layer models
- B) Automatic mapping between input and output
- C) Representation learning

**Correct Answer:**  **C) Representation learning**

---

### Question 3  
Which of these statements is wrong?

- A) Many factors of variation influence every single piece of data
- B) Factor of variations is directly observable
- C) Factor of variation can correspond to abstract features

**Correct Answer:**  **B) Factor of variations is directly observable**

---

### Question 4  
ADALINE (flag the correct answer)

- A) Used continuous predicted values to learn
- B) Was a non-linear model
- C) Had weights that needed to be set by an operator

**Correct Answer:**  **A) Used continuous predicted values to learn**

---

### Question 5  
Which of these statements is correct?

- A) Deep learning is primarily concerned with building more accurate models of how the brain actually works
- B) Neuroscience was the inspiring science for neural networks
- C) Connectionism arose in the context of cognitive science
- D) One key concept of connectionism is the distributed representation
- E) In deep learning each concept is represented by one neuron
- F) The backpropagation is based on the concept of semantic similarity

**Correct Answers:**

- **B)** **Neuroscience was the inspiring science for neural networks.**
- **C)** **Connectionism arose in the context of cognitive science.**
- **D)** **One key concept of connectionism is the distributed representation.**

---

### Question 6  
Which among these factors did enable Deep Learning?

- A) Availability of large datasets
- B) Network connectivity
- C) The implementation of backpropagation
- D) Distributed representation
- E) Availability of faster CPUs

**Correct Answers:** 
- **A)** **Availability of large datasets**,
- **C)** **The implementation of backpropagation**,
- **D)** **Distributed representation**.

---

### Question 7  
Algorithms for weight adjustment in a classification network are made to linearly separate the points belonging to the 2 classes

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 8  
Weight adjustments are aimed at learning a good separation function

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 9  
Recurrent neural networks..

- A) In RNN, the information flow only moves in one direction
- B) Are suitable for learning patterns in image data
- C) Capture sequential information through loop connections

**Correct Answer:**  **C) Capture sequential information through loop connections**

---

### Question 10  
The behavior of intermediate layers in a FFN are simply specified by the training data

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 11  
Multilayer networks need to specify the kernel function

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 12  
Kernel functions first map features in a different space and then evaluate the inner product

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 13  
In general, kernel machines suffer from the high computational cost of training when the dataset is large

- A) Yes, because their complexity is non linear in the number of training examples
- B) No, because their complexity is linear in the number of training examples
- C) No, because they don't depend on the number of training examples

**Correct Answer:**  **A) Yes, because their complexity is non linear in the number of training examples**

---

### Question 14  
Which of these sentences is true?

- A) The gradient descent algorithm may not converge if the learning rate is too big.
- B) The gradient descent algorithm may not converge if the learning rate is too small.
- C) The gradient descent algorithm always converges to the global optimum
- D) The gradient descent algorithm may converge to a local optimum

**Correct Answers:**  

- **A)** **The gradient descent algorithm may not converge if the learning rate is too big**,
- **D)** **The gradient descent algorithm may converge to a local optimum**

---

### Question 15  
To apply an iterative numerical optimization procedure for learning the weight of a FFN

- A) The cost function may be a function which we cannot evaluate analytically
- B) We need to know the analytical form of the gradient
- C) It is enough to have some way of approximating the gradient

**Correct Answer:**  **C) It is enough to have some way of approximating the gradient**

---

### Question 16  
For training a multilayer NN

- A) We must always adjust the weight of all the layers in one go
- B) We can train one layer at a time using the error made by each layer in predicting the final result
- C) We can train one layer at a time using the error made in reproducing its own input

**Correct Answer:**  **C) We can train one layer at a time using the error made in reproducing its own input**

---

### Question 17  
A SVM can be trained by solving a system of equations while a neural network can always be trained by using convex optimization

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 18  
It is always possible to train a neural network by solving a system of equations?

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 19  
Sparse representations seem to be more beneficial than dense representations

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 20  
In a neural network, the nonlinearity causes the most interesting loss function to become non-convex

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 21  
The loss function produces a numerical score that also depends on the set of parameters which characterizes the FFN model

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 22  
The gradient can be estimated using a sample of training examples because it is an expectation

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 23  
Regularization functions are added to the loss function to reduce their training error

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 24  
The sigmoid function

- A) Saturates for large argument values
- B) Has a sensitive gradient when z is close to zero
- C) Has a zero gradient when the argument is close to zero
- D) Has a large gradient when it reaches saturation
- E) Is 0 for large negative argument values
- F) In some cases it can produce a sparse network (many zero weights) that may be useful

**Correct Answers:**  

- **A) Saturates for large argument values**,
- **B) Has a sensitive gradient when z is close to zero**,
- **E) Is 0 for large negative argument values**

---

### Question 25  
Rectified linear unit (ReLU) is proposed to speed up the learning convergence

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 26  
Advantages of the ReLU functions are

- A) ReLUs are much simpler computationally
- B) Reduced likelihood of the gradient to vanish
- C) The gradient is constant for z > 0
- D) Differentiability

**Correct Answers:**  
- **A) ReLUs are much simpler computationally**,
- **B) Reduced likelihood of the gradient to vanish**
- **C) The gradient is constant for z > 0**

---

### Question 27  
Leaky ReLUs

- A) Saturate when the input is less than 0
- B) Need to perform exponential operations
- C) Tend to blow up when Z is large

**Correct Answer:**  ** C) Tend to blow up when Z is large**

---

### Question 28  
Maxout

- A) Has as special cases ReLU and Leaky ReLU
- B) Requires fewer parameters to be learned
- C) Does not have the problem of saturation
- D) Can approximate any convex function

**Correct Answers:**  
- **A) Has as special cases ReLU and Leaky ReLU**,
- **C) Does not have the problem of saturation**,
- **D) Can approximate any convex function**

---

### Question 29  
Weights in a network must be initialized

- A) At zero
- B) By maintaining symmetry
- C) With zero variance
- D) Randomly
- E) It is indifferent

**Correct Answer:**  **D) Randomly**

---

### Question 30  
Which of the following statements are true?

- A) When using SGD with mini-batches the model updates do not depend on the number of training examples
- B) When using SGD with mini-batches the number of updates to reach convergence does not depend on the number of training examples
- C) The choice of cost functions is tightly coupled with the choice of the output unit

**Correct Answers:** 
- **B) When using SGD with mini-batches the number of updates to reach convergence does not depend on the number of training examples**,
- **C) The choice of cost functions is tightly coupled with the choice of the output unit**

---

### Question 31  
The function max{0, min{1, Wh + b}} is a good choice as an output function for classification problems

- A) No because it does not return a value between 0 and 1
- B) Yes, because it returns a value between 0 and 1
- C) Yes, because it is linear
- D) No, because it is not good for training

**Correct Answer:**  **D) No, because it is not good for training**

---

### Question 32  
Softmax function

- A) Is a good choice for representing discrete probability distributions with n possible values
- B) It is a good output function because it is continuous and differentiable
- C) Since its output is a probability distribution it can always be interpreted as a confidence level
- D) If the prediction is correct the penalty is always 0

**Correct Answer:**  
**A) Is a good choice for representing discrete probability distributions with n possible values**

---

### Question 33
A Gaussian mixture output function

- A) Can represent multimodal functions
- B) The weight associated to a Gaussian in the mixture represents the probability of the output
- C) Mixtures are particularly suitable for generative models for speech or for movements of objects

**Correct Answers:**  **All seem to be true**

---

### Question 34  
Regularization

- A) It reduces the validation/test error at the expense of (acceptable) training error
- B) Enables the model to reach a point that does minimize the loss function
- C) It enables the model to reduce the variability of the data

**Correct Answer:** 
- **A) It reduces the validation/test error at the expense of (acceptable) training error**,

---

### Question 35  
Which of these sentences are true?

- A) Simpler models generalize better
- B) Multiple hypothesis (ensemble) models generalize better
- C) More complex models can represent the true data generating process
- D) In general, when building machine learning models, the data generating process is not known

**Correct Answers:** 
- **A) Simpler models generalize better**,
- **B) Multiple hypothesis (ensemble) models generalize better**,
- **D) In general, when building machine learning models, the data generating process is not known**

---

### Question 36  
Regularizing estimators

- A) Reduce bias
- B) Reduce the gap between training error and validation error
- C) Reduce underfitting problems
- D) Can reduce the complexity of the model

**Correct Answers:**  
- **B) Reduce the gap between training error and validation error**,
- **D) Can reduce the complexity of the model**

---

### Question 37  
Which of these sentences are false?

- A) If the weight of the regularization term in the loss function is too high it may imply underfitting
- B) If the weight of the penalization term in the loss function is too high it may imply overfitting
- C) Regularizing the bias parameters can introduce a significant amount of underfitting
- D) Regularizing the bias parameters can introduce a significant amount of overfitting
- E) Usually the bias parameters are not constrained by regularizing constraints

**Correct Answers:**  
- **B) If the weight of the penalization term in the loss function is too high it may imply overfitting**,
- **D) Regularizing the bias parameters can introduce a significant amount of overfitting**

---

### Question 38  
Parameter norm penalties

- A) Make the network more stable
- B) Minor variation or statistical noise on the inputs will result in large differences in the output
- C) Encourage the network toward using small weights

**Correct Answers:**  
- **A) Make the network more stable**,
- **C) Encourage the network toward using small weights**

---

### Question 39  
Consider norm penalizations

- A) Sum of absolute weights penalizes small weights more
- B) Squared weights penalize large values more
- C) L2 results in more sparse weights than L1
- D) The addition of the L2 term modifies the learning rule by shrinking the weight factor by a constant factor on each parameter update
- E) L2 rescales the weights along the axes defined by the eigenvectors of the Hessian matrix

**Correct Answers:** 
- **B) Squared weights penalize large values more,**
- **D) The addition of the L2 term modifies the learning rule by shrinking the weight factor by a constant factor on each parameter update**

---

### Question 40  
Which of these sentences are true?

- A) Regularizing operators can be seen as soft constraints of the learning optimization problem
- B) Regularizing operators can be done by optimizing with respect to the loss function and then re-projecting the solution in the feasible region $(k\Omega(\theta)) < 0$
- C) Explicit constraints implemented by re-projection do not necessarily encourage the weights to approach the origin
- D) Explicit constraints implemented by re-projection only have an effect when the weights become large and attempt to leave the constraint region

**Correct Answers:** 
- **A) Regularizing operators can be seen as soft constraints of the learning optimization problem**,
- **C) Explicit constraints implemented by re-projection do not necessarily encourage the weights to approach the origin**,
- **D) Explicit constraints implemented by re-projection only have an effect when the weights become large and attempt to leave the constraint region**

---

### Question 41  
Dataset augmentation

- A) Creates fake data and adds it to the training set
- B) It is very effective for non-supervised tasks
- C) Injecting noise in the input to a neural network can also be seen as a form of data augmentation

**Correct Answers:** 
- **A) Creates fake data and adds it to the training set**,
- **C) Injecting noise in the input to a neural network can also be seen as a form of data augmentation**

---

### Question 42  
Which of these sentences is true?

- A) Label smoothing is used for solving regression tasks
- B) Label smoothing makes models robust to possible errors in the training set
- C) Label smoothing can help convergence of maximum likelihood learning with a softmax classifier and hard targets

**Correct Answers:** 
- **B) Label smoothing makes models robust to possible errors in the training set**,
- **C) Label smoothing can help convergence of maximum likelihood learning with a softmax classifier and hard targets**

---

### Question 43  
Which of these sentences is false?

- A) Multitask forces to share a set of parameters across different tasks
- B) Multitask improves generalization when tasks are very different

**Correct Answer:**  **B) Multitask improves generalization when tasks are very different**

---

### Question 44  
Which of the following sentences are true?

- A) Parameter tying imposes a subset of parameters to be equal
- B) Early stopping is a form of regularization
- C) Bagging is a form of ensemble model
- D) Bagging is more effective if the output of the models learned are correlated
- E) Dropout is generally coupled with mini-batch based learning algorithms

**Correct Answers:**  
- **B) Early stopping is a form of regularization**
- **C) Bagging is a form of ensemble model**
- **E) Dropout is generally coupled with mini-batch based learning algorithms**

---

### Question 45  
In machine learning, the cost function to minimize during the training process is the performance measure P representing the number of correct classifications on the test set

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 46  
The final aim of Machine Learning is

- A) The minimization of the true risk function
- B) The minimization of the empirical risk using a surrogate loss function

**Correct Answer:**  **B) The minimization of the empirical risk using a surrogate loss function**

---

### Question 47  
A surrogate loss function

- A) Acts as a proxy to the true risk being "nice" enough to be optimized efficiently
- B) Acts as a proxy to empirical risk being "nice" enough to be optimized efficiently

**Correct Answer:**  **A) Acts as a proxy to the true risk being "nice" enough to be optimized efficiently**

---

### Question 48  
Early stopping halt criterion

- A) Is typically based on the performance obtained on a validation set
- B) Is typically based on the performance obtained on the training set

**Correct Answer:**  **A) Is typically based on the performance obtained on a validation set**

---

### Question 49  
The accuracy of the estimated mean of the gradient

- A) It depends on the number of samples used
- B) Has a standard error which decreases linearly with the number of samples used
- C) Has a standard error which decreases less than linearly with the number of samples

**Correct Answers:**  
- **A) It depends on the number of samples used**,
- **C) Has a standard error which decreases less than linearly with the number of samples**

---

### Question 50  
Ill conditioning of the Hessian matrix of the cost function

- A) Can be partly overcome by using the momentum strategy
- B) Can prevent the gradient from arriving at a critical point
- C) Can imply that very small steps are needed to decrease the cost function

**Correct Answers:**  **A) and C)**

---

### Question 51  
Local minima in deep learning problems

- A) Are rare
- B) Are more common than saddle points
- C) Are much more likely to have a low cost than a high cost

**Correct Answer:**  **A) Are rare**

---

### Question 52  
Neural networks with many layers

- A) Often have extremely steep regions (cliffs)
- B) Often have flat regions
- C) Often have many local minima of similar cost

**Correct Answer:**  **All true**

---

### Question 53  
Momentum update rule

- A) Accumulates previous values of the cost function
- B) Can be incorporated in SGD
- C) Its step size is larger if previous gradients point in the same direction

**Correct Answers:** 
- **B) Can be incorporated in SGD**,
- **C) Its step size is larger if previous gradients point in the same direction**

---

### Question 54  
AdaGrad algorithm

- A) Takes into account previous squared gradients
- B) Decreases the learning rate too much in the early stages
- C) Uses the same learning rate for all parameters

**Correct Answers:** 
- **A) Takes into account previous squared gradients**,
- **B) Decreases the learning rate too much in the early stages**

---

### Question 55  
RMSProp

- A) It takes into account the square gradients
- B) It is a modification of the AdaGrad algorithm
- C) It does not require hyperparameters

**Correct Answer:**  
- **A) It takes into account the square gradients**,
- **B) It is a modification of the AdaGrad algorithm**

---

### Question 56  
Adam algorithm

- A) It also takes into account the curvature of the cost function through the second order derivatives
- B) It uses the momentum strategy
- C) It is based on RMSProp

**Correct Answer:** 
- **B) It uses the momentum strategy**,
- **C) It is based on RMSProp**

---

### Question 57  
A good initialization procedure

- A) Assigns large weights
- B) Assigns extremely small weights
- C) None of the two

**Correct Answer:**  **C) None of the two**

---

### Question 58  
In initialization

- A) Larger weights break symmetry more
- B) Smaller weights propagate information more efficiently
- C) Large weights make the model more likely to reach solutions with good generalization property
- D) Small weights make the model more robust

**Correct Answer:**
- **A) Larger weights break symmetry more**,
- **D) Small weights make the model more robust**

---

## Second part of the course

### Question 1  
Convolution is

- A) Local in space, local in depth
- B) Local in space, full in depth
- C) Full in space, local in depth
- D) Full in space, full in depth

**Correct Answer:**  **B) Local in space, full in depth**

---

### Question 2  
Given the input volume with size H * W * K and a filter bank with size h * w * k we want to convolve them. The size of the output volume will be:

- A) H * W * K
- B) h * w * k
- C) (H - h + 1) * (W - w + 1) * k
- D) (H - h - 1) * (W - w - 1) * k
- E) (H - h + 1) * (W - w + 1) * 1
- F) (H - h - 1) * (W - w - 1) * 1

**Correct Answer:**  **E) (H - h + 1) * (W - w + 1) * 1**

---

### Question 3  
How many channels (i.e. depth size) will have the output volume resulting from the convolution of an input volume with 16 channels (i.e. depth=16) with a filter bank of 16 filters?

- A) 1
- B) 16
- C) 32

**Correct Answer:**  **B) 16**

---

### Question 4 
Compute the output after the application of max pooling to the following input volume IN with a neighborhood of size=2 and stride=2. IN = [12 23 40 31] [11 15 42 52]

- A) [40; 52]
- B) [23 42 52] 
- C) [23 52]

**Correct Answer:**  **C) [23 52]**

---

### Question 5  
Which are common techniques to reduce overfitting?

- A) Weight decay
- B) Local response normalization
- C) Data augmentation
- D) Data normalization
- E) Dropout

**Correct Answers:**  **A) Weight decay, C) Data augmentation, E) Dropout**

---

### Question 6  
In data augmentation we have seen different policies (e.g. cropping, rotation, color cast, vignetting...)

- A) All policies are safe to use to any problem
- B) Only a subset of policies is safe to be used for each problem

**Correct Answer:**  **B) Only a subset of policies is safe to be used for each problem**

---

### Question 7  
We can diagnose the training and understand if we are overfitting:

- A) By plotting the loss on the training set across epochs
- B) By plotting the loss on the validation set across epochs
- C) By plotting the loss on the test set across epochs
- D) By plotting the accuracy on the training and validation sets across epochs

**Correct Answer:**  **D) By plotting the accuracy on the training and validation sets across epochs**

---

### Question 8  
We can continue adding layers to a NN and we will continue to obtain better results

- A) Yes
- B) No

**Correct Answer:**  **B) No**

---

### Question 9  
GoogLeNet (i.e. Inception-v1) introduced the use of auxiliary classifiers:

- A) To mitigate the problem of vanishing gradients
- B) To perform multi-task classification
- C) To reduce overfitting

**Correct Answer:**  **A) To mitigate the problem of vanishing gradients**

---

### Question 10  
ResNets were able to train a model with 150+ layers by:

- A) Using just one fully-connected layer
- B) Introducing the residual connections
- C) Using just 3x3 convolutional filters

**Correct Answer:**  **B) Introducing the residual connections**

---

### Question 11  
If we have few data:

- A) We cannot use Deep Learning
- B) We can still use deep learning

**Correct Answer:**  **B) We can still use deep learning**

---

### Question 12  
From which layer we can extract activations to be used as features to classify data for a new small dataset that is similar to the dataset used to pre-train the whole network?

- A) Conv1
- B) Conv2
- C) Conv3
- D) Conv4
- E) Conv5
- F) FC6
- G) FC7
- H) FC8

**Correct Answer:**  **F) FC6, G) FC7, H) FC8**

---

### Question 13  
Model compression is only used to allow models to run on mobile devices:

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 14  
Which is not a model compression technique?

- A) Weight sharing
- B) Network pruning
- C) Low rank matrix decomposition
- D) Dropout
- E) Knowledge distillation
- F) Quantization

**Correct Answer:**  **D) Dropout**

---

### Question 15  
Magnitude weight pruning removes the weights having:

- A) The lowest value
- B) The lowest absolute value
- C) The highest absolute value
- D) The highest value

**Correct Answer:**  **B) The lowest absolute value**

---

### Question 16  
Structured pruning

- A) Aims to preserve network density for computational efficiency
- B) Aims to increase network sparsity for computational efficiency

**Correct Answer:**  **A) Aims to preserve network density for computational efficiency**

---

### Question 17  
Low rank matrix decomposition is particularly useful in:

- A) Convolutional layers
- B) Pooling layers
- C) Fully connected layers

**Correct Answer:**  **C) Fully connected layers**

---

### Question 18
Global MBP tends to outperform layer-wise MBP:

- A) True
- B) False
- 
**Correct Answer:**  **A) True**

---

### Question 19  
RNNs are a family of neural networks for processing sequential data:

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 20  
The computation in most RNNs can be decomposed in 3 blocks of parameters and associated transformations:

- A) From the input to the hidden state
- B) From the hidden state to the input
- C) From the previous hidden state to the next hidden state
- D) From the next hidden state to the previous hidden state
- E) From the hidden state to the output
- F) From the output to the hidden state

**Correct Answers:**  
- **A) From the input to the hidden state**,
- **C) From the previous hidden state to the next hidden state**,
- **E) From the hidden state to the output**

---

### Question 21  
What is the name of the algorithm used to train RNNs?

- A) Backpropagation
- B) Backpropagation through recurrency
- C) Backpropagation through time

**Correct Answer:**  **C) Backpropagation through time**

---

### Question 22  
Vanishing gradients are more easy to identify than exploding gradients:

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 23 
Exploding gradients are more difficult to handle than vanishing gradients:

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 24  
Vanishing gradients

- A) Bias the parameters to capture short-term dependencies
- B) Bias the parameters to capture long-term dependencies

**Correct Answer:**  **A) Bias the parameters to capture short-term dependencies**

---

### Question 25  
Gated RNNs are based on the idea of creating paths through time that have derivatives that neither vanish nor explode:

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 26  
LSTMs have the following gates:

- A) Input gate
- B) Remember gate
- C) Recurrent gate
- D) Forget gate
- E) Output gate
- F) Hidden gate

**Correct Answer:**  **A) Input gate, D) Forget gate, E) Output gate**

---

### Question 27  
GRUs have:

- A) Significantly less parameters than LSTMs
- B) Significantly more parameters than LSTMs

**Correct Answer:**  **A) Significantly less parameters than LSTMs**

---

### Question 28  
Federated learning aims to:

- A) Collaboratively train a ML model
- B) Independently train a ML model

**Correct Answer:**  **A) Collaboratively train a ML model**

---

### Question 29  
In federated learning, the data:

- A) Is shared across parties/server
- B) Is kept private

**Correct Answer:**  **B) Is kept private**

---

### Question 30  
In federated learning:

- A) We control how the data is distributed across parties/workers
- B) Data in each party/worker is not independent and identically distributed

**Correct Answer:**  **B) Data in each party/worker is not independent and identically distributed**

---

### Question 31  
In federated learning there is always a server to orchestrate the training:

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 32
In the FedAVG algorithm the central model is updated:

- A) Taking the minimum value of the parameters in the corresponding layers across the models sent by the different workers
- B) Taking the mean value of the parameters in the corresponding layers across the models sent by the different workers
- C) Taking the median value of the parameters in the corrisponding layers across the models sent by the different workers
- D) Taking the maximum value of the parameters in the corrisponding layers across the models sent by the different workers

**Correct Answer:**  **B) Taking the mean value of the parameters in the corresponding layers across the models sent by the different workers**

---

### Question 33
Transformers can process sequences of arbitrary length

- A) True
- B) False

**Correct Answer:**  **A) True**
---

### Question 34
Transformers process each input independently

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 35
The only processing module in the transformer layer is self attention
- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 36
Self attention, being a composition of two linear transformations is linear

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 37
The elements that are needed for the computation of self attention are

- A) Outputs
- B) Inputs
- C) Values
- D) Variables
- E) Queries
- F) Questions
- G) Keys
- H) Chains

**Correct Answer:**  **C) Values, E) Queries, G) Keys**

---

### Question 38
A trasformer head is completely defined by 3 weight matrices and 3 biases

- A) True
- B) False

**Correct Answer:**  **A) True**

---

### Question 39
To obtain the best performance usually transformers use one single head
- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 40
Self-Supervised Learning refers to learning methods in which the models are trained:
- A) with supervisory signals that are generated from the data itself by leveraging its structure
- B) with supervisory signals provided by human annotated labels
- C) without the need of any supervisory signal

**Correct Answer:**  **A) with supervisory signals that are generated from the data itself by leveraging its structure**

---
### Question 41
How are called the labels automatically created for the pretext task?
- A) Pretext labels
- B) Pseudo labels
- C) Proxy labels
- D) Generated labels

**Correct Answer:**  **B) Pseudo labels**

---

### Question 42
Image generation cannot be used as a pretext task
- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 43
When we use image generation as the pretext task, after training we are interested to use
- A) The generator
- B) The discriminator
- C) Both

**Correct Answer:**  **B) The discriminator**

---

### Question 44
The performance of SSL models are usually measured comparing their accuracy on the pretext task

- A) True
- B) False

**Correct Answer:**  **B) False**

---

### Question 45
Pretext tasks cannot expolit multimodal properties of the data
- A) True
- B) False

**Correct Answer:**  **B) False**
`;

export function parseQuestions(markdown: string): Question[] {
  const lines = markdown.split('\n');
  const questions: Question[] = [];
  
  let currentCategory = "General";
  let currentQuestion: Partial<Question> | null = null;
  let buffer: string[] = [];

  // Helper to map letter A,B,C to index 0,1,2
  const charToIndex = (char: string) => {
    const code = char.toUpperCase().charCodeAt(0);
    return code - 65;
  };

  const flushQuestion = () => {
    if (currentQuestion && currentQuestion.text) {
        // Clean up text
        currentQuestion.text = currentQuestion.text.trim();
        if(currentQuestion.options && currentQuestion.correctIndices && currentQuestion.options.length > 0) {
             questions.push(currentQuestion as Question);
        }
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Headers
    if (line.startsWith('## ')) {
      currentCategory = line.replace('## ', '').trim();
      continue;
    }

    // Start of a question
    if (line.startsWith('### Question')) {
      flushQuestion();
      currentQuestion = {
        id: `q-${questions.length + 1}`,
        category: currentCategory,
        text: '',
        options: [],
        correctIndices: []
      };
      buffer = [];
      continue;
    }

    if (!currentQuestion) continue;

    // Options
    if (line.match(/^-\s[A-Z]\)/)) {
      // It's an option. 
      // First, if we have buffer text, that's the question body (if not set yet)
      if (!currentQuestion.text && buffer.length > 0) {
        currentQuestion.text = buffer.join(' ').trim();
      }
      currentQuestion.options?.push(line.substring(2).trim()); // Remove "- "
      continue;
    }

    // Correct Answer parsing
    if (line.startsWith('**Correct Answer')) {
        // Handle "All seem to be true" or "All true"
        if (line.toLowerCase().includes('all seem to be true') || line.toLowerCase().includes('all true')) {
             if(currentQuestion.options) {
                 currentQuestion.correctIndices = currentQuestion.options.map((_, idx) => idx);
             }
        } else {
             // Look for A), B), etc.
            const matches = line.matchAll(/([A-Z])\)/g);
            for (const match of matches) {
                currentQuestion.correctIndices?.push(charToIndex(match[1]));
            }
        }
       continue;
    }

    // Normal text line
    if (line !== '---' && line.length > 0) {
        // If we haven't hit options yet, it's question text
        if (currentQuestion.options?.length === 0) {
             buffer.push(line);
        }
    }
  }

  flushQuestion(); // Flush last one
  return questions;
}