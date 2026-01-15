import { Question } from './types';

export const questions: Question[] = [
  {
    "id": "q-1",
    "category": "Other questions from previous years",
    "text": "Which of these AI systems is based on hard-coded rules?",
    "options": ["Deep Blue", "AlphaGo", "ADALINE"],
    "correctIndices": [0]
  },
  {
    "id": "q-2",
    "category": "Other questions from previous years",
    "text": "What has been enabled by Deep Learning?",
    "options": ["Multi-layer models", "Automatic mapping between input and output", "Representation learning"],
    "correctIndices": [2]
  },
  {
    "id": "q-3",
    "category": "Other questions from previous years",
    "text": "Which of these statements is wrong?",
    "options": ["Many factors of variation influence every single piece of data", "Factor of variations is directly observable", "Factor of variation can correspond to abstract features"],
    "correctIndices": [1]
  },
  {
    "id": "q-4",
    "category": "Other questions from previous years",
    "text": "ADALINE (flag the correct answer)",
    "options": ["Used continuous predicted values to learn", "Was a non-linear model", "Had weights that needed to be set by an operator"],
    "correctIndices": [0]
  },
  {
    "id": "q-5",
    "category": "Other questions from previous years",
    "text": "Which of these statements is correct?",
    "options": [
      "Deep learning is primarily concerned with building more accurate models of how the brain actually works",
      "Neuroscience was the inspiring science for neural networks",
      "Connectionism arose in the context of cognitive science",
      "One key concept of connectionism is the distributed representation",
      "In deep learning each concept is represented by one neuron",
      "The backpropagation is based on the concept of semantic similarity"
    ],
    "correctIndices": [1, 2, 3]
  },
  {
    "id": "q-6",
    "category": "Other questions from previous years",
    "text": "Which among these factors did enable Deep Learning?",
    "options": [
      "Availability of large datasets",
      "Network connectivity",
      "The implementation of backpropagation",
      "Distributed representation",
      "Availability of faster CPUs"
    ],
    "correctIndices": [0, 2, 3]
  },
  {
    "id": "q-7",
    "category": "Other questions from previous years",
    "text": "Algorithms for weight adjustment in a classification network are made to linearly separate the points belonging to the 2 classes",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-8",
    "category": "Other questions from previous years",
    "text": "Weight adjustments are aimed at learning a good separation function",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-9",
    "category": "Other questions from previous years",
    "text": "Recurrent neural networks..",
    "options": [
      "In RNN, the information flow only moves in one direction",
      "Are suitable for learning patterns in image data",
      "Capture sequential information through loop connections"
    ],
    "correctIndices": [2]
  },
  {
    "id": "q-10",
    "category": "Other questions from previous years",
    "text": "The behavior of intermediate layers in a FFN are simply specified by the training data",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-11",
    "category": "Other questions from previous years",
    "text": "Multilayer networks need to specify the kernel function",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-12",
    "category": "Other questions from previous years",
    "text": "Kernel functions first map features in a different space and then evaluate the inner product",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-13",
    "category": "Other questions from previous years",
    "text": "In general, kernel machines suffer from the high computational cost of training when the dataset is large",
    "options": [
      "Yes, because their complexity is non linear in the number of training examples",
      "No, because their complexity is linear in the number of training examples",
      "No, because they don't depend on the number of training examples"
    ],
    "correctIndices": [0]
  },
  {
    "id": "q-14",
    "category": "Other questions from previous years",
    "text": "Which of these sentences is true?",
    "options": [
      "The gradient descent algorithm may not converge if the learning rate is too big.",
      "The gradient descent algorithm may not converge if the learning rate is too small.",
      "The gradient descent algorithm always converges to the global optimum",
      "The gradient descent algorithm may converge to a local optimum"
    ],
    "correctIndices": [0, 3]
  },
  {
    "id": "q-15",
    "category": "Other questions from previous years",
    "text": "To apply an iterative numerical optimization procedure for learning the weight of a FFN",
    "options": [
      "The cost function may be a function which we cannot evaluate analytically",
      "We need to know the analytical form of the gradient",
      "It is enough to have some way of approximating the gradient"
    ],
    "correctIndices": [2]
  },
  {
    "id": "q-16",
    "category": "Other questions from previous years",
    "text": "For training a multilayer NN",
    "options": [
      "We must always adjust the weight of all the layers in one go",
      "We can train one layer at a time using the error made by each layer in predicting the final result",
      "We can train one layer at a time using the error made in reproducing its own input"
    ],
    "correctIndices": [2]
  },
  {
    "id": "q-17",
    "category": "Other questions from previous years",
    "text": "A SVM can be trained by solving a system of equations while a neural network can always be trained by using convex optimization",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-18",
    "category": "Other questions from previous years",
    "text": "It is always possible to train a neural network by solving a system of equations?",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-19",
    "category": "Other questions from previous years",
    "text": "Sparse representations seem to be more beneficial than dense representations",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-20",
    "category": "Other questions from previous years",
    "text": "In a neural network, the nonlinearity causes the most interesting loss function to become non-convex",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-21",
    "category": "Other questions from previous years",
    "text": "The loss function produces a numerical score that also depends on the set of parameters which characterizes the FFN model",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-22",
    "category": "Other questions from previous years",
    "text": "The gradient can be estimated using a sample of training examples because it is an expectation",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-23",
    "category": "Other questions from previous years",
    "text": "Regularization functions are added to the loss function to reduce their training error",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-24",
    "category": "Other questions from previous years",
    "text": "The sigmoid function",
    "options": [
      "Saturates for large argument values",
      "Has a sensitive gradient when z is close to zero",
      "Has a zero gradient when the argument is close to zero",
      "Has a large gradient when it reaches saturation",
      "Is 0 for large negative argument values",
      "In some cases it can produce a sparse network (many zero weights) that may be useful"
    ],
    "correctIndices": [0, 1, 4]
  },
  {
    "id": "q-25",
    "category": "Other questions from previous years",
    "text": "Rectified linear unit (ReLU) is proposed to speed up the learning convergence",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-26",
    "category": "Other questions from previous years",
    "text": "Advantages of the ReLU functions are",
    "options": [
      "ReLUs are much simpler computationally",
      "Reduced likelihood of the gradient to vanish",
      "The gradient is constant for z > 0",
      "Differentiability"
    ],
    "correctIndices": [0, 1, 2]
  },
  {
    "id": "q-27",
    "category": "Other questions from previous years",
    "text": "Leaky ReLUs",
    "options": [
      "Saturate when the input is less than 0",
      "Need to perform exponential operations",
      "Tend to blow up when Z is large"
    ],
    "correctIndices": [2]
  },
  {
    "id": "q-28",
    "category": "Other questions from previous years",
    "text": "Maxout",
    "options": [
      "Has as special cases ReLU and Leaky ReLU",
      "Requires fewer parameters to be learned",
      "Does not have the problem of saturation",
      "Can approximate any convex function"
    ],
    "correctIndices": [0, 2, 3]
  },
  {
    "id": "q-29",
    "category": "Other questions from previous years",
    "text": "Weights in a network must be initialized",
    "options": ["At zero", "By maintaining symmetry", "With zero variance", "Randomly", "It is indifferent"],
    "correctIndices": [3]
  },
  {
    "id": "q-30",
    "category": "Other questions from previous years",
    "text": "Which of the following statements are true?",
    "options": [
      "When using SGD with mini-batches the model updates do not depend on the number of training examples",
      "When using SGD with mini-batches the number of updates to reach convergence does not depend on the number of training examples",
      "The choice of cost functions is tightly coupled with the choice of the output unit"
    ],
    "correctIndices": [1, 2]
  },
  {
    "id": "q-31",
    "category": "Other questions from previous years",
    "text": "The function max{0, min{1, Wh + b}} is a good choice as an output function for classification problems",
    "options": [
      "No because it does not return a value between 0 and 1",
      "Yes, because it returns a value between 0 and 1",
      "Yes, because it is linear",
      "No, because it is not good for training"
    ],
    "correctIndices": [3]
  },
  {
    "id": "q-32",
    "category": "Other questions from previous years",
    "text": "Softmax function",
    "options": [
      "Is a good choice for representing discrete probability distributions with n possible values",
      "It is a good output function because it is continuous and differentiable",
      "Since its output is a probability distribution it can always be interpreted as a confidence level",
      "If the prediction is correct the penalty is always 0"
    ],
    "correctIndices": [0, 1]
  },
  {
    "id": "q-33",
    "category": "Other questions from previous years",
    "text": "A Gaussian mixture output function",
    "options": [
      "Can represent multimodal functions",
      "The weight associated to a Gaussian in the mixture represents the probability of the output",
      "Mixtures are particularly suitable for generative models for speech or for movements of objects"
    ],
    "correctIndices": [0, 1, 2]
  },
  {
    "id": "q-34",
    "category": "Other questions from previous years",
    "text": "Regularization",
    "options": [
      "It reduces the validation/test error at the expense of (acceptable) training error",
      "Enables the model to reach a point that does minimize the loss function",
      "It enables the model to reduce the variability of the data"
    ],
    "correctIndices": [0]
  },
  {
    "id": "q-35",
    "category": "Other questions from previous years",
    "text": "Which of these sentences are true?",
    "options": [
      "Simpler models generalize better",
      "Multiple hypothesis (ensemble) models generalize better",
      "More complex models can represent the true data generating process",
      "In general, when building machine learning models, the data generating process is not known"
    ],
    "correctIndices": [0, 1, 3]
  },
  {
    "id": "q-36",
    "category": "Other questions from previous years",
    "text": "Regularizing estimators",
    "options": [
      "Reduce bias",
      "Reduce the gap between training error and validation error",
      "Reduce underfitting problems",
      "Can reduce the complexity of the model"
    ],
    "correctIndices": [1, 3]
  },
  {
    "id": "q-37",
    "category": "Other questions from previous years",
    "text": "Which of these sentences are false?",
    "options": [
      "If the weight of the regularization term in the loss function is too high it may imply underfitting",
      "If the weight of the penalization term in the loss function is too high it may imply overfitting",
      "Regularizing the bias parameters can introduce a significant amount of underfitting",
      "Regularizing the bias parameters can introduce a significant amount of overfitting",
      "Usually the bias parameters are not constrained by regularizing constraints"
    ],
    "correctIndices": [1, 3]
  },
  {
    "id": "q-38",
    "category": "Other questions from previous years",
    "text": "Parameter norm penalties",
    "options": [
      "Make the network more stable",
      "Minor variation or statistical noise on the inputs will result in large differences in the output",
      "Encourage the network toward using small weights"
    ],
    "correctIndices": [0, 2]
  },
  {
    "id": "q-39",
    "category": "Other questions from previous years",
    "text": "Consider norm penalizations",
    "options": [
      "Sum of absolute weights penalizes small weights more",
      "Squared weights penalize large values more",
      "L2 results in more sparse weights than L1",
      "The addition of the L2 term modifies the learning rule by shrinking the weight factor by a constant factor on each parameter update",
      "L2 rescales the weights along the axes defined by the eigenvectors of the Hessian matrix"
    ],
    "correctIndices":  1, 3]
  },
  {
    "id": "q-40",
    "category": "Other questions from previous years",
    "text": "Which of these sentences are true?",
    "options": [
      "Regularizing operators can be seen as soft constraints of the learning optimization problem",
      "Regularizing operators can be done by optimizing with respect to the loss function and then re-projecting the solution in the feasible region $(k\\Omega(\\theta)) < 0$",
      "Explicit constraints implemented by re-projection do not necessarily encourage the weights to approach the origin",
      "Explicit constraints implemented by re-projection only have an effect when the weights become large and attempt to leave the constraint region"
    ],
    "correctIndices": [0, 2, 3]
  },
  {
    "id": "q-41",
    "category": "Other questions from previous years",
    "text": "Dataset augmentation",
    "options": [
      "Creates fake data and adds it to the training set",
      "It is very effective for non-supervised tasks",
      "Injecting noise in the input to a neural network can also be seen as a form of data augmentation"
    ],
    "correctIndices": [0, 2]
  },
  {
    "id": "q-42",
    "category": "Other questions from previous years",
    "text": "Which of these sentences is true?",
    "options": [
      "Label smoothing is used for solving regression tasks",
      "Label smoothing makes models robust to possible errors in the training set",
      "Label smoothing can help convergence of maximum likelihood learning with a softmax classifier and hard targets"
    ],
    "correctIndices": [1, 2]
  },
  {
    "id": "q-43",
    "category": "Other questions from previous years",
    "text": "Which of these sentences is false?",
    "options": [
      "Multitask forces to share a set of parameters across different tasks",
      "Multitask improves generalization when tasks are very different"
    ],
    "correctIndices": [1]
  },
  {
    "id": "q-44",
    "category": "Other questions from previous years",
    "text": "Which of the following sentences are true?",
    "options": [
      "Parameter tying imposes a subset of parameters to be equal",
      "Early stopping is a form of regularization",
      "Bagging is a form of ensemble model",
      "Bagging is more effective if the output of the models learned are correlated",
      "Dropout is generally coupled with mini-batch based learning algorithms"
    ],
    "correctIndices": [1, 2, 4]
  },
  {
    "id": "q-45",
    "category": "Other questions from previous years",
    "text": "In machine learning, the cost function to minimize during the training process is the performance measure P representing the number of correct classifications on the test set",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-46",
    "category": "Other questions from previous years",
    "text": "The final aim of Machine Learning is",
    "options": [
      "The minimization of the true risk function",
      "The minimization of the empirical risk using a surrogate loss function"
    ],
    "correctIndices": [0]
  },
  {
    "id": "q-47",
    "category": "Other questions from previous years",
    "text": "A surrogate loss function",
    "options": [
      "Acts as a proxy to the true risk being \"nice\" enough to be optimized efficiently",
      "Acts as a proxy to empirical risk being \"nice\" enough to be optimized efficiently"
    ],
    "correctIndices": [1]
  },
  {
    "id": "q-48",
    "category": "Other questions from previous years",
    "text": "Early stopping halt criterion",
    "options": [
      "Is typically based on the performance obtained on a validation set",
      "Is typically based on the performance obtained on the training set"
    ],
    "correctIndices": [0]
  },
  {
    "id": "q-49",
    "category": "Other questions from previous years",
    "text": "The accuracy of the estimated mean of the gradient",
    "options": [
      "It depends on the number of samples used",
      "Has a standard error which decreases linearly with the number of samples used",
      "Has a standard error which decreases less than linearly with the number of samples"
    ],
    "correctIndices": [0, 2]
  },
  {
    "id": "q-50",
    "category": "Other questions from previous years",
    "text": "Ill conditioning of the Hessian matrix of the cost function",
    "options": [
      "Can be partly overcome by using the momentum strategy",
      "Can prevent the gradient from arriving at a critical point",
      "Can imply that very small steps are needed to decrease the cost function"
    ],
    "correctIndices": [0, 2]
  },
  {
    "id": "q-51",
    "category": "Other questions from previous years",
    "text": "Local minima in deep learning problems",
    "options": ["Are rare", "Are more common than saddle points", "Are much more likely to have a low cost than a high cost"],
    "correctIndices": [0, 2]
  },
  {
    "id": "q-52",
    "category": "Other questions from previous years",
    "text": "Neural networks with many layers",
    "options": [
      "Often have extremely steep regions (cliffs)",
      "Often have flat regions",
      "Often have many local minima of similar cost"
    ],
    "correctIndices": [0, 1, 2]
  },
  {
    "id": "q-53",
    "category": "Other questions from previous years",
    "text": "Momentum update rule",
    "options": [
      "Accumulates previous values of the cost function",
      "Can be incorporated in SGD",
      "Its step size is larger if previous gradients point in the same direction"
    ],
    "correctIndices": [1, 2]
  },
  {
    "id": "q-54",
    "category": "Other questions from previous years",
    "text": "AdaGrad algorithm",
    "options": [
      "Takes into account previous squared gradients",
      "Decreases the learning rate too much in the early stages",
      "Uses the same learning rate for all parameters"
    ],
    "correctIndices": [0]
  },
  {
    "id": "q-55",
    "category": "Other questions from previous years",
    "text": "RMSProp",
    "options": [
      "It takes into account the square gradients",
      "It is a modification of the AdaGrad algorithm",
      "It does not require hyperparameters"
    ],
    "correctIndices": [0, 1]
  },
  {
    "id": "q-56",
    "category": "Other questions from previous years",
    "text": "Adam algorithm",
    "options": [
      "It also takes into account the curvature of the cost function through the second order derivatives",
      "It uses the momentum strategy",
      "It is based on RMSProp"
    ],
    "correctIndices": [1, 2]
  },
  {
    "id": "q-57",
    "category": "Other questions from previous years",
    "text": "A good initialization procedure",
    "options": ["Assigns large weights", "Assigns extremely small weights", "None of the two"],
    "correctIndices": [2]
  },
  {
    "id": "q-58",
    "category": "Other questions from previous years",
    "text": "In initialization",
    "options": [
      "Larger weights break symmetry more",
      "Smaller weights propagate information more efficiently",
      "Large weights make the model more likely to reach solutions with good generalization property",
      "Small weights make the model more robust"
    ],
    "correctIndices": [0, 3]
  },
  {
    "id": "q-part2-1",
    "category": "Second part of the course",
    "text": "Convolution is",
    "options": ["Local in space, local in depth", "Local in space, full in depth", "Full in space, local in depth", "Full in space, full in depth"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-2",
    "category": "Second part of the course",
    "text": "Given the input volume with size H * W * K and a filter bank with size h * w * k we want to convolve them. The size of the output volume will be:",
    "options": [
      "H * W * K",
      "h * w * k",
      "(H - h + 1) * (W - w + 1) * k",
      "(H - h - 1) * (W - w - 1) * k",
      "(H - h + 1) * (W - w + 1) * 1",
      "(H - h - 1) * (W - w - 1) * 1"
    ],
    "correctIndices": [4]
  },
  {
    "id": "q-part2-3",
    "category": "Second part of the course",
    "text": "How many channels (i.e. depth size) will have the output volume resulting from the convolution of an input volume with 16 channels (i.e. depth=16) with a filter bank of 16 filters?",
    "options": ["1", "16", "32"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-4",
    "category": "Second part of the course",
    "text": "Compute the output after the application of max pooling to the following input volume IN with a neighborhood of size=2 and stride=2. IN = [12 23 40 31] [11 15 42 52]",
    "options": ["[40; 52]", "[23 42 52]", "[23 52]"],
    "correctIndices": [2]
  },
  {
    "id": "q-part2-5",
    "category": "Second part of the course",
    "text": "Which are common techniques to reduce overfitting?",
    "options": ["Weight decay", "Local response normalization", "Data augmentation", "Data normalization", "Dropout"],
    "correctIndices": [0, 2, 4]
  },
  {
    "id": "q-part2-6",
    "category": "Second part of the course",
    "text": "In data augmentation we have seen different policies (e.g. cropping, rotation, color cast, vignetting...)",
    "options": ["All policies are safe to use to any problem", "Only a subset of policies is safe to be used for each problem"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-7",
    "category": "Second part of the course",
    "text": "We can diagnose the training and understand if we are overfitting:",
    "options": [
      "By plotting the loss on the training set across epochs",
      "By plotting the loss on the validation set across epochs",
      "By plotting the loss on the test set across epochs",
      "By plotting the accuracy on the training and validation sets across epochs"
    ],
    "correctIndices": [3]
  },
  {
    "id": "q-part2-8",
    "category": "Second part of the course",
    "text": "We can continue adding layers to a NN and we will continue to obtain better results",
    "options": ["Yes", "No"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-9",
    "category": "Second part of the course",
    "text": "GoogLeNet (i.e. Inception-v1) introduced the use of auxiliary classifiers:",
    "options": ["To mitigate the problem of vanishing gradients", "To perform multi-task classification", "To reduce overfitting"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-10",
    "category": "Second part of the course",
    "text": "ResNets were able to train a model with 150+ layers by:",
    "options": ["Using just one fully-connected layer", "Introducing the residual connections", "Using just 3x3 convolutional filters"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-11",
    "category": "Second part of the course",
    "text": "If we have few data:",
    "options": ["We cannot use Deep Learning", "We can still use deep learning"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-12",
    "category": "Second part of the course",
    "text": "From which layer we can extract activations to be used as features to classify data for a new small dataset that is similar to the dataset used to pre-train the whole network?",
    "options": ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5", "FC6", "FC7", "FC8"],
    "correctIndices": [5, 6, 7]
  },
  {
    "id": "q-part2-13",
    "category": "Second part of the course",
    "text": "Model compression is only used to allow models to run on mobile devices:",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-14",
    "category": "Second part of the course",
    "text": "Which is not a model compression technique?",
    "options": ["Weight sharing", "Network pruning", "Low rank matrix decomposition", "Dropout", "Knowledge distillation", "Quantization"],
    "correctIndices": [3]
  },
  {
    "id": "q-part2-15",
    "category": "Second part of the course",
    "text": "Magnitude weight pruning removes the weights having:",
    "options": ["The lowest value", "The lowest absolute value", "The highest absolute value", "The highest value"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-16",
    "category": "Second part of the course",
    "text": "Structured pruning",
    "options": ["Aims to preserve network density for computational efficiency", "Aims to increase network sparsity for computational efficiency"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-17",
    "category": "Second part of the course",
    "text": "Low rank matrix decomposition is particularly useful in:",
    "options": ["Convolutional layers", "Pooling layers", "Fully connected layers"],
    "correctIndices": [2]
  },
  {
    "id": "q-part2-18",
    "category": "Second part of the course",
    "text": "Global MBP tends to outperform layer-wise MBP:",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-19",
    "category": "Second part of the course",
    "text": "RNNs are a family of neural networks for processing sequential data:",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-20",
    "category": "Second part of the course",
    "text": "The computation in most RNNs can be decomposed in 3 blocks of parameters and associated transformations:",
    "options": [
      "From the input to the hidden state",
      "From the hidden state to the input",
      "From the previous hidden state to the next hidden state",
      "From the next hidden state to the previous hidden state",
      "From the hidden state to the output",
      "From the output to the hidden state"
    ],
    "correctIndices": [0, 2, 4]
  },
  {
    "id": "q-part2-21",
    "category": "Second part of the course",
    "text": "What is the name of the algorithm used to train RNNs?",
    "options": ["Backpropagation", "Backpropagation through recurrency", "Backpropagation through time"],
    "correctIndices": [2]
  },
  {
    "id": "q-part2-22",
    "category": "Second part of the course",
    "text": "Vanishing gradients are more easy to identify than exploding gradients:",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-23",
    "category": "Second part of the course",
    "text": "Exploding gradients are more difficult to handle than vanishing gradients:",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-24",
    "category": "Second part of the course",
    "text": "Vanishing gradients",
    "options": ["Bias the parameters to capture short-term dependencies", "Bias the parameters to capture long-term dependencies"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-25",
    "category": "Second part of the course",
    "text": "Gated RNNs are based on the idea of creating paths through time that have derivatives that neither vanish nor explode:",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-26",
    "category": "Second part of the course",
    "text": "LSTMs have the following gates:",
    "options": ["Input gate", "Remember gate", "Recurrent gate", "Forget gate", "Output gate", "Hidden gate"],
    "correctIndices": [0, 3, 4]
  },
  {
    "id": "q-part2-27",
    "category": "Second part of the course",
    "text": "GRUs have:",
    "options": ["Significantly less parameters than LSTMs", "Significantly more parameters than LSTMs"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-28",
    "category": "Second part of the course",
    "text": "Federated learning aims to:",
    "options": ["Collaboratively train a ML model", "Independently train a ML model"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-29",
    "category": "Second part of the course",
    "text": "In federated learning, the data:",
    "options": ["Is shared across parties/server", "Is kept private"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-30",
    "category": "Second part of the course",
    "text": "In federated learning:",
    "options": ["We control how the data is distributed across parties/workers", "Data in each party/worker is not independent and identically distributed"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-31",
    "category": "Second part of the course",
    "text": "In federated learning there is always a server to orchestrate the training:",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-32",
    "category": "Second part of the course",
    "text": "In the FedAVG algorithm the central model is updated:",
    "options": [
      "Taking the minimum value of the parameters in the corresponding layers across the models sent by the different workers",
      "Taking the mean value of the parameters in the corresponding layers across the models sent by the different workers",
      "Taking the median value of the parameters in the corrisponding layers across the models sent by the different workers",
      "Taking the maximum value of the parameters in the corrisponding layers across the models sent by the different workers"
    ],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-33",
    "category": "Second part of the course",
    "text": "Transformers can process sequences of arbitrary length",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-34",
    "category": "Second part of the course",
    "text": "Transformers process each input independently",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-35",
    "category": "Second part of the course",
    "text": "The only processing module in the transformer layer is self attention",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-36",
    "category": "Second part of the course",
    "text": "Self attention, being a composition of two linear transformations is linear",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-37",
    "category": "Second part of the course",
    "text": "The elements that are needed for the computation of self attention are",
    "options": ["Outputs", "Inputs", "Values", "Variables", "Queries", "Questions", "Keys", "Chains"],
    "correctIndices": [1, 2, 4, 6]
  },
  {
    "id": "q-part2-38",
    "category": "Second part of the course",
    "text": "A trasformer head is completely defined by 3 weight matrices and 3 biases",
    "options": ["True", "False"],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-39",
    "category": "Second part of the course",
    "text": "To obtain the best performance usually transformers use one single head",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-40",
    "category": "Second part of the course",
    "text": "Self-Supervised Learning refers to learning methods in which the models are trained:",
    "options": [
      "with supervisory signals that are generated from the data itself by leveraging its structure",
      "with supervisory signals provided by human annotated labels",
      "without the need of any supervisory signal"
    ],
    "correctIndices": [0]
  },
  {
    "id": "q-part2-41",
    "category": "Second part of the course",
    "text": "How are called the labels automatically created for the pretext task?",
    "options": ["Pretext labels", "Pseudo labels", "Proxy labels", "Generated labels"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-42",
    "category": "Second part of the course",
    "text": "Image generation cannot be used as a pretext task",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-43",
    "category": "Second part of the course",
    "text": "When we use image generation as the pretext task, after training we are interested to use",
    "options": ["The generator", "The discriminator", "Both"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-44",
    "category": "Second part of the course",
    "text": "The performance of SSL models are usually measured comparing their accuracy on the pretext task",
    "options": ["True", "False"],
    "correctIndices": [1]
  },
  {
    "id": "q-part2-45",
    "category": "Second part of the course",
    "text": "Pretext tasks cannot expolit multimodal properties of the data",
    "options": ["True", "False"],
    "correctIndices": [1]
  }
];
