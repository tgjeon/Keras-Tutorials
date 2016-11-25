# 딥러닝 용어사전

A
--------------

### **ACTIVATION FUNCTION**

뉴럴 네트워크가 복잡한 결정 경계 (decision boundaries)를 학습하기 위해서, 특정 레이어에 비선형 활성화 함수를 적용합니다. 일반적으로 많이 사용되는 함수는 [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [tanh](http://mathworld.wolfram.com/HyperbolicTangent.html), [ReLU (Rectified Linear Unit)](http://cs231n.github.io/neural-networks-1/)이며, 다양한 변형 기법도 사용됩니다. 


### **ADADELTA**

Adadelta는 경사강하 기반 학습 기법으로, 학습률을 시간이 변화함에 따라 조정합니다. 하이퍼파라미터에 민감하며, 너무 빠르게 학습률이 감소하는 [Adagrad]보다 향상시키기 위해 제안되었습니다. Adadelta는 [rmsprop] 기법과 유사하며 기본 [SGD]를 대신하여 사용할 수 있습니다.

* [ADADELTA: An Adaptive Learning Rate Method](http://arxiv.org/abs/1212.5701)
* [Stanford CS231n: Optimization Algorithms](http://cs231n.github.io/neural-networks-3/)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


### **ADAGRAD**

Adagrad는 적응형 학습률 변화기법입니다. 기울기의 제곱을 시간 변화에 따라 계속 유지하도록 하는 것이 특징이다. 기본 SGD를 대신항 사용할 수 있으며, 희소 데이터 (sparse data)에 사용하기 좋다. 희소 데이터에 높은 학습률을 할당하면 불규칙적으로 파라미터가 업데이트 되기 때문이다. (불규칙적인 데이터에 대해서도 방향성을 유지한다)

* [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.magicbroom.info/Papers/DuchiHaSi10.pdf)
* [Stanford CS231n: Optimization Algorithms](http://cs231n.github.io/neural-networks-3/)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


### **ADAM**

Adam은 적응형 학습률 변화기법입니다. rmsprop과 유사하지만, 업데이트는 현재 기울기의 1차, 2차 모멘트, 편향(bias)를 이용해 즉시 추정됩니다.

* [Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
* [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/)


### **AFFINE LAYER**

뉴럴 네트워크에서 완전 연결된 레이어 (fully-connected layer)입니다. Affine은 이전 레이어의 각 뉴런들이 현재 레이어의 각 뉴런들에 연결되어 있음을 뜻합니다. 여러모로, 이것은 뉴럴 네트워크의 "표준" 레이어입니다. Affine layer는 종종 Convolutional Neural Networks나 Recurrent Neural Networks의 가장 상위 출력에서 최종 예측을 하기 이전에 추가됩니다. Affine layer는 일반적으로 y = f(Wx + b)의 형태로 나타내며, x는 입력 레이어, W는 파라미터, b는 편향 (bias), f는 비선형 활성화 함수입니다. 


### **ATTENTION MECHANISM**

Attention 메커니즘은 이미지에서 특정 부분에 집중하는 사람의 시각적 주목 능력에 영감을 받았습니다. Attention 메커니즘은 자연어 처리와 이미지 인식 아키텍쳐에서 네트워크가 예측하기 위해서 주목해서 학습해야 하는 것에 대해 표현 가능합니다. 

* [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)


### **ALEXNET**

Alexnet은 이미지넷 대회 (ILSVRC 2012)중 이미지 인식에서 가장 큰 성능차이로 우승했던, Convolutional Neural Networks (CNN) 구조이며, CNN의 관심을 부활시킨 장본인입니다. 5개의 convolutional 레이어와 그 뒤 일부분의 max-pooling 레이어로 구성되어 있으며, 3개의 완전 연결된 (fully-connected) 레이어와 1000개의 출력을 가진 softmax를 포함합니다. Alexnet은 [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)에 소개되었습니다.


### **AUTOENCODER**

Autoencoder는 뉴럴 네트워크 모델 중 입력 자체를 예측하기 위한 목적을 가진 모델입니다. 네트워크 내부에 병목지역 ("bottleneck")을 포함하고 있는 것이 특징입니다. 병목지역부터 소개하자면, 네트워크로 하여금 입력에 대한 저차원 표현법을 학습하게끔 합니다. 효율적으로 입력에 대한 좋은 표현법으로 압축하는 것이죠. Autoencoder는 PCA나 다른 차원 축소 기법과 관련이 있습니다. 하지만 비선형적인 특성때문에 더 복잡한 연관성을 학습할 수 있습니다. [Denoising Autoencoders](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), [Variational Autoencoders](http://arxiv.org/abs/1312.6114), [Sequence Autoencoders](http://arxiv.org/abs/1511.01432)와 같이 다양한 형태의 autoencoder 모델이 있습니다.

### **AVERAGE-POOLING**

Average-Pooling은 이미지 인식을 위해 Convolutional Neural Networks에 사용되는 pooling 기법중 하나입니다. 특징이 표현된 패치 위를 윈도우가 순회하며 해당 윈도우의 모든 값의 평균을 취합니다. 이는 입력 표현을 저차원의 표현으로 압축하는 역할을 합니다.



B
--------------

### **BACKPROPAGATION**

Backpropagation은 뉴럴 네트워크에서 (혹은, 일반적인 feedforward 그래프에서) 효율적으로 경사(gradient) 를 계산하기 위한 방법입니다. 네트워크 출력으로 부터 편미분 연쇄 법칙을 이용해 경사도를 계산하여 입력쪽으로 전달합니다. Backpropagation의 첫 사용은 1960년대의 Vapnik의 사례로 거슬러 올라갑니다. 하지만 [Learning representations by back-propagating errors](http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html)이 근원이라고 종종 인용되고 있습니다. 

* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)


### **BACKPROPAGATION THROUGH TIME (BPTT)**

Backpropagation Through Time ([논문](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf))은 Backpropagation 알고리즘이 Recurrent Neural Networks (RNNs) 에 적용된 것입니다. BPTT는 RNN에 적용되는 표준적인 backpropagation 알고리즘이라고 볼수 있습니다. RNN은 각 시점이 recurrent 레이어를 나타내며, 각 recurrent 레이어는 파라미터를 공유합니다. BPTT는 이름 그대로, RNN이 모든 시점에 통해 같은 파라미터를 공유하기 때문에, 한 시점의 에러는 모든 이전 시점에게 "시간을 거슬러 (through time)" 전달됩니다. 수백개의 입력으로 구성된 긴 sequences를 처리할 때, 계산 효율을 위해 truncated BPTT가 종종 사용됩니다. Truncated BPTT는 정해진 시점까지만 에러를 전달하고 멈춥니다. 

* [Backpropagation Through Time: What It Does and How to Do It](http://deeplearning.cs.cmu.edu/pdfs/Werbos.backprop.pdf)


### **BATCH NORMALIZATION**

Batch Normalization은 레이어의 입력을 mini-batch로 정규화 하기 위한 기법입니다. 학습 속도를 높여주고, 높은 학습률을 사용 가능하게 하며, 정규화 (regularization) 하도록 합니다. Batch Normalization은 Convolutional Neural Networks (CNN)과 Feedforward Neural Networks (FNN)에 아주 효과적이라고 밝혀졌습니다. 하지만 Recurrent Neural Networks 에는 아직 성공적으로 적용되지 않았습니다. 

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
* [Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378)


### **BIDIRECTIONAL RNN**

Bidirectional Recurrent Neural Network은 두개의 서로 다른 방향을 가진 RNN을 포함하는 뉴럴 네트워크를 의미합니다. 순방향 (forward) RNN은 입력 sequence를 처음부터 끝까지 읽고, 역방향 (backward) RNN은 끝에서 부터 처음의 방향으로 읽습니다. 두 RNN은 두가지 방향 벡터로 볼수 있으며, 입력 레이어 상단에 두 RNN을 쌓고, 그 위에 출력 레이어로 묶게 됩니다. Bidirectional RNN은 자연어 처리 문제에서 종종 사용됩니다. 특정 단어의 앞, 뒤 단어의 의미를 통해 현재 단어를 예측하는 문제에 적용됩니다.

* [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)


### **BOLTZMAN MACHINE**

볼츠만 머신은 0 또는 1의 값을 취하는 다수의 뉴런으로 구성된 네트워크로써, 모든 뉴런은 서로 연결되어 있습니다. 볼츠만 머신은 트레이닝 데이터로부터 복잡한 규칙적인 패턴을 나타내는 흥미로운 특징들을 발견할 수 있는 학습 알고리즘을 가지고 있습니다. 하지만 모델 구현이 매우 어렵기 때문에 비져블 레이어와 히든 레이어를 분리시킨 제한된 볼츠만 머신(RBM)의 형태로 주로 사용됩니다.


C
--------------

### **CAFFE**

[Caffe](http://caffe.berkeleyvision.org/)는 Berkeley Vision and Learning Center에서 개발된 딥러닝 프레임워크입니다. Caffe는 특히 비전관련 영역과 CNN 모델을 처리하는데 특화되어 있습니다. 


### **CATEGORICAL CROSS-ENTROPY LOSS**

카테고리별 cross-entropy 손실은 음수 로그 우도값이라고도 알려져있습니다. 분류 문제에 대해 많이 사용되는 손실함수 입니다. 두 확률 분포가 얼마나 유사한지 평가하는데 사용됩니다. 보통 정답 레이블과 예측된 레이블에 대해서 유사함을 평가하는데 사용합니다. 수식으로는 ``` L = -sum(y * log(y_prediction)) ``` 와 같이 표현됩니다. 여기서 ```y```는 정답 레이블의 확률 분포 (보통 one-hot vector로 표현됩니다) 이며, ```y_prediction```은 예측된 레이블의 확률 분포입니다 (보통 [softmax](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#softmax)의 결과).        

### **CHANNEL**

딥러닝 모델의 입력은 여러 채널을 가질 수 있습니다. 기본적인 예는 이미지이며, RGB채널을 가지고 있죠. (R: Red, G: Green, B: Blue). 이미지는 3차원의 텐서로 표현 가능합니다. 각 차원은 이미지의 너비(width), 높이(height), 채널(channel)로 나타납니다. 다른 예로, 자연어도 다양한 채널을 가지고 있습니다. [embedding](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#embedding)하는 형태에 따라 달라지게 됩니다. 

### **CONVOLUTIONAL NEURAL NETWORK (CNN, CONVNET)**

CNN은 [convolutions](https://en.wikipedia.org/wiki/Convolution)을 사용하여 입력의 일부 영역으로부터 특징점을 추출합니다. 대부분의 CNN은 convolutional layer, [pooling layer](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#pooling), [affine layers](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#affine-layer)를 조합해서 구성됩니다. CNN은 visual recognition에서 수년간 최고의 성능을 보이며, 인기를 얻게 되었습니다. 

* [Stanford CS231n class – Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
* [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

D
--------------

### **DEEP BELIEF NETWORK (DBN)** (작성중)

DBN은 데이터로 부터 비교사 방식 (unsupervised manner)을 통해 계층적 표현법 (hierarchical representation)을 배우는 확률 그래프 모델중 하나입니다. 

* [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)


### **DEEP DREAM**

구글에서 개발되었으며, CNN을 통해 학습된 지식을 추출하도록 시도된 기술입니다. 이 기술을 통해 새로운 이미지를 생성하거나, 기존의 이미지를 마치 꿈꾸는 것처럼 반복적으로 적용하여 변형시킬 수 있습니다. 

* [Deep Dream on Github](https://github.com/google/deepdream)
* [Inceptionism: Going Deeper into Neural Networks](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html)

### **DROPOUT** (작성중)

Dropout은 뉴럴 네트워크가 과적합 (overfitting)되는 것을 막는 정규화 (regularization) 기법입니다. 

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
* [Recurrent Neural Network Regularization](http://arxiv.org/abs/1409.2329)


E
--------------

### **EMBEDDING**

### **EXPLODING GRADIENT PROBLEM**


F
--------------

### **FINE-TUNING**


G
--------------

### **GRADIENT CLIPPING**

### **GLOVE**

### **GOOGLELENET**

### **GRU**


H
--------------

### **HIGHWAY LAYER**



I
--------------
### **ICML**

[International Conference for Machine Learning](http://icml.cc/)의 줄임말. 머신러닝 관련 학회에서 최고수준 학회입니다.

### **ILSVRC**

[ImageNet Large Scale Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/)는 대규모 물체 인식과 이미지 분류에 대해 알고리즘을 평가합니다. 컴퓨터 비전 분야에서 가장 유명한 학술적 challenge입니다. 과거 수년간, 딥러닝 기술이 인식오차를 30%대에서 5%대로 끌어내리는 주요한 역할을 했으며, 다양한 분류 임무에서 사람의 능력을 뛰어넘었습니다. 

### **INCEPTION MODULE**

인셉션 모듈은 CNN에서 사용되었으며, 효율적인 계산을 가능하게 하였습니다. 그리고, 1x1 convolutions을 쌓아서 차원 축소가 가능한 deeper network가 가능합니다. 

* [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)

K
--------------
### **KERAS**

[Keras](http://keras.io/)는 파이썬 기반 딥러닝 라이브러리 입니다. 많은 고차원의 딥러닝 표현법을 제공하고 있습니다. [TensorFlow](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#tensorflow)나 [Theano](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#theano)를 내부적인 동작법으로 선택하여 사용할 수 있습니다. 

L
--------------
### **LSTM**

Long Short-Term Memory 네트워크는 Recurrent Neural Network에서 나타는 [기울기가 사라지는 문제 (vanishing gradient problem)](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#vanishing-gradient-problem)를 막기위해 기억을 열고 닫는 메커니즘 (memory gating mechanism)으로 개발되었습니다. LSTM 유닛을 통해 RNN에서  hidden state를 계산하는 부분에서 효율적으로 gradient를 전달할 수 있습니다. 또한, 긴 길이의 의존성 (long-range dependencies)을 학습 할 수 있습니다.

* [Long Short-Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [Recurrent Neural Network Tutorial, Part 4 – Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/) 


M
--------------
### **MAX-POOLING**

### **MNIST**

[MNIST 데이터셋](http://yann.lecun.com/exdb/mnist/)은 이미지 인식 분야에서 가장 널리 쓰이는 데이터셋일 겁니다. 숫자 필기체에 대해 6만개의 학습 예제와 1만개의 테스트 예제를 포함하고 있습니다. 각 이미지는 28x28 픽셀 크기이며, 가장 우수한 모델은 테스트 예제에 대해 99.5%대의 정확도를 보이고 있습니다.   

### **MOMENTUM**

모멘텀은 Gradient Descent Algorithm (기울기 강하 기법)를 확장하기 위한 기능이며, 파라미터 갱신을 가속시키거나, 느리게 합니다. 실제로, 기울기 강하 갱신부분에서 모멘텀을 포함시키면 Deep Network에서 더 나은 수렴률을 보입니다. 

* [Learning representations by back-propagating errors](http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html)

### **MULTILAYER PERCEPTRON (MLP)**

MLP는 여러개의 fully connected 레이어로 구성된 Feedforward Neural Network (전방향 뉴럴 네트워크)입니다. 선형으로 분류하기 힘든 데이터를 다루기 위해 비선형 [활성화 함수](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#activation-function)를 사용하고 있습니다. MLP는 2개 이상의 레이어를 지닌 다층 뉴럴 네트워크 (multilayer Neural Network)나 딥 뉴럴 네트워크 (deep Neural Networks)의 가장 기본적인 형태입니다. 


N
--------------

### **NEGATIVE LOG LIKELIHOOD (NLL)**

[Categorical Cross Entropy Loss](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#categorical-cross-entropy-loss)를 확인하세요.


### **NEURAL MACHINE TRANSLATION (NMT)**

### **NEURAL TURING MACHINE (NTM)**

### **NONLINEARITY**

[Activation Function](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#activation-function)을 확인하세요.

### **NOISE-CONTRASTIVE ESTIMATION (NCE)**


P
--------------
### **POOLING**


R
--------------

### **RESTRICTED BOLTZMANN MACHINE (RBM)**

RBM(제한된 볼츠만 머신)은 인공신경망으로 해석될 수 있는 확률 그래프 모델 중 하나입니다. RBM은 비교사 방식(unsupervised manner)으로 데이터 표현법을 학습합니다. 한 개의 RBM은 비져블 레이어와 히든 레이어, 그리고 각 레이어 간 모든 뉴런의 연결로 구성되어 있습니다. 경사하강법에 대한 근사인 Contrastive Divergence를 이용하여 RBM을 효율적으로 학습할 수 있습니다.


### **RECURRENT NEURAL NETWORK (RNN)**

RNN은 히든 스테이트, 또는 메모리를 통해 순차적으로 반응하며, N 개의 입력을 취하여 최대 N 개의 출력을 만들 수 있습니다. 예를 들어, 입력 시퀀스는 대화를 구성하는 단어(음성)들이며 출력 시퀀스는 이에 해당하는 문자입니다(N-to-N). 한 개의 입력 문장에 해당하는 감정을 분석할 수도 있습니다(N-to-1). 어떤 이미지에 설명 문장을 생성하는 것 또한 가능합니다(1-to-N).
각 단위 시간마다, RNN은 현재 입력과 이전 히든 스테이트(“Memory”)를 기준으로 새로운 히든 스테이트를 계산합니다. “recurrent”라는 단어는 각 단계에서 동일한 파라미터가 사용되고 네트워크가 서로 다른 입력을 기반으로 동일한 계산을 수행한다는 사실에서 비롯되었습니다.


### **RECURSIVE NEURAL NETWORK**

재귀 신경망은 RNN을 트리 구조로 일반화 한 것입니다. RNN과 마찬가지로 재귀 신경망은 backpropagation을 사용하여 end-to-end로 학습시킬 수 있으며, 각 재귀에서 동일한 가중치가 적용됩니다. 최적화 문제의 일부로 트리 구조를 학습하는 것이 가능하지만 자연어 처리의 구문 분석 트리와 같이 미리 정의된 구조를 갖는 문제에 재귀 신경망을 적용하는 경우가 종종 있습니다.


### **RELU**

Rectified linear unit의 약자로 DNN에서 활성화 함수로 사용됩니다. f(x) = max(0, x)로 정의되며, tanh와 같은 활성화 함수들에 비해 활성화 결과가 쉽게 0으로 설정되기 때문에 희소성이 높고 vanishing gradient 문제로 인한 피해가 적다는 장점이 있습니다. CNN에서 가장 일반적으로 사용되는 활성화 함수이며, Leaky ReLU, Parametric ReLU 또는 Softplus와 같은 여러 변형이 있습니다.


### **RESNET**

Deep Residual Networks는 ILSVRC 2015 대회에서 우승한 모델입니다. 보다 쉬운 학습을 위해 shortcut 연결을 도입하여 깊은 망에 대한 효과를 극대화하였습니다. 기존 모델이 weighted sum 및 활성화 함수를 거쳐 결과값 H(x)를 출력하며 학습을 통해 최적의 H(x)를 얻는 것이 목표였다면, ResNet은 shortcut 연결을 통하여 H(x) - x의 결과를 최적화 시키는 것이 목표입니다. 최적점에서는 H(x) = x 이므로 학습에 대한 목표가 뚜렸하며 H(x) - x가 0으로 수렴하는 작은 움직임을 학습한다고 하여 Residual mapping이라는 용어를 사용합니다. 이러한 shortcut 연결은 highway 레이어와 비슷하지만, 데이터에 독립적이므로 추가 매개 변수나 학습 과정이 복잡하지 않습니다. ResNet은 ImageNet 테스트 셋에서 3.57%의 오류율을 달성하였습니다.


### **RMSPROP**

S
--------------

### **SEQ2SEQ**

### **SGD**

### **SOFTMAX**


T
--------------

### **TENSORFLOW**

[TensorFlow](https://www.tensorflow.org/)는 데이터 흐름 그래프를 이용한 수치 계산용 오픈소스 C++/파이썬 소프트웨어 라이브러리입니다. 구글에서 제작되었으며, 특히 딥러닝을 위해 많이 사용되고 있습니다. 설계적인 측면에서 봤을 때, [Theano](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#theano)랑 유사하며, [Caffe](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#caffe)나 [Keras](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#keras)보다 낮은 단계의 표현을 지닙니다.    

### **THEANO**

[Theano](http://deeplearning.net/software/theano/)는 파이썬 라이브러리이며, 사용자가 수학표현에 대해 정의, 최적화, 평가를 가능하게 합니다. 딥 뉴럴 네트워크에 대해 다양한 구성요소를 포함하고 있습니다. [TensorFlow](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#tensorflow)와 유사한 낮은 단계의 라이브러리이며, 이보다 높은 레벨의 표현을 지니는 라이브러라는 [Keras](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#keras)와 [Caffe](https://github.com/tgjeon/Keras-Tutorials/blob/master/DeepLearningGlossary.md#caffe)가 있습니다.  
 

V
--------------

### **VANISHING GRADIENT PROBLEM**

### **VGG**

VGG는 2014년 ImageNet 대회에서 localization 부문 1위, classification 부문 2위를 차지했던 convolutional neural network 모델을 말합니다. VGG 모델은 16-19개의 가중치 레이어를 포함하고 있으며 3x3과 1x1의 작은 필터 크기를 가진 convolution이 사용되었습니다.

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/abs/1409.1556)


W
--------------

### **WORD2VEC**





## References

* [Deep Learning Glossary @ wildml.com](http://www.wildml.com/deep-learning-glossary/)
