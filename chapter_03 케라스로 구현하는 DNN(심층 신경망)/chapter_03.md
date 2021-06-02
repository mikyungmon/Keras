# Chapter 03 # 

**심층 신경망(Deep Neural Network)이란?**

- 은닉 계층을 많이 쌓아서 만든 인공지능 기술이다.

- 2장에서 다룬 ANN은 주로 은닉 계층 하나를 포함했다. 그러나 DNN은 수십에서 수백의 은닉 계층으로 구성되기도 한다.

- 그덕분에 더 우수한 성능을 낼 수 있으며 적용 분야도 훨씬 다양하다.

## 3.1 DNN 원리 ## 

### 3.1.1 DNN 개념과 구조 ###

DNN은 은닉 계층이 여러 개인 신경망이다. 

구조는 다음 그림과 같다.

![image](https://user-images.githubusercontent.com/66320010/120432086-5acf5980-c3b4-11eb-869e-1b08e37534a9.png)

2장의 ANN과 달리 제1 은닉 계층의 결과는 출력 계층이 아닌 제2 은닉 계층으로 들어간다.

또한 제2 은닉 계층의 결과도 이어지는 다음 은닉 계층으로 계속해서 들어갈 수 있다.

**이런 방식으로 다수의 은닉 계층을 활용하면 은닉 계층 하나를 사용할 때보다 입력 신호를 더 정교하게 처리할 수 있다.**

DNN은 이때문에 전체 노드 수가 늘어나 **과적합**이 될 수 있지만 최근 이를 효과적으로 해결하는 다양한 방식이 제시되었다.

DNN은 복잡도가 높은 비정형 빅데이터에 용이하지만 결국은 과적합을 얼마나 방지하느냐가 이를 제대로 활용하는 열쇠이다.

### 3.1.2 경사도 소실 문제와 ReLU 활성화 함수 ###

DNN을 포함한 인공신경망에 있어서 **경사도 소실 문제(vanishing gradient problem)** 에 대한 이해와 적절한 활성화 함수 선택은 최적화에 중요하다.

- 경사도 소실 문제

  - DNN은 여러 은닉 계층으로 구성되어 지능망의 최적화 과정에서 학습에 사용하는 활성화 함수에 따라 경사도 소실이 발생할 수 있다.

  - DNN은 여러 계층으로 구성되어 있고 각 계층 사이에 활성화 함수가 반복적으로 들어있어 오차역전파를 계산할 때 경사도 계산이 누적된다 => 이로인해 경사 하강법을 사용하는 오차역전파 알고리즘의 성능이 나빠질 수 있다.

  - 시그모이드 함수와 같이 입력을 특정 범위로 줄이는 활성화 함수들은 입력이 크면 경사도가 매우 작아져 경사도 소실을 유발할 가능성이 높다.

- ReLU 활성화 함수

  - DNN에서는 경사도 소실 문제를 극복하는 활성화 함수로 ReLU 등을 사용한다.

  - ReLU는 입력이 0보다 큰 구간에서는 직선 함수이기 때문에 값이 커져도 경사도를 구할 수 있다 => 따라서 **경사도 소실 문제에 덜 민감하다.**

![image](https://user-images.githubusercontent.com/66320010/120432124-6cb0fc80-c3b4-11eb-9b2a-69c55ff2f62d.png)

### 3.1.3 DNN 구현 단계 ###

분류 DNN 구현을 다음과 같이 4단계로 구성해보았다.

이 장에서는 패키지 불러오기를 별도 단계로 수행하지 않고 각 단계에서 필요할 때 수행하여 단계별로 필요한 패키지를 각각 부르게 하여 어떤 단계에서 어느 패키지를 부르는지 확인하도록 한다.

< 구현 단계 >

1) 기본 파라미터 설정
2) 분류 DNN 모델 구현
3) 데이터 준비
4) DNN의 학습 및 성능 평가

## 3.2 필기체를 분류하는 DNN 구현 ##

이번에 사용할 데이터셋은 앞에서 사용한 0에서 9까지로 구분된 필기체 숫자들의 모음이다(5만 개의 학습데이터와 1만 개의 성능 평가 데이터로 구성).

위에서 말한 4단계로 은닉 계층이 늘어난 DNN을 구현한다.

### 3.2.1 기본 파라미터 설정 ###

1. DNN 구현에 필요한 파라미터 정의

       Nin = 784
       Nh_l = [100, 50]
       number_of_class = 10
       Nout = number_of_class

    - 입력 노드 수는 입력 이미지 크기에 해당하는 784개

    - 출력 노드 수는 분류할 클래스 수와 같은 10개

    - 은닉 계층은 두 개 이므로 각각에 대한 은닉 노드 수를 100과 50으로 지정

### 3.2.2 DNN 모델 구현 ###

2. DNN 모델 구현

   이번에는 객체지향 방식으로 DNN모델링 구현한다. 연쇄 방식으로 계층들을 기술할 것이므로 DNN 객체를 models.Sequential로부터 상속 받는다.

   모델링은 객체의 초기화 함수인 __init__()에서 구성한다.

        class DNN(models.Sequential):
          def __init__(self,Nin,Nh_l,Nout):
            super.__init__()

   - 연쇄 방식으로 구성할 것이므로 부모 클래스의 초기화 함수를 먼저 불러서 모델링이 시작됨을 알린다.

   - DNN의 은닉 계층과 출력 계층은 모두 케라스의 layers 서브패키지 아래에 Dense()개체로 구성한다.

         self.add(layers.Dense(Nh_l[0], activation = 'relu',input_shape = (Nin,), name = 'Hidden-1'))
         self.add(layers.Dropout(0.2))

   - 연쇄 방식으로 모델링을 기술하는 self.add()는 제1 은닉 계층부터 기술한다. 이름은 'Hidden-1'로 설정한다.

   - 입력 계층 정의는 첫 번째 은닉 계층의 정의와 함께 이루어진다. 첫 번째 은닉 계층 정의 시 input_shape을 적어줌으로 입력 계층이 Nin개의 노드로 구성된 벡터 방식임을 지정한다.

   - Dropout(p) : p라는 확률로 출력 노드의 신호를 보내다 말다 한다. p라는 확률로 앞 계층의 일부 노드들이 단절되기 때문에 훨씬 더 견고하게 신호에 적응한다.

         self.add(layers.Dense(Nh_l[1], activation = 'relu',input_shape = (Nin,), name = 'Hidden-2'))
         self.add(layers.Dropout(0.2))  
         self.add(layers.Dense(Nout,activation = 'softmax'))
    
   - 제2 은닉 계층을 제1 은닉 계층과 유사하게 구성한다(노드 수는 제1 은닉 계층의 노드 수와 다를 수 있다).

   - 이번에는 앞 계층의 노드 수를 적지 않았다. **제2 은닉 계층부터는 케라스가 자동으로 현재 계층의 입력 노드 수룰 앞에 나온 은닉 계층 출력 수로 설정해주기 때문이다.**

이제 모델 컴파일 할 단계이다.

    self.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
   - 분류할 클래스 수가 2개 이상이므로 loss를 categorical_crossentropy로 설정하였고 최적화는 adam방식을 사용하였다.

### 3.2.3 데이터 준비 ###

3. 분류 DNN을 위한 데이터 준비

    - 2.2.2절 '분류 ANN에 사용할 데이터 불러오기'와 동일하다.

### 3.2.4 학습 및 성능 평가 ###

4. 학습 및 성능 평가

    - 학습과 성능 평가를 수행하는 것은 은닉 계층이 늘어도 분류 ANN의 구현과 같다.

    - 이번 예제의 경우 ANN의 결과와 거의 유사하다. MNIST의 경우 비교적 이미지가 간단하기 때문에 ANN과 DNN의 성능 차이가 거의 없다.

## 3.3 컬러 이미지를 분류하는 DNN 구현 ##

필기체보다 복잡도가 높은 컬러 이미지들을 DNN으로 분류해보자.

1) CIFAR-10 데이터 소개
2) 데이터 불러오기
3) DNN 모델링
4) 학습 효과 분석 준비
5) DNN 학습 및 성능 평가

### 3.3.1 CIFAR-10 데이터 소개 ###

CIFAR-10 데이터셋은 다음과 같이 10가지 사물이 담긴 컬러 이미지이다.

총 6만 장이며, 이 중 5만 장은 학습용이고 1만 장은 평가용이다. 

한 사진의 크기는 32 x 32이다. 

### 3.3.2 데이터 불러오기 ###

데이터 불러오는 데 필요한 패키지는 MNIST 필기체 때와 같다.

    import numpy as np
    from keras import datasets
    from keras.utils import np_utils
    
이제 데이터를 불러올 차례이다. 재사용성을 고려하여 데이터 불러오기 코드를 함수로 만들고 CIFAR-10컬러 이미지 데이터를 불러온다.

    def Data_func():
      (X_train,y_train), (X_test, y_test) = datasets.cifar10.load_data()
      
1차원으로 구성된 목푯값 배열들인 y_train, y_test는 MNIST와 동일하게 10가지 클래스로 구분된 2차원 배열로 변환해준다.

     Y_train = np_utils.to_categorical(y_train)   # np_utils.to_categorical은 원핫인코딩 해준다고 생각하면 될듯
     Y_test = np_utils.to_categorical(y_test)

   - 분류 DNN의 출력은 원소 10개로 구성된 이진 벡터이다.

   - 일반적으로 분류 방식은 각 클래스마다 출력 노드가 하나이다. 출력값은 주어진 입력 이미지에 대해 각 클래스에 해당될 가능성을 나타낸다.

   - CIFAR-10 데이터셋의 경우 클래스가 10개이므로 출력 노드 수는 10개가 된다. 반면 목푯값은 0 ~ 9까지 정숫값으로 저장되어 있기 때문에 정숫값을 np_utils.to_categorical()을 이용해 10개 원소를 가진 이진 벡터로 변환하였다.

   - 성능 분석 등을 위해 분류 DNN의 출력값 Y_train을 이진벡터에서 정수 스칼라로 역변환할 때가 있다. 이때는 y_train = np.argmax(Y_Train,axis=1)과 같이 최댓값의 아규먼트를 찾아주면 된다.

이제 **컬러값을 포함하는 이미지 배열을 DNN이 다룰 수 있도록 차원을 바꿔야한다.**

    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W * H * C)
    X_test = X_test.reshape(-1, W * H * C)
    
   - L : 데이터 수 , W : 이미지 넓이(y축), H :이미지 높이(x축), C :이미지 채널 수 
    
   - DNN은 벡터 형태의 정보를 다루기 때문에 데이터의 차원을 2로하고 첫 줄은 L로 설정하고 둘째 줄은 W * H * C를 곱한 값이 되도록 한다.

### 3.3.3 DNN 모델링 ###

DNN 모델은 비교를 위해 3.2절에서 사용한 모델을 그대로 사용한다. 

다만, 드롭아웃 확률은 함수의 아규먼트로 전달해 원하는 값으로 조정하도록 한다.

    from keras import layers,models
    
    class DNN(models.Sequential):
      def __init__(self,Nin,Nh_l,Pd_l,Nout):
        super().__init__()
        
        self.add(layers.Dense(Nh_l[0], activation = 'relu', input_shape = (Nin,), name = 'Hidden-1'))
        self.add(layers.Dropout(Pd_l[0]))
        self.add(layers.Dense(Nh_l[1], activation = 'relu', name = 'Hidden-2'))
        self.add(layers.Dropout(Pd_l[1]))
        self.add(layers.Dense(Nout, activation = 'softmax'))
        self.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        
  - 초기화 함수가 Pd_l 배열을 입력받도록 했다. Pd_l은 두 아규먼트로 드롭아웃 확률을 지정한다.

### 3.3.4 학습 효과 분석 ###

학습 효과를 분석하는 코드는 2.2절의 코드를 임포트한다.

    from ann_mnist_cl import plot_loss, plot_acc
    import matplotlib.pyplot as plt
    
### 3.3.5 학습 및 성능 평가 ###

사진 데이터를 분류하는 학습을 진행한 후 그 성능을 평가한다.

- 모델링에 필요한 파라미터들 설정
 
      Nh_l = [ 100,50 ]
      Pd_l = [ 0.0 , 0.0 ]
      number_of_class = 10
      Nout = number_of_class 
      
    - Pd_l은 새롭게 추가된 배열이다. 이번에 정의한 DNN모델은 드롭아웃을 두 번 수행한다. 각 드롭아웃의 확률을 Pd_l로 조정할 수 있다.

- 앞서 만든 Data_func()함수로 데이터 불러오기 + DNN 객체의 인스턴스 만들기

      (X_train,Y_train), (X_test,Y_Test) = Data_func()
      model = DNN(X_train.shape[1], Nh_l, Pd_l, Nout)
  
- 학습하기

      history = model.fit(X_train, Y_train, epochs = 10, batch_size = 100, validation_split=0.2)

    - 학습 진행 경과는 history 변수에 저장된다.

    - 총 100회의 학습이 진행된다.

- 평가 데이터를 활용해 최종 성능 알아보기

      performance_test = model.evaluate(X_test,Y_test,batch_size = 100)
      print('Test Loss and Accuracy ->' , performance_test)
      
- 학습이 어떻게 진행되었는지 그 과정을 그래프로 분석하기

       plot_acc(history)
       plt.show()
       plot_loss(history)
       plt.show()
       
![image](https://user-images.githubusercontent.com/66320010/120432154-7d617280-c3b4-11eb-9085-19d11820ccac.png)

위 그림은 학습에서 드롭아웃을 사용하지 않은 경우이다. 그래서 학습 데이터와 검증 데잉터 간에 성능 차이가 많았다.

둘이 유사하게 되려면 드롭아웃 값ㅇ을 조정해야한다.

Pd_l = [0.05, 0.5]로 설정하여 과적합을 줄여 학습 데이터 성능과 검증 데이터 성능이 유사하도록 만든 것을 다음 그림에서 확인할 수 있다.

![image](https://user-images.githubusercontent.com/66320010/120432289-aeda3e00-c3b4-11eb-95a5-f4a2d035d511.png)

이 경우 검증 성능은 드롭아웃을 포함하지 않은 때와 차이가 크지 않지만 학습이 오래 진행되는 경우 과적합 방지를 하지 않으면 검증 성능이 다시 나빠질 수 있기 때문에 적정한 드롭아웃 값을 설정해야한다.


