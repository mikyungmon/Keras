# Chapter 02 # 

**인공신경망(Artificial Neural Network)** 이란? 

  - **생체의 신경망을 흉내 낸 인공지능이다.**

  - 입력, 은닉, 출력 계층으로 구성되어 있으며 은닉 계층을 한 개 이상 포함할 수 있다.

  - 초기에는 기술적인 한계로 은닉 계층을 한 개만 포함하여 전체 3개의 계층으로 ANN을 구성했다. 

  - 넓은 의미로 인공 신경망을 총칭하는 용어로 ANN을 사용하기도해서 단일 은닉 계층의 ANN을 얉은 신경망으로 구분해서 부르기도 한다.
   
   
## 2.1 ANN의 원리 ## 

### 2.1.1 ANN의 개념 ### 

ANN은 생체 신경망 구조와 유사하게 은닉 계층을 포함하는 인공신경망 기술이다. 

초기에는 은닉 계층이 하나였다. 은닉 계층이 둘 이상이면 가중치 최적화가 어려워 신경망의 성능을 보장할 수 없었기 때문이다.

과거에 가중치 최적화에 대한 연구가 활발하지 못했던 이유로는 충분하지 못한 데이터양도 있다. 빅데이터 시대 이전에는 데이터 양이 많지 않았기 때문이다.

그리고 스몰데이터에는 신경망 대신할 다신 머신러닝 방법이 존재했다. 예를 들어 SVM은 데이터가 적을 때 아주 우수한 성능을 보인다.

그렇지만 처리할 데이터양이 늘어나거나 비정형 데이터인 경우는 복잡도가 높아서 활용하기가 쉽지 않다 => 이런 경우 **ANN이 더 효과적이다.**

### 2.1.2 ANN 구조 ### 

은닉 계층이 하나인 ANN으로 기본 구조를 살펴보자.

ANN은 **입력계층, 은닉계층, 출력계층**으로 구성되고 각 계층은 순서별로 여러 입력노드, 은닉노드, 출력노드를 포함한다.

< 동작 순서 >

1. 입력 계층을 통해 외부로부터 들어온 입력 신호 벡터(x)에 ㅈ가중치 행렬 W_xh를 곱하여 은닉 계층으로 보낸다.

2. 은닉 계층의 각 노드들은 자신에게 입력된 신호 벡터에 **활성화 함수**인 f_n()를 적용한 결과 벡터(h)로 내보낸다. 활성화 함수에는 시그모이드, 하이퍼볼릭탄센트 함수 등이 있다.

3. 은닉 계층의 결과 벡터에 새로운 가중치 행렬 W_hy를 곱한 뒤 출력 계층으로 보낸다.

4. 출력 계층으로 들어온 신호 벡터에 출력 활성화 함수인 f_y()를 적용하고 그 결과 벡터(y)를 신경망 외부로 최종 출력한다. 분류의 경우에는 출력용 활성화 함수로 **소프트맥스** 연산을 주로 사용한다.

![image](https://user-images.githubusercontent.com/66320010/119773060-10a13080-befb-11eb-9f3f-d0251d8f2a89.png)

### 2.1.3 ANN 활용 ### 

ANN의 기본적인 활용 방법은 **분류**와 **회귀**로 나눌 수 있다. 

  - 분류 ANN : 입력 정보를 클래스 별로 분류하는 방식 -> ex) 필기체 숫자를 0부터 9로 분류하기
  - 회귀 ANN : 입력 정보로 다른 값을 예측하는 방식 -> ex) 최근 일주일간 평균 온도로 내일 평균 온도 예측하기

#### [ 분류 ANN ] ####

분류 ANN이란?

- 입력 정보를 바탕으로 해당 입력이 어느 클래스에 속하는지를 결정하는 것

- 예를 들어 필기체 숫자가 적힌 그림을 보고 어느 숫자인지 분류하는 경우

- 입력 계층은 필기체 숫자를 받아들이고 출력 계층은 분류한 결과를 출력한다.

- 분류 결과를 출력하는 방법으로 노드 하나를 사용해 분류 결과를 수로 표현하는 방법이 있다. 그러나 분류 ANN은 숫자로 출력하는 방법보다 분류할 클래스 수만큼 출력 노드를 만드는 방법이 효과적이라고 알려져 있다.

- 따라서 만약 0과 1 두 숫자에 해당하는 필기체 그림을 분류하는 경우에는 출력 노드를 두 개 만드는 것이 효과적이다.

- 판별은 두 출력 노드의 값을 비교하여 더 큰 쪽을 선택하도록 구현한다.

**이와 같이 ANN을 이용하여 예측값을 추론하는 과정을 전방향 예측이라고 한다. 반면 ANN을 구성하는 가중치의 학습은 예측값의 목표값에 대한 오차를 역방향으로 되돌리면서 이루어지기 때문에 오차역전파라고 한다.**

오차 역전파 : 오차를 줄이는 경사 하강법(gradient descent)에서 유도된 방법이다. 경사 하강법은 가중치에 대한 손실 함수를 미분하고 그 미분값의 방향과 크기를 사용해 가중치를 보상하는 방법이다.

손실 함수(loss function) : 가중치에 따라 오차가 얼마나 커지는지 작아지는지를 평가한다. 

분류 ANN은 손실함수로 교차 엔트로피(cross entropy)함수를 주로 사용한다. 교차 엔트로피 함수를 적용하려면 출력 노드의 결과를 확률 값으로 바꿔야 한다. 확률 값은 출력 노드 값을 소프트맥스 연산으로 구한다.

#### [ 회귀 ANN ] ####

회귀 ANN이란?

- 입력값으로부터 출력값을 직접 예측하는 방법이다.

- 실제 데이터가 분포해 있다고 할 때 이 데이터의 규칙을 잘 표현하는 함수를 찾는 것이 회귀이다.

- 예를 들어 다중 회귀를 이용하면 집과 관련된 정보를 활용해서 집값을 예측할 수 있다. 

- 신경망 학습은 많은 집 정보를 이용하여 수행하게 된다. 

- 신경망 학습이 완료되면 집에 대한 정보와 시세 간의 관계가 성립된다. 즉, 임의의 집정보를 넣으면 시세를 예측할 수 있다. 물론 학습에 사용하지 않은 집에 대한 정보로도 시세를 예측할 수 있다.

- ANN학습은 분류 ANN에서와 같이 **오차역전파 방법을 사용한다.**

- 회귀 ANN을 오차역전파 방법으로 학습시키려면 주로 평균제곱오차(mean-square error)를 손실함수로 사용한다.

*오차 역전파를 활용해 학습하는 최적화 방법으로 확률적 경사 하강법을 많이 사용해왔다. 최근에는 더 발전된 방법으로 Adam, Adagrad, RMSprop 등과 같은 방법을 사용한다.*

*다양한 최적화 방법 중에 주어진 데이터에 맞는 하나를 선택해야 한다.*

### 2.1.4 ANN 구현 방법 및 단계 ### 

인공지능을 케라스로 구현하는 방법은 크게 **함수형 구현**과 **객체지향형 구현**이 있다. 

  - 함수형 구현 : ANN모델을 직접 설계하는 인공지능 전문가에게 적합
  - 객체 지향방식 구현: 전문가가 만들어놓은 ANN모델을 사용하는 사용자에게 적합

ANN모델링은 분산 방식, 연쇄 방식 또는 둘의 혼합 방식으로 구현할 수 있다. 

분산 방식은 구조가 복잡한 경웨 적합하며 연쇄 방식은 하나의 순서로 구성된 간단한 신경망의 구현에 적합하다.

< 구현 순서 >

1. 인공지능 구현용 패키지 불러오기
2. 인공지능에 필요한 파라미터 설정
3. 인공지능 모델 구현
4. 학습과 성능 평가용 데이터 불러오기
5. 인공지능 학습 및 성능평가
6. 인공지능 학습 결과 분석

## 2.2 필기체를 구분하는 분류 ANN 구현 ##

케라스로 분류 ANN을 구현하면서 기본적인 인공지능 구현 방법을 익혀보자.

또한 학습이나 평가에 사용할 데이터를 인공지능 알고리즘에 적용하기 전에 어떻게 전처리하는지도 알아보자.

분류 ANN 구현도 앞에서 설명한 인공지능 구현 6단계를 따른다.

1. 분류 ANN 구현용 패키지 불러오기
2. 분류 ANN에 필요한 파라미터 설정
3. 분류 ANN 모델 구현
4. 학습과 성능 평가용 데이터 불러오기
5. 분류 ANN 학습 및 검증
6. 분류 ANN 학습 결과 분석

### 2.2.1 분류 ANN을 위한 인공지능 모델 구현 ### 

1. 케라스 패키지로 2가지 모듈을 불러온다.

       from keras import layers, models
        
    - layers : 각 계층을 만드는 모듈
    - models : 각 layer들을 연결하여 신경망 모델을 만든 후, 컴파일하고 , 학습시키는 역할
    - 객체 지향 방식을 지원하는 케라스는 **models.Model 객체에서 complie() , fit(), predict(), evaluate() 등 딥러닝 처리 함수 대부분을 제공**해 편리하게 사용할 수 있다.

2. 분류 ANN에 필요한 파라미터 설정한다.

필요한 파라미터는 Nin(입력 계층의 노드 수), Nh(은닉 계층의 노드 수), number_of_class(출력값이 가질 클래스 수), Nout(출력 노드 수)이다.

실제 이 값의 정의는 main()함수 안에서 진행한다( 꼭 전역 변수로 지정할 필요 없다면 파라미터들을 시작 함수인 main()에 넣어준다 ).


3. 모델링

분류 ANN모델은 인공지능 기술에서 가장 기본이 되는 구조이다.

연쇄 방식은 복잡도가 높은 모델에 적용하기에는 한계가 있기 때문에 분산 방식까지 알아두는 것이 좋다.

또한 모델을 구현하는 방식도 함수형과 객체지향형 방법을 모두 다룬다.

< 고려하는 모델 구현 방식 >
   
  1) 분산 방식 모델링을 포함하는 함수형 구현
  2) 연쇄 방식 모델링을 포함하는 함수형 구현
  3) 분산 방식 모델링을 포함하는 객체지향형 구현
  4) 분산 방식 모델링을 포함하는 객체지향형 구현

**[ 분산 방식 모델링을 포함하는 함수형 구현 ]**

- 입력계층 정의
  
      x = layers.Input(shape(Nin,))  # 원소 Nin개를 가지는 입력 신호 벡터
       
- 은닉 계층 정의

      h = layers.Activation('relu')(layers.Dense(Nh)(x))
       
    - 은닉 계층은 layers.Dense()로 지정한다. 노드가 Nh개인 경우에 은닉 계층을 layers.Dense(Nh)(x)로 지정한다. 참고로 layers.Dense(Nh)는 layers.Dense 객체의 인스턴스이다.

    - 객체를 함수처럼 사용할 수 있기 때문에 ()를 사용해 호출이 가능한 것이다.

    - 활성화 함수는 layers.Activation('relu')로 지정한다. 

    - **입력 벡터인 x를 완전히 연결된 은닉 계층의 노드들로 모두 보내고 은닉 계층의 각 노드들은 ReLU로 활성화 처리한 뒤에 다음 계층으로 내보낸다.**

- 출력 계층

      y = layers.Activation('softmax')(layers.Dense(Nout)(h))
      
    - 다중 클래스 분류를 ANN으로 구현하고 있으므로 출력 노드 수는 클래스 수 (Nout)으로 지정한다. 

    - 이 때 **출력 노드에 입력되는 정보는 은닉 노드의 출력값이다.**

- 인공지능 모델 만들기
  
     - 앞서 설명한 계층들을 합쳐 인공지능 모델을 만든다.
     
     - 케라스는 컴파일을 수행하여 타깃 플래폼에 맞게 딥러닝 코드를 구성한다.

     - 컴파일(모델 파라미터를 통해 모델 구조를 생성하는 과정) 과정
      
           model. compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
            
          loss : 손실함수 / 
          optimizer : 최적화 함수 / 
          metrics : 학습이나 예측이 진행될 때 성능 검증을 위해 손실 뿐 아니라 정확도도 측정하라는 의미
      
      
**[ 연쇄 방식 모델링을 포함하는 함수형 구현 ]**      

분산 방식과 다르게 모델을 먼저 설정한다.

- 모델 설정
        
      model = models.Sequential()
          
  - 연쇄 방식은 모델 구조를 정의하기 전에 Sequential()함수로 모델을 초기화 해야한다.

- 모델의 구조 설정

      model.add(layers.Dense(Nh, activation = 'relu', input_shape = (Nin,))
      model.add(layers.Dense(Nout, activation = 'softmax')
        
  - 첫 번째 add() 단계에서 입력 계층과 은닉 계층의 형태가 동시에 정해진다. 

  - **입력노드 Nin개는 완전 연결 계층 Nh개로 구성된 은닉 계층으로 보내지며 이 은닉 계층 노드들은 ReLU활성화 함수로 사용한다.**

  - **은닉 계층의 출력은 출력이 Nout개인 출력노드로 보내진다. 출력 노드들의 활성화 함수는 소프트맥스 연산으로 지정했다.**
      
  - 이처럼 연쇄 방식은 추가되는 계층을 기술할 때 간편하게 기술할 수 있다는 장점이 있다(add()를 이용해 연속되는 계층을 계속 더해주면 된다).

  - 복잡한 인공신경망을 기술하는 부분은 연쇄 모델링만으로 구현이 힘든 경우도 존재 => **분산 방식 모델링**을 사용해야한다.
      
**[ 분산 방식 모델링을 포함하는 객체지향형 구현 ]**        

ANN 코드의 재사용성을 높이기 위해 객체지향 방식으로 구현할 수도 있다.

먼저 클래스를 만들고 models.Model로부터 특성을 상속해온다. models.Model은 신경망에서 사용하는 학습, 예측, 평가와 같은 다양한 함수를 제공한다.

- 클래스의 초기화 함수를 정의

      class ANN(models.Model):
        def __init__(self,Nin,Nh,Nout):
          
  - 초기화 함수는 입력 계층, 은닉 계층, 출력 계층의 노드 수를 각각 Nin,Nh,Nout으로 받는다. 

- 신경망 모델에 사용할 계층을 정의

      hidden = layers.Dense(Nh)
        
  - 은닉 계층이 하나이므로 은닉 계층 출력 변수로 hidden하나만 사용했다.

  - 은닉 계층이 만약 셋이라면 반복문을 사용해서 생성할 수 있다.

  - 이 때는 각 계층마다 노드 수가 Nh_1 = [5,10,5]라고 한다. 은닉 계층들을 hidden_1 = map(layers.Dense,Nh_1)과 같이 반복문을 사용해서 만들 수 있다.

  - for문을 사용해서 hidden_1 = [layers.Dense(n) for n in Nh_1]과 같이 만들 수 있다.

  - 단일 문장이 아니라 여러 문장으로 구성한다면 다음과 같이 구성할 수 있다.

        hidden_1 = []
        for n in Nh_1
          hidden_1.append(layers.Dense(n))
 
- 이제 노드 수가 Nout개인 출력 계층 정의

      output = layers.Dense(Nout)
  
- activation함수를 정의

      relu = layers.Activation('relu')
      softmax = layers.Activation('softmax')
        
- 상속 받은 부모 클래스의 초기화를 진행
  
      super().__init__(x,y)
        
  - 여기서 부모 클래스가 models.Model이다. 적어주지 않으면 자동으로 선정된다.

- 모델 사용하기 위해 인스턴스 생성

      model = ANN(Nin,Nh,Nout)
      
**케라스를 이용하면 이렇게 만들어진 모델을 불러와서 사용하면 되므로 복잡한 인공지능 수식을 일일이 파악해야하는 수고가 줄어든다. 따라서 클래스를 사용하는 객체지향형 구현 방식을 이용하면 케라스를 더욱 쉽게 사용할 수 있다.**
      
      
**[ 연쇄 방식 모델링을 포함하는 객체지향형 구현 ]**        
      
앞에서는 Models.Model에서 모델을 상속 받았지만 이번에는 models.Sequetial에서 상속 받는다.
 
직접 모델링 방법에서는 컴파일하기 직전에 초기화했는데 연쇄 방식에서는 부모 클래스의 초기화 함수를 자신의 초기화 함수 가장 앞 단에서 부른다.
 
    class ANN(models.Sequential):
      def __init__(self,Nin,Nh,Nout):
        super().__init__()
            
모델링은 앞의 계층에 새로운 층을 추가하는 형태이다. 연쇄 방법은 입력 계층을 별도로 정의하지 않고 은닉 계층부터 추가해나간다.
 
    self.add(layers.Dense(Nh,activation = 'relu', input_shape = (Nin,)))
      
  - 은닉 계층을 붙일 때, 변수 중 하나로 입력 계층 노드 수를 input_shape = (Nin,)과 같이 포함해주어 간단한 입력 계층 정의하였다.

- 출력계층 정의

      self.add(layers.Dense(Nout,),activation = 'softmax')
      
### 2.2.2 분류 ANN에 사용할 데이터 불러오기 ###       
      
- 케라스는 자주 쓰는 데이터셋을 쉽게 불러오는 라이브러리를 제공한다.

- 여기서는 MNIST 데이터셋을 사용한다(MNIST는 6만 건의 필기체 숫자를 모은 공개 데이터이다).

< 데이터 불러오고 전처리 하는 단계 >

  1. 데이터 처리에 사용할 패키지 임포트
  2. 데이터 불러오기
  3. 출력값 변수를 이진 벡터 형태로 바꾸기
  4. 이미지를 나타내는 아규먼트를 1차원 벡터 형태로 바꾸기
  5. ANN을 위해 입력값들을 정규화하기
      
- 데이터 불러오는 라이브러리 가져오기

      import numpy as np  # reshape() 사용하고자 가져옴
      from keras import datasets  # mnist
      from keras.utils import np_utils  # to_categorical
      
- MNIST 데이터를 불러와서 변수에 저장

      (x_train,y_train), (x_test,y_test) = datasets.mnist.load_data()
         
  - 학습에 사용하는 데이터는 6만 개이고, 성능 평가에 사용하는 데이터는 1만 개이다.
      
- 0부터 9까지 숫자로 구성된 출력값을 0과 1로 표현하는 벡터 10개로 바꾸기
    
      Y_train = np.utils.to_categorical(y_train)
      Y_test = np.utils.to_categorical(y_test)
  
  - **이렇게 전환하는 이유는 ANN을 이용한 분류 작업 시 정수보다 이진 벡터로 출력 변수를 구성하는 것이 효율적이기 때문이다.**
      
- x_train, x_test에 (x,y)축에 대한 픽셀 정보가 들어가 있는 3차원 데이터인데 실제 학습 및 평가용 이미지를 2차원으로 조정

      L,M,H = x_train.shape
      x_train = x_train.reshape(-1, W*H)   # reshape에서 -1은 행렬의 행을 자동으로 설정하게 만듦
      x_test = x_test.reshape(-1, W*H)
      
  - L : 학습 데이터셋에 있는 샘플 수 
  - 따라서 L * W * H 와 같은 모양의 텐서로 저장되어있음     

- ANN 최적화를 위해 아규먼트를 정규화

      x_train = x_train / 255.0
      x_test = x_test / 255.0
      
  - 여기서는 0 ~ 255 사이의 정수로 입력된 입력값을 255로 나누어 0 ~ 1 사이의 실수로 바꾼다. 

**여기까지 과정이 진행되면 학습을 진행하는 데 필요한 데이터가 준비된 것이다.**     
      
### 2.2.3 분류 ANN 학습 결과 그래프 구현 ###         
      
- 학습 결과 분석은 학습이 진행되는 동안 측정한 솔실과 정확도의 추이를 관찰하여 이루어진다.

- 이 값들은 fit()함수의 결과인 history변수에 저장되어있다(2.2.5절에서 다룰 내용).

- 우선 history에 들어있는 손실과 정확도 추이를 시각적으로 표현하는 두 함수를 만들어보자.

  - 우선 그래프를 그리는 라이브러리 plt를 불러오기

        import matplotlib.pyplot as plt
   
   - 이 라이브러리에서 활용할 함수는 plt.plot(), plt.title(), plt.xlabel(), plt.ylabel(), plt.legend()이다
   
     plt.plot() : 선 그리기
     
     plt.title() : 그래프 제목 표시
     
     plt.xlabel() : x축 이름 표시
     
     plt.ylabel() : y축 이름 표시
     
     plt.legend() : 각 라인의 표식 표시
  
- 손실 그리는 함수

      def plot_loss(history):
        plt.plot(history.history['loss'])   # 실제 데이터로 구한 손실 값
        plt.plot(history.history['val_loss'])  # 검증 데이터로 구한 손실 값
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','Test'], loc = 0)
      
  - plt.plot()을 이용하여 history에 들어있는 손실값을 그린다.
      
- 정확도를 그리는 함수

      def plot_acc(history):
        plt.plot(history.history['acc'])   # 실제 데이터로 구한 정확도
        plt.plot(history.history['val_acc'])  # 검증 데이터로 구한 정확도
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train','Test'], loc = 0)
        
### 2.2.4 분류 ANN 학습 및 성능 분석 ###         
        
- ANN에 사용할 파라미터 정의

      def main() :
        Nin = 784   
        Nh = 100
        number_of_class = 10
        Nout = number_of_class
   
- 앞서 만들어두었던 모델의 인스턴스를 만들고 데이터 불러오기

      model = ANN_seq_class(Nin,Nh,Nout)
      (x_train,y_train),(x_test,y_test) = Data_func()
      
 - 만들어진 인스턴스와 불러온 데이터를 이용해 모델 학습하기

  - **학습은 모델에 fit()함수를 이용하여 데이터를 제공하여 진행한다.**
    
        history = model.fit(x_train,y_train, epochs = 5, batch_size = 100, validation_split = 0.2)
        
    - 학습 진행 사항을 변수 history에 저장한다.

    - 파라미터

        x_train, y_train : 학습에 이용할 입력 데이터와 출력 레이블

        epochs : 총 반복할 에포크 지정
  
        batch_size : 한 데이터를 얼마씩 나눠서 넣을지 지정하는 값
  
        validation_split : 전체 학습 데이터 중에서 학습 진행 중 성능 검증에 데이터를 얼마나 사용할지를 결정하는 변수 ( 이 예제에서는 학습데이터의 20%를 성능 검증에 활용 )
  
- 학습이나 검증에 사용되지 않은 데이터 (x_test, y_test)로 성능을 최종 평가한 결과 

      performance_test = model.evaluate(x_test, y_test, batch_size = 100)
      print('Test Loss and Accuracy -> '{:.2f},{:.2f}'.format(*performance_test))
        
- 손실과 정확도의 추이를 그림으로 그려보기

      plot_loss(history)
      plt.show()
      
      plot_acc(history)
      plt.show()
        
  - 이 함수들을 실행하면 손실 학습 곡선과 정확도 곡선이 출력된다.
  
  - 과적합 방지 방법으로는 조기 종료나 모델에 사용된 파라미터 수를 줄이는 방법이 있다.

## 2.3 시계열 데이터를 예측하는 회귀 ANN 구현 ##        
  
보스턴 집값을 예측하는 회귀 ANN을 구현해보자.

< 구현 절차 >

1. 회귀 ANN 구현
2. 학습과 평가용 데이터 불러오기
3. 회귀 ANN 학습 및 성능 평가
4. 회귀 ANN학습 결과 분석
        
### 2.3.1 회귀 ANN 모델링 ###        
        
회귀 모델을 구현해보자.

- 케라스 패키지에 들어 있는 서브패키지 layers와 models를 임포트

      from keras import layers, models
      
  - layers와 models는 각각 계층을 구성하는 툴들과 계층을 합쳐 하나의 모델로 만드는 툴이 들어있는 케라스 서브패키지이다.

- 클래스를 만들고 클래스 생성사 함수에 사용될 신경망 계층을 정의

      class ANN(models.Model):
        def __init__(self,Nin,Nh,Nout):
          hidden = layers.Dense(Nh)
          output = layers.Dense(Nout)
          relu = layers.Activation('relu')

- ANN의 각 계층의 신호 연결 상황 정의

      x = layers.Input(shape = (Nin,))    # numpy 라이브러리는 열 벡터 모양을 (Nin,)와 같이 표현
      h = relu(hidden(x))
      y = output(h)
        
  - x : Nin 길이를 가지는 1차원 열 벡터
  - 출력은 활성화 함 수 없이 바로 y로 나온다(회귀에서는 통상적으로 출력 노드에 활성화 함수를 사용하지 않는다).

- 입력과 출력을 이용해 모델을 만들고 만들어진 모델을 사용하도록 컴파일

      super().__init__(x,y)
        self.compile(loss = 'mse', optimizer = 'sgd')

### 2.3.2 학습과 평가용 데이터 불러오기 ###   

데이터셋 Boston housing에는 총 506건의 보스턴 집값과 관련된 13가지 정보가 담겨져 있다.

- 데이터를 불러와 딥러닝 이전에 전처리 하는 응용 패키지 불러오기

      from keras import datasets   
      from sklearn import preprocessing
      
  - datasets은 MNIST, Boston housing 등 잘 알려진 머신러닝 공개 데이터를 자동으로 불러오는 패키지

  - sklearn.preprocessing은 머신러닝에 사용되는 패키지. sklearn은 여러 패키지를 포함하는 컨테이너 패키지

- 앞서 말한 Boston housing 관련 데이터를 가져온 뒤 데이터 정규화 진행

      (x_train,y_train),(x_test,y_test) = datasets.boston_housing.load_data()
      scaler = preprocessing.MinMaxScaler()
      x_train = scaler.fit_transform(x_train)
      x_test = scaler.transform(x_test)

  - MinMaxScaler는 최곳값 최젓값을 1과 0으로 정규화해주는 함수이다.
  
  - 이 스케일러의 객체를 인스턴스로 생성한 뒤에 x_train으로 학습과 변환을 한 뒤 x_test를 변환시킨다.


### 2.3.3 회귀 ANN 학습 결과 그래프 구현 ###  

앞에서 그린 학습 결과를 표시하는 그래프 코드를 그대로 사용한다. 

      plot_loss(history)
      plt.show()
      
      plot_acc(history)
      plt.show()

### 2.3.4 회귀 ANN 학습 및 성능 분석 ###

    def main() :
      Nin = 13   
      Nh = 5
      Nout = 1
      
  - 여기서는 분류가 아니라 회귀를 통해 결괏값을 직접 예측하기 때문에 출력 계층의 길이 (Nout)을 1로 설정하였다.

- 앞에서 구현한 회귀 ANN 모델의 인스턴스를 생성하고 적용할 데이터를 불러오기 + 불러온 데이터를 이용해 생성된 모델의 인스턴스 학습

      model = ANN(Nin,Nh,Nout)
      (x_train,y_train),(x_test,y_test) = Data_func()
      history = model.fit(x_train,y_train,epochs = 100, batch_size = 100, validation_split = 0.2, verbose = 2)
      
  - model의 멤버 함수 fit()으로 학습한다.

- 학습이 끝난 신경망으로 성능을 평가

성능 평가에는 model.evaluate()를 사용한다.

    performance_test = model.evaluate(x_test,y_test, batch_size = 100)
    print('\nTest Loss -> '{:2f}'.format(performance_test))
