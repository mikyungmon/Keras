# Chapter 04 #

**합성곱 신경망(CNN)이란?**

- 영상 처리에 많이 활용되는 합성곱을 이용하는 인공신경망 기술이다.

- CNN은 합성곱을 이용해 가중치 수를 줄여 연산량을 줄이면서도 이미지 처리를 효과적으로 할 수 있다 => 이를 통해 이미지의 특징점을 효율적으로 찾을 수 있어 인공신경망 효능을 더 높일 수 있다.

## 4.1 CNN 원리 ##

CNN은 합성곱 필터(convolution filter)를 이용하여 신경망 동작을 수행한다. 

여러 작은 필터가 이미지 위를 돌아다니면서 특징점들을 찾아 그 합성곱 결과를 다음 계층으로 보낸다. 

적은 수의 가중치로 이미지 처리를 효율적으로 할 수 있다.

![image](https://user-images.githubusercontent.com/66320010/120619397-e5888500-c496-11eb-9d48-caa38244e810.png)

CNN은 위 그림과 같이 주로 입력 부근 계층들의 합성곱 계층으로 구성하고 출력 부근 계층들을 완전 연결 계층으로 구성한다.

CNN을 구성하는 계층 중에 **합성곱 계층**들은 특징점을 효과적으로 찾는 데 활용되고, **완전 연결 계층**들은 찾은 특징점을 기반으로 이미지를 분류하는데 주로 활용된다.

CNN은 2차원이나 그 이상 차원의 데이터 처리에 적합하다. CNN은 2차원 합성곱으로 각 노드를 처리하기 때문에 이미지에 더 적합하다.

**즉,CNN은 이미지의 높이와 넓이를 생각하면서 2차원 처리를 수행한다.**

그리고 컬러 이미지를 다룰 경우 컬러에 대한 계층은 **깊이**라는 별도의 차원으로 관리한다.

또한 CNN은 합성곱 계층이 끝나면 아래 그림과 같이 맥스풀링(max-pooling)계층을 이용해 각 지역별로 최댓값을 찾아준다 => 이렇게 하면 특징점 위치가 약간씩 달라져도 딥러닝을 제대로 수행한다.

![image](https://user-images.githubusercontent.com/66320010/120620062-8d9e4e00-c497-11eb-88b1-f87fa040f97b.png)

## 4.2 필기체를 분류하는 CNN 구현 ##

2,3장에서 다룬 필기체 데이터를 CNN으로 분류한다.

다음과 같은 순서로 구현한다.

1) 분류 CNN 모델링
2) 분류 CNN을 위한 데이터 준비
3) 학습 효과 분석
4) 분류 CNN 학습 및 성능 평가

### 4.2.1 분류 CNN 모델링 ###

1. 합성곱 계층들과 완전 연결 계층들이 결합하여 구성된 분류 CNN 모델링하기

    - 모델링에 필요한 케라스 패키지들을 불러온다.

          import keras

    - 케라스 구현에 필요한 계층과 모델 방식에 대한 서브패키지를 불러온다.      
      
          from keras import models,layers 
 
    - models라는 서브패키지에 들어있는 연쇄 방식 모델링 객체인 Sequential 을 사용한다.
  
    - layers라는 패키지에서는 **Dense, Dropout, Flatten, Conv2D, MaxPooling2D**를 사용한다.

        - **Conv2D** : 2차원 합성곱을 계산하는 클래스 
        - **MaxPooling2D** : 2차원 맥스풀링을 계산하는 클래스 
        - **Flatten** : 다차원의 입력을 1차원의 입력으로 변환하는 클래스

   - 사용할 계층들을 임포트에서 from keras.layers imoort Conv2D처럼 구체적으로 지정하면 layers.Conv2D()를 Conv2D로 줄여서 쓸 수 있다.

   - 케라스의 강력한 기능 중에 하나인 딥러닝 엔진들의 함수를 직접 호출하거나 주요 파라미터를 제어하는 서브 패키지 임포트한다.

         from keras import backend
         
       - backend 서브패키지를 사용하면 딥러닝 엔진을 직접 제어할 수 있고 엔진에서 사용하는 시스템 파라미터값들을 참조하거나 변경할 수 있다.

   - CNN객체를 models.Sequential로 상속하여 연쇄방식으로 모델을 구현한다.

         class CNN(self.Sequential):
           def __init__(self,input_shape, num_classes) : 
             super().__init__()
          
       - 여기서 super()는 기본 특성을 상속한 부모 함수를 부른다 => 따라서 super.__init__()은 models.Sequential.__init__(self)와 동일하게 동작한다.

   - 이제 CNN의 은닉 계층을 정의할 시점이다.
   
         self.add(layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape)
          
       - 입력 계층의 형태는 객체의 초기화 변수인 input_shape값을 따른다. 이는 초기화 함수의 입력값이므로 모델의 인스턴스 만들 때 정해진다.

       - 합성곱 계층은 완전 연결 계층과 달리 input_shape이 2차원 이미지들로 구성된다. RGB로 여러 색상을 표현하기 때문에 채널을 포함하여 input_shape은 길이가 3인 리스트이다.   

       - 이 계층의 합성곱 형태는 크기가 3 x 3인 커널 32개로 구성된다.

    - 두 번째 CNN계층은 첫 번째 계층과 유사하지만 2가지 기능이 더 추가되었다.

          self.add(layers.Conv2D(64, kernel_size = (3,3), activation = 'relu')
          self.add(layers.MaxPooling2D(pool_size = (2,2))
          self.add(layers.Dropout(0.25))
          
    - 다음 부속 계층은 Flatten()처리로 입력을 벡터로 바꾼다. 합성곱은 이미지 형태로 프로세스가 진행되지만 완전 연결 계층은 이미지가 벡터로 통합한 뒤 진행된다.

          self.add(layers.Flatten())
          
       - 완전 연결 계층으로 구성된 은닉 계층 1개와 출력 계층 1개를 포함한다.

    - 은닉 계층 길이가 128이고 ReLU를 활성화 함수로 사용. 이후 50% 확률로 드롭아웃을 수행 

          self.add(layers.Dense(128, activation = 'relu'))
          self.add(layers.Dropout(0.5))
          self.add(layers.Dense(num_classes, activation = 'softmax'))
          
        - 완전 연결 계층 둘로 구성된 DNN 계층이 필요한 이유는 실질적 분류 작업을 하기 위해서이다.
        
    - CNN 모델 구성되고 나면 컴파일

          self.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizer,Adadelta(), metrics = ['accuracy'])
          
### 4.2.2 분류 CNN을 위한 데이터 준비 ###  
 
2. 분류 CNN에 사용할 데이터는 앞선 장들에 나왔던 MNIST이다.

  - 데이터셋 내려받으려면 datasets라는 서브패키지를 호출

        from keras import datasets
        
        (x_train,y_train), (x_test,y_test) = datasets.mnist.load_data()
        
  - datasets의 함수로 불러온 데이터를 CNN에서 쓸 수 있게 전처리하기
        
     - CNN은 DNN에서의 데이터 준비 과정과 2가지 면에서 다르다.
    
      1) 이미지를 벡터화하지 않고 그대로 사용 => 따라서 2차원의 이미지를 1차원 벡터로 변환하지 않는다.
      
      2) 흑백 이미지의 채널 정보를 처리하려면 추가적인 차원을 이미지 데이터에 포함해야한다. 컬러이미지는 RGB 색상을 다루는 채널 정보가 이미지 데이터에 이미 포함되어 있어 이미지를 나타내는 각 입력 데이터가 3차원으로 구성된다. 그러나 흑백 이미지는 채널 정보가 존재하지 않아서 입력 데이터의 차원을 하나 더 추가한다. 

     - 흑백 이미지 데이터에 채널 정보 추가하는 방법
     
        채널은 이미지 배열의 앞 단에 추가되어야 할 수도 있고 뒷 단에 추가되어야할 수도 있다. 
     
        채널의 위치는 케라스의 시스템 파라미터인 'image_data_format'에 지정되어있다. 따라서 먼저 이 파라미터를 확인한다.
     
            if backend.image_data_format() == 'channels_first'
     
        이 경우에는 이미지 배열의 앞 단에 존재한다. **앞 단에 존재할 때는 이미지 열과 행 앞에 채널에 해당하는 차원을 위치시킨다.**

           x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)    # 파라미터 - 샘플 수, 채널수, 이미지 가로 길이, 이미지 세로 길이
           x_test = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
           input_shape = (1, img_rows,img_cols)  # 맨 앞에 채널 수 표시

        만약 케라스 파라미터 image_data_format이 'channel_first'가 아니라면 **채널 차원이 이미지 배열 다음에 위치하도록 해야한다.**
         
           x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)   
           x_test = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
           input_shape = (img_rows,img_cols,1)  # 맨 끝에 채널 수 표시

         - DNN과 달리 이미지의 가로와 세로가 구분되어 다뤄진다.

### 4.2.3 학습 효과 분석 ###

3. 학습 효과를 분석하기 위해 그래프를 그리는 기능을 임포트 

        import matplotlib.pyplot as plt

        def plot_loss(history):
            plt.plot(history.history['loss'])   # 실제 데이터로 구한 손실 값
            plt.plot(history.history['val_loss'])  # 검증 데이터로 구한 손실 값
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train','Test'], loc = 0)

        def plot_acc(history):
            plt.plot(history.history['acc'])   # 실제 데이터로 구한 정확도
            plt.plot(history.history['val_acc'])  # 검증 데이터로 구한 정확도
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train','Test'], loc = 0)


### 4.2.4 분류 CNN 학습 및 성능 평가 ###

4. 데이터와 모델이 준비되었으니 이것들을 이요해 학습과 성능 평가를 진행

  - 딥러닝 학습에 사용할 파라미터 정의
  
        epochs =  10
        batch_size = 128
    
      - 학습을 위한 에포크는 10번이고 1회 학습 시 입력 데이터를 128개씩 나눠서 입력한다.
    
  - 데이터 객체와 모델 객체의 인스턴스를 생성

        data = DATA()
        model = CNN(data.input_shape, data_num_classes)
       
     - 준비된 데이터를 data 인스턴스에 넣어두고 모델을 model인스턴스에 저장한다.

  - 분류 CNN 학습을 진행

        model.fit(data.x_train,data.y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.2)
        
     - 모델에 학습할 데이터와 그 데이터의 레이블 정보를 같이 제공한다. 

     - validation_split = 0.2 -> **성능 검증용 데이터는 학습 데이터의 일부를 사용한다.** 검증 데이터는 학습 데이터의 일부분이므로 나중에 사용하는 평가 데이터와는 다르다.

  - 성능 평가

        score = model.evaluate(data.x_test, data.y_test)
        print('Test Loss ->', score[0])
        print('Test Accuracy ->', score[1])
        
     - **평가 데이터**는 학습이나 검증에 사용되지 않은 별도의 데이터이다 => 이를 통해 학습한 모델의 성능을 객관적으로 검증한다.

  - 학습이 잘 되었는지 학습 곡선을 그려서 확인

        plot_acc(history)
        plt.show()
        plot_loss(history)
        plt.show()

## 4.3 컬러 이미지를 분류하는 CNN 구현 ##

이 절에서는 CNN을 이용해 사진을 분류하는 방법을 다룬다. 

사진 분류와 필기체 분류는 CNN 입장에서 크게 다르지 않다(둘 다 이미지를 학습하여 인식하기 때문이다).

앞에서 다룬 필기체 이미지는 형태가 단순하고 흑백이었는데 이번에는 형태가 더 복잡한 컬러 이미지이다.

따라서 CNN의 분류 모델을 좀 더 일반화하여 흑백과 컬러 이미지를 모두 처리하는 코드를 구현한다.

여기에서 사용할 데이터는 '컬러 이미지를 분류하는 DNN 구현'에서 다룬 CIFAR-10 이다.

구현 순서는 다음과 같다.

1) 분류 CNN 패키지 임포트
2) 분류 CNN 모델링
3) 분류 CNN을 위한 데이터 준비
4) 분류 CNN의 학습 및 성능 평가를 위한 머신 클래스
5) 분류 CNN의 수행

### 4.3.1 분류 CNN 패키지 임포트 ###

1. 필요한 패키지들을 임포트한다. 

       from sklearn import model_selection,metrics
       from sklearn.preprocessing import MinMaxScaler
    
    - sklearn의 model_selection은 **인공지능 모델의 최적화에 필요한 다양한 툴을 제공한다.**
    
    - metircs는 모델 결과의 성능을 나타내는 지표를 계산하는 툴을 제공한다.
    
    - MinMaxScaler는 지정한 최댓값과 최솟값을 이용해 입력값의 크기를 조정하는 클래스이다. 이 클래스를 사용하면 입력값의 최댓값과 최솟값이 지정한 최댓값과 최솟값이 되도록 입력 데이터의 평균과 크기를 일괄 변경한다.

    유용한 기능을 제공하는 패키지도 임포트한다.

        import numpy as np  # numpy는 계산용 패키지
        import matplotlib.pyplot as plt  # 그래픽을 위한 패키지
        import os  # 파일 처리와 관련된 툴을 제공

    다음은 케라스 모델링을 위한 서브패키지들을 불러온다.

        from keras import backend as K
        from keras.utils import np_utils
        from keras.models import Model
        from keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout

### 4.3.2 분류 CNN 모델링 ###

2. 사용할 인공신경망 모델은 **LeNet**이다. LeNet신경망은 합성곱 방식 인공신경망이다. **합성곱 계층 두 개와 완전 연결 계층 하나로 구성된다.**

    - LeNet의 구조는 다음과 같다.
    
    ![image](https://user-images.githubusercontent.com/66320010/120799609-ee9e5280-c579-11eb-9f47-9fdbd34c1e47.png)
    
    - 이 모델을 구현하기 위해 CNN 클래스를 선언하고 초기화 멤버 함수를 만든다. CNN은 모델의 일종이므로 케라스의 Model 클래스를 상속해서 만든다.
    
          class CNN(Model):
              def __init__(model,nb_classes, in_shape = None):
                  model.nb_classes = nb_classes
                  model.in_shape = in_shape
                  model.build_model()
                  super().__init__(model.x,model.y)
                  model.compile()
                
      - 초기화 함수는 아규먼트 nb_classes와 in_shape의 값을 같은 이름의 멤버 변수들에 각각 저장했고, build_model()로 모델을 만들었다.
     
      - 부모 클래스의 초기화 함수는 super().__init__()와 같이 부르고 나서 구성한 모델을 컴파일 했다.
     
    - 다음은 모델을 구성하는 build_model() 멤버 함수를 만든다. 함수에서 필요한 멤버 변수를 지역 변수로 치환한다.
    
          def build_model(model):
            nb_classes = model.nb_classes
            in_shape = model.in_shape
            
    - 주어진 입력 이미지의 크기를 처리하는 입력 계층을 정의한다. 그 다음은 완전 연결 계층으로 구성된 은닉 계층 두 개를 정의한다.
    
          x = Input(in_shape)
          h = Conv2D(32,kernel_size=(3,3), activation = 'relu', input_shape = in_shape)(x)
          h = Conv2D(64, (3,3), activation = 'relu')(h)
        
      - 여기서 두 은닉 계층 모두 (3,3)크기로 구성된 합성곱 필터를 사용한다.
    
    - 이제 합성곱 계층의 처리결과를 완전 연결 계층으로 보내기 위해 3차원 텐서를 1차원 벡터로 바꾸는 Flatten 작업이 필요하다. 합성곱 계층은 3차원 데이터를 다루지만 완전 연결 계층은 1차원 데이터를 다루기 때문에 필요한 변환 작업이다.
   
          h = MaxPooling2D(pool_size = (2,2))(h)  # 입력 크기가 가로 세로 두 축으로 각각 반씩 줄어듦
          h = Dropout(0.25)(h)
          h = Flatten()(h)   # 3차원 데이터를 1차원으로 줄임

   - 여기까지가 합성곱 계층의 출력임을 변수 z_cl을 사용하여 기억해둔다.

         z_cl = h
   
     - 이렇게 해놓고 나중에 x와 z_cl사이의 모델을 만들면 추후 합성곱 계층을 지난 결과를 별도로 분석할 수 있다.
   
   - 이제 완전 연결 계층으로 구성된 은닉 계층과 출력 계층을 정의할 차례이다.

         h = Dense(128, activation = 'relu')(h)
         h = Dropout(0.5)(h)
        
   - 출력 계층으로 나가기 전의 완전 연결 층의 출력도 별도로 저장한다.

         z_fl = h
   
     - 이렇게 하면 x와 z_fl 사이의 모델을 별도 구성해 추후 환전 연결 계층의 출력을 분석하기 용이해진다. 
   
   - 출력 계층을 nb_classes에 해당하는 만큼의 노드 수로 구성하고 활성화 함수를 소프트맥스로 지정한다. 그리고 z_cl,z_fl을 이용해 부가적인 2가지 모델을 만든다.

         y = Dense(nb_classes, activation = 'softmax', name = 'preds')(h)
         model.cl_part = Model(x,z_cl)
         model.fl_part = Model(x,z_fl)
         
   - 또한 본 모델을 만들 수 있도록 입력과 출력을 멤버 변수로 정의한다.

         model.x , model.y = x,y
   
### 4.3.3 분류 CNN을 위한 데이터 준비 ###   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
