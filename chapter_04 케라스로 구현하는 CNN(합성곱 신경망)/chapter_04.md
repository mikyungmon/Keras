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
