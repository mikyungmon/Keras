# Chapter 05 #

:bulb: **순환신경망(recurrent neural network)이란?**

  - 계층의 출력이 순환하는 인공신경망이다.

  - 순환 방식은 은닉 계층의 결과가 다음 계층으로 넘어갈 뿐 아니라 자기 계층으로 다시 들어온다. 따라서 시계열 정보처럼 앞뒤 신호가 서로 상관도 있는 경우 인공신경망의 성능을 더 높일 수 있다.

## 5.1 RNN 원리 ## 

RNN의 개념과 구조를 살펴본다. 

RNN의 기본 개념을 다룬 뒤 현재 가장 많이 사용되는 RNN방식인 LSTM을 다룬다(LSTM은 long-term short term memory의 약자).

### 5.1.1 RNN의 개념과 구조 ###

![image](https://user-images.githubusercontent.com/66320010/122911824-a9a85780-d392-11eb-8790-87bac4e17e2d.png)

RNN은 신호를 순환하여 시계열 신호와 같이 상호 관계가 있는 신호를 처리하는 인공신경망이다.

그러나 이렇게 단순한 방식으로 구현하면 경우에 따라서 학습이 제대로 이루어지지 않을 수 있다.

**출력된 신호가 계속 순환하면 활성화 함수를 반복적으로 거치게 되어서 경삿값을 구하기 힘들기 때문이다.**

경삿값을 구하기 힘든 이유는 다음과 같다.

:heavy_check_mark: 경사는 학습 방향을 결정하는 연산자로서 활성화 함수와 연계된다. 활성화 함수는 신호가 커지면 포화가 되는 특성이 있기 때문에 약간만 입력값이 커져도 미분값이 매우 작아진다. 이때 순환하여 활성화 함수를 반복해서 지나다 보면 이 미분값이 0에 매우 가까워져 학습이 어렵게 되는 것이다. 따라서 이런 문제를 없애려면 곱셈보다는 덧셈을 사용해 과거의 값을 처리하는 것이 유리하다.

:heavy_check_mark: LSTM은 이런 기본 RNN의 문제점을 개선하고자 제안된 RNN의 한 방법이다.

### 5.1.2 LSTM 구조 및 동작 ###

LSTM은 다음 그림과 같이 입력 조절 벡터, 망각 벡터, 출력 조절 벡터를 이용해 입력과 출력 신호를 게이팅(gating)한다. 

여기서 게이팅은 신호의 양을 조정해주는 기법이다.

![image](https://user-images.githubusercontent.com/66320010/122911908-bfb61800-d392-11eb-9ee2-cc09933f7e3c.png)

- **입력 조절 벡터**는 입력 신호가 tanh활성화 함수의 완전 연결 계층을 거친 이후의 값을 조절한다.

- **망각 벡터**는 과거 입력의 일부를 현재 입력에 반영한다. 

- **출력 조절 벡터**는 과거의 값과 수정된 입력값을 고려해 tanh 활성화 함수로 게이팅을 수행한다.

케라스는 순환 계층의 구현에 사용하는 LSTM클래스 등을 제공한다. 

## 5.2 문장을 판별하는 LSTM 구현 ##

LSTM을 이용하여 문장의 의미를 이해하는 예제를 구현해보자.

영화 추천 데이터베이스를 이용해 같은 사람이 영화에 대한 느낌을 서술한 글과 영화가 좋은지 나쁜지 별표 등으로 판단한 결과와의 관계를 학습한다.

학습이 완료된 후 새로운 평가글을 주었을 때 인공신경망이 판별 결과를 예측하도록 만든다.

해당 예제의 구현 절차는 다음과 같다.

1) 라이브러리 임포트
2) 데이터 준비
3) 모델링
4) 학습 및 성능 평가

### 5.2.1 라이브러리 임포트 ###

:one: LSTM을 이용한 판별망 구현에 필요한 라이브러리 임포트를 먼저 살펴본다.

- 먼저 __ future__패키지에서 print_fuction을 불러온다.

      from __ future__ import print_fuction

     - 이 패키지는 파이썬 2와 파이썬 3간의 호환성을 위한 것이다. 코드의 가장 첫 부분에 이렇게 정의하고 나면 파이썬 3문법으로 pring()명령을 사용해도 파이썬 2에서도 코드를 돌릴 수 있게 된다. 
  
    :heavy_exclamation_mark: 예제들은 파이썬 3을 기반으로 한다. 이 예제들이 파이썬 2에서 돌아가도록 쉽게 고치는 방법이 있는데 그 중 하나가 __ future__패키지를 사용하는 것이다. 여기서는 패키지 기능 중 가장 많이 사용되는 print_fuction에 대해서만 간단히 설명하였다(파이썬 2에서는 print가 함수 아니라서 괄호 달지 않고 사용, 파이썬 3는 함수로 다루기 때문에 괄호가 필요).
  
 - RNN에 필요한 케라스 클래스들을 불러온다.

       from keras.preprocessing import sequence
       from keras.datasets import imdb
       from keras import layers,models
       
     - sequence는 preprocessing이 제공하는 서브패키지이다. 이 패키지는 pad_sequence()와 같은 sequence를 처리하는 함수를 제공한다.

     - models는 케라스 모델링에 사용되는 서브패키지이다.

     - layers는 인공신경망의 계층을 만드는 서브패키지이고 이 layer 아래에 들어있는 Dense,Embedding,LSTM을 사용한다. Dense는 완전 연결 계층을 만드는 클래스이고 Embedding은 단어를 의미 벡터로 바꾸는 계층에 대한 클래스이며, LSTM은 LSTM계층을 만드는 클래스이다.

### 5.2.2 데이터 준비 ###

:two: 데이터는 케라스가 제공하는 공개 데이터인 IMDB를 사용한다.

IMDB는 25,000건의 영화평과 이진화된 영화 평점 정보(추천 = 1, 미추천 = 0)를 담고 있다.

평점 정보는 별점이 많은 경우는 긍정, 그렇지 않은 경우는 부정으로 나눠진 정보이다.

이 예제는 학습을 마친 후 인공신경망이 주어진 영화평을 분석해 영화 평점 정보를 예측한다.

- 먼저 Data 클래스를 선언하고 IMDB데이터셋을 불러온다.

      class Data :
        def __init__(self,max_features = 20000, maxlen = 80):
      (x_train,y_train), (x_test,y_test) = imdb.load_data(num_words = max_features)
        
  - 서브패키지인 imdb안에 load_data()함수를 이용해 데이터를 불러온다. 불러올 때 최대 단어 빈도를 max_features값으로 제한했다.

- 일반적으로 데이터셋에 들어 있는 문장들은 길이가 다르기 때문에 LSTM이 처리하기 적합하도록 길이를 통일하는 작업을 진행한다.

      x_train = sequence.pad_sequences(x_train,maxlen = maxlen)
      x_test = sequence.pad_sequences(x_test, maxlen = maxlen)
      
    - 문장에서 maxlen이후에 있는 단어들은 케라스 서브패키지인 sequence에서 제공하는 pad_sequences()함수로 잘라낸다.
    
    - 여기서는 최대 길이를 80으로 설정했다. 문장 길이가 maxlen보다 작으면 부족한 부분을 0으로 채운다. value라는 키워드 아규먼트로 채우는 값을 설정할 수 있다.

### 5.2.3 모델링 ###

:three: LSTM 모델링을 위한 클래스를 선언한다.

    class RNN_LSTM(models.Model):   # 모델링은 models.Model 클래스를 상속하여 만듦
      def __init__(self,max_features, maxlen)
       
먼저 입력층을 만들고 다음으로 임베딩 계층을 포함한다.      
  
    model = Sequential()
    model.add(Embedding(max_features,128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1,activation = 'sigmoid'))
         
최대 특징 점수는 max_features에 정의된 20,000으로 설정했고 임베딩 후 출력 벡터 크기를 128로 설정했다.
      
    x = layers.Input((maxlen,))
    h = layers.Embedding(max_features, 128)(x)

입력에 각 샘플은 80개의 원소로 된 1차 신호열이었지만 임베딩 계층을 통과하면서 각 단어가 128의 길이를 가지는 벡터로 바뀌면서 입력 데이터의 모양이 80 * 128로 변경된다.

다음으로 노드 128개로 구성된 LSTM 계층을 포함한다.

    h = layers.LSTM(128,dropout = 0.2, recurrent_dropout = 0.2)(h)   # 일반 드롭아웃, 순환 드롭아웃 모두 20%로 설정
      
최종적으로 출력을 시그모이드 활성화 함수로 구성된 출력 노드 하나로 구성한다.  

    y = layers.Dense(1, activation = 'sigmoid')(h)
    super().__init__(x,y)
    
손실 함수와 최적화 함수를 아규먼트로 지정하여 모델을 컴파일한다.

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam'. metrics = ['accuracy']
    
  - 긍정인지 부정인지에 대한 이진 판별값을 출력으로 다루므로 손실 함수를 binary_crossentropy로 , 최적화 함수를 adam으로 설정했다.

### 5.2.4 학습 및 성능 평가 ###

:four: 학습 및 성능 평가를 담당할 머신 클래스를 만든다.

    class Machine:
      def __init__(self, max_features = 20000, maxlen = 80):   # max_features는 다루는 단어의 최대 수, 글에는 다양한 단어가 무작위로 사용되지만 빈도 순위가 20,000등 안에 드는 단어까지 취급한다는 의미
        self.data = Data(max_features, maxlen)    # maxlen은 한 문장의 최대 단어 수 의미
        self.model = RNN_LSTM(max_features, maxlen)
    
    
- 학습과 평가를 수행할 run() 멤버 함수를 만든다.

      def run(self, epochs = 3, batch_size = 32):
        data = self.data
        model = self.model
        print('Training stage')
        print('========================')
        
        model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test))
        
        score, acc = model.evaluate(data.x_test,data.y_test, batch_size = batch_size)  # 학습 잘 되었는지 평가 데이터 이용해 확인, 검증 데이터와 평가 데이터 같은 것으로 사용
        print('Test performance: accuracy={0}, loss{1}'.format(acc,score)

## 5.3 시계열 데이터를 예측하는 LSTM 구현 ## 

LSTM을 이용해 시계열 데이터에 대한 예측을 해본다.

세계 항공 여행 승객 수의 증가에 대한 데이터를 활용하여 이전 12개월치 승객 수를 이용해 다음 달의 승객 수를 예측한다.

다음과 같은 순서로 시계열 예측 인공신경망을 구현한다.

1) 패키지 임포트
2) 코드 실행 및 결과 보기
3) 학습하고 성능 평가하기
4) LSTM 회귀 모델링
5) 데이터 불러오기

### 5.3.1 패키지 임포트 ### 

:one: 데이터를 불러오고 기본적인 처리를 하는 데 필요한 두 패키지를 임포트 한다.

    import pandas as pd
    import numpy as np
    import seaborn as sns
  
  - pandas는 엑셀과 같이 시트로 관리되는 데이터를 처리하는 패키지이다.

  - seaborn은 통계 그래프를 그리는 패키지이다.

- 모델링을 진행한 후 성능 평가는 매우 중요하다. 이를 위해서는 일부 데이터를 성능 평가에 사용하도록 학습에 사용하지 않고 남겨 두어야 한다.

      from sklearn import model_selection
     
   - model_selection()은 데이터를 학습과 검증용으로 나누는 함수이다.

- 이제 케라스와 관련된 패키지들을 임포트한다.

      from keras import models, layers
     
### 5.3.2 코드 실행 및 결과 보기 ### 

:two: 세부 코드를 보기 전에 머신을 만들고 실행하는 부분을 먼저 보자.

- LSTM을 이용하는 회귀 인공신경망을 실행하는 코드의 첫 줄은 다음과 같다.

      def main():
        machine = Machine()  # Machine()클래스를 이용해 machine을 인스턴스로 만듦
        machine.run(epochs = 400)
        
     - 이 줄을 실행하면 데이터 시각화 그래프 두 개와 모델링 요약 정보가 출력된다.

     - machine 인스턴스를 이용해 학습 및 성능 평가를 진행한다. 
        
### 5.3.3 학습하고 평가하기 ### 

3️⃣ 머신 클래스는 시계열 LSTM을 학습하고 평가하는 플랫폼이다. 초기화 함수와 실행 함수를 만들면 된다.

- 우선 클래스를 선언한 후 머신을 초기화 한다.

      class Machine():
        def __init__(self):
          self.data = Dataset()  # 데이터 생성에 사용한 Dataset()클래스의 인스턴스 만듦
          shape = self.data.X.shape[1:]   # LSTM의 입력 계층 크기를 shape변수에 저장
          self.model = rnn_model(shape)
          
- 이제 머신을 실행하는 멤버 함수를 만들 차례이다. 

      def run(self, epochs = 400):
        d = self.data
        X_train,X_test = d.X_train, d.X_test
        y_train, y_test = d.y_train, d.y_test
        X,y = d.X, d.y
        m = self.model

- 모델의 학습을 진행할 차례이다.

      h = m.fit(X_train,y_train, epochs = epochs, validation_data = [X_test, y_test], verbose =0)  # 학습
      
- 학습이 얼마나 잘 진행되었는지를 알아본다.

      yp = m.predict(X_test).reshape(-1)
      print('Loss:', m.evaluate(X_test,y_test)
      
- 그래프를 이용해 원래 레이블 정보인 y_test와 예측한 레이블 정보인 yp를 같이 그려서 서로 비교한다.

      yp = m.predict(X_test).reshape(-1)
      print('Loss:', m.evaluate(X_test,y_test)
      plt.plot(yp, label = 'original')
      plt.plot(y_test, label = 'prediction')
      plt.legend(loc=0)
      plt.title('validation results')
      plt.show()
      
:heavy_exclamation_mark: 그러나 이 방법은 데이터가 시간적인 관계가 없는데 시간순으로 되어있어서 확인에 효과적이지 않을 수 있다. 따라서 막대 그래프를 이용해 다음과 같이 그리도록 했다.

    yp = m.predict(X_test).reshape(-1)
    print('Loss:', m.evaluate(X_test,y_test))
    print(yp.shape,y_test.shape)
    
    df = pd.DataFrame()
    df['Sample'] = list(range(len(y_test))) * 2
    df['Normalized #Passengers'] = np.concatenate([y_test,yp], axis =0)
    df['Type'] = ['original'] *len(y_test) + ['prediction'] * len(yp)
    
    plt.figure(figsize = (7,5))
    sns.barplot(x="Sample", y="Normalized #Passengers", hue = "Type", data = df)
    plt.y_label('Normalized #Passengers')
    plt.show()
    
  - 막대 그래프 그리는 데 seaborn 패키지 이용하였다. 

  - pandas 패키지 이용해 시계열 데이터를 데이터 프레임으로 변환해 seaborn에 제공한다.

  - 목표 결과와 예측 결과를 비교하기 위해 다음과 같은 방법으로 시계열 데이터를 3개의 열로 구성된 데이터 프레임으로 변환했다.
  
    1. 목표 결과와 예측 결과의 순서를 표시하는 Sample이라는 열을 만든다. 이 열은 0부터 len(y_test)-1까지 정수가 두 번 반복되어 들어간다. 첫 번째는 목표 결과의 순서이고 두 번째는 예측 결과의 순서이다.
    2. 목표 결과와 예측 결과를 순서대로 결합해서 'Normalized #Passnegers'라는 열을 만든다.
    3. 두 열들의 앞쪽은 모두 목표 결과와 관련된 정보이고 뒤쪽은 모두 예측 결과와 관련된 정보임을 표시하는 Type열을 만든다.

- 이제 학습 데이터와 평가 데이터의 결과를 합쳐서 보여준다.

      yp = m.predict(X)
      plt.plot(yp, label = 'original')
      plt.plot(y, label = 'prediction')
      plt.legend(loc=0)
      plt.title('All results')
      plt.show()

  - 모델의 멤버 함수인 predict를 이용해 X_test가 아닌 전체 데이터인 X에 대해 예측하도록 했다.

  - 그리고 그 결과를 원래 레이블 정보를 시간 축으로 그렸다. 전체 데이터가 시간순으로 배열되어 있기 때문이다.










