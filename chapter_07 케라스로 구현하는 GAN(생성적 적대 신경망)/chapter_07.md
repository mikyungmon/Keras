# Chapter 07 #

🔎 생성적 적대 신경망(GAN)이란 ? 

  - 경쟁하여 최적화를 수행하는 생성형 신경망이다.
  
  - GAN 내부의 두 신경망이 상호 경쟁하면서 학습한다.
  
  - 두 신경망 중 하나는 **생성망**이고 다른 하나는 **판별망**이다.

▶ 해당 장에서 다루는 내용은 다음과 같다. 
  
  1) GAN의 원리
  2) 완전 연결 계층을 이용한 GAN 구현 : 확률분포
  3) 합성곱 계층을 이용한 GAN 구현 : MNIST

## 7.1 GAN의 원리 ##

GAN은 **경쟁적 학습 방법을 이용하는 생성형 인공지능**이다. 이 절에서는 GAN의 목적과 개념, 구조, GAN을 구성하는 생성망과 판별망에 대해서 배운다.

### 7.1.1 GAN의 목적과 개념 ###

GAN은 생성형 인공지능 방식으로 **실제 데이터**와 비슷한 확률분포를 가지는 **허구 데이터**를 생성한다.

허구 데이터는 GAN에서 만들어진 데이터이기 때문에 **생성 데이터**라고도 한다.

예를 들어 GAN은 실제 데이터로 얼굴 사진을 제공하면 비슷한 확률분포를 가지는 새로운 허구 얼굴 사진을 생성한다.

반면 DNN, CNN과 같은 판별형 신경망은 데이터의 레이블을 학습하여 데이터를 판별한다.

GAN은 레이블이 없는 정보를 다루는 **비지도학습**이다.

입력 데이터는 **무작위 잡음**이고 출력 데이터는 입력 데이터보다 높은 차원으로 구성된다.

입력한 무작위 잡음은 출력 데이터와 상관이 없기 때문에 GAN은 비지도형의 생성형 신경망이다.

출력 데이터는 특정 분포도를 갖은 데이터로 가정한다. 예를 들어 필기체 숫자나 사람 얼굴 사진 등이 될 수 있다.

**학습을 마친 GAN에 새로운 무작위 잡음을 입력하면 학습한 실제 데이터와 유사한 형태의 허구 데이터를 출력한다.**

예를 들어 필기체 숫자나 사람 얼굴을 학습시키면 학습시킨 것과 비슷한 필기체 숫자나 사람 얼굴이 나온다.

잡음은 이론적으로 무한한 변위가 가능하므로 출력으로 나온 결과는 학습에 사용한 그 어떤 데이터와도 완전히 같지는 않다.

### 7.1.2 GAN의 구조 ###

![image](https://user-images.githubusercontent.com/66320010/124115061-4c09be80-daa8-11eb-9b8a-6f9ac040787f.png)

- GAN 인공지능은 경쟁적인 방법으로 학습을 수행한다. 위 그림과 같이 두 망이 복합 구성되어 있다(첫 번째는 생성망이고 다음은 판별망이다).

- 학습의 목적은 학습한 실제 데이터와 같은 확률분포를 가지는 새로운 허구 데이터를 만들도록 생성망을 학습시키는 것이다.

- 생성망이 실제 데이터와 확률 분포상 유사한 데이터를 만들려면 GAN의 학습이 잘 진행되어야 한다.

- **판별망**은 실제 데이터인지 만들어진 허구 데이터인지를 더 잘 구분하도록 학습하고, **생성망**은 학습을 통해 판별망을 더 잘 속이도록 학습한다.

### 7.1.3 GAN의 동작 원리 ###

GAN의 생성망과 판별망의 동작 원리를 알아보자.

GAN은 **생성망**과 **판별망**이라는 두 신경망이 합쳐져 구성된다. 

 ▶ 생성망은 주어진 데이터와 유사한 허구 데이터를 생성한다.
 
 ▶ 판별망은 생성망에서 전달된 데이터가 생성망에서 만든 허구 데이터인지 아니면 원래 주어진 실제 데이터인지를 구분한다.

**생성망**은 저차원 무작위 잡음을 입력받아 고차원 허구 이미지를 생성한다. 허구 이미지는 생성망이 만든 이미지이므로 생성 이미지라고도 부른다. 

실제 이미지를 학습하여 실제 이미지와 확률분포가 최대한 비슷하도록 허구 이미지를 만든다. 

💡 **이 학습 과정에서 판별망의 판단 결과를 활용하게 된다.** 생성망이 만든 허구 이미지를 판별망이 실제 이미지로 착각하도록 만드는 방향으로 생성망 학습이 이루어지기 때문이다.

**판별망**은 입력된 이미지가 실제인지 허구인지 판별한다. 문제는 실제 이미지는 변하지 않지만 허구 이미지는 생성망의 학습이 진행됨에 따라 점점 실제 이미지와 유사해진다는 점이다.

그래서 판별망은 바로 앞서 생성망에 의해 만들어진 허구 이미지들과 주어진 실제 이미지들을 판별할 수 있도록 점진적으로 학습을 진행한다. 

발전된 생성망의 결과를 허구로 판별할 수 있도록 상호 공진화하는 방식이다.

이미지 판별과 생성 시 합성곱 계층을 사용하면 GAN에서도 완전 연결 계층보다 더 효과적인 처리를 할 수 있다. 예제를 통해 두 경우 모두에 대해 알아볼 것이다.

### 7.1.3 GAN의 동작 사례 ###

GAN을 제안한 이안 굿펠로우의 논문에 제시된 예제를 활용하여 GAN이 어떻게 구현되는지 알아본다. 

다음 그림은 GAN이 입력된 확률변수를 변환시켜 원하는 확률변수로 바꾸어 출력하는 예를 보여준다.

![image](https://user-images.githubusercontent.com/66320010/124116521-ff26e780-daa9-11eb-833f-7e63b7c48b0d.png)

우선 판별망의 동작에 대해 알아보자.

![image](https://user-images.githubusercontent.com/66320010/124116904-752b4e80-daaa-11eb-9b11-8d015281fd2b.png)

- 판별망은 무작위 잡음 벡터 z를 입력받아 생성하는 생성망의 결과를 판별하는 신경망이다. 

- 일반적으로 판별망은 개별 이미지가 무엇인지 판별하는 데 사용된다.

- 그렇지만 GAN의 판별망은 **개별 이미지가 아닌 이미지의 확률분포를 판별한다.**

- 판별망은 실제 데이터와 생성망이 만든 허구 데이터의 확률분포 차이를 판별하도록 학습된다.

- 목표 데이터를 1로 판별하는 과정은 위 그림의 a 흐름도와 같다. 먼저 실제 데이터의 일부를 판별망에 입력한다. 실제 데이터 전체에서 일부를 가져오는 방식은 배치처리와 같이 이루어진다.

- 미분 가능한 판별 함수 D가 실제 데이터의 샘플을 1로 판별할 수 있도록 학습시킨다. 판별 함수 D는 정확도가 높은 판별을 위해 신경망으로 구성한다.

- 생성 데이터를 0으로 판별하는 단계를 위 그림의 b 흐름도와 같다. GAN은 실제 데이터의 확률분포와 다른 임의의 확률 분포를 가진 무작위 잡음을 만든다.

- 만들어진 무작위 잡음을 미분 가능한 생성 함수 G에 통과시킨다.

- 여기서도 복잡한 확률분포를 변환하기 위해 생성 함수 G를 신경망으로 구성한다. 

- 다음으로 생성 함수 G가 생성한 데이터를 추출하고 추출된 데이터를 목표 데이터와 동일하게 판별 함수 D에 통과시킨다. 

- 판별 함수는 실제 데이터 판별에 사용한 신경망을 한 번 더 사용한다. 판별값은 거짓 즉 0이 되어야한다.

📍 판별값은 실제 데이터를 입력하면 1, 생성 데이터를 입력하면 0이어야 한다.
 
이 판별은 학습 과정이 진행되면서 점점 명확해진다. 초기 생성망은 임의의 형태로 구성된다고 가정한다. 

**판별망을 학습시킬 때 생성망까지 학습되면 안되므로 생성망의 가중치는 학습이 되지 않도록 고정해야한다.**

이제 판별에 들어가는 입력은 실제 데이터에서 추출한 배치 데이터와 생성망에서 만든 허구 데이터로 구성된다. 

그리고 목표 출력값은 각 데이터 군별로 1과 0 에 해당한다.

이런 조건으로 판별망 학습을 진행한다. **단, 판별망 학습 이후에 생성망이 또다시 진화되기 때문에 순환적으로 계속 판별망을 학습시키는 점은 기존의 일반 판별망과 다르다.** 이는 뒤에서 좀 더 자세히 다룬다.

판별망 학습이 끝나면 생성망을 학습할 차례이다. 우선 생성망의 결과가 판별망으로 들어가도록 가상 신경망 모델을 구성한다. 이를 '생성망 최적화를 위한 학습용 생성함수 GD'또는 '학습용 생성망'이라고 한다.

주의할 점은 이 학습용 생성망은 새로운 신경망이 아니라 기존의 생성망과 판별망이 합쳐진 가상 생성망이라는 점이다.

또한 이 학습용 생성망 모델에서 판별망 부분은 학습이 되지 않도록 가중치를 고정한다. 이렇게 하면 학습용 생성망에서 판별망은 무작위 잡음 벡터 z로부터 생성된 허구 이미지가 얼마나 실제 이미지와 유사한지 판별한 결과를 내게 된다.

그리고 이 판별한 결과가 모든 실제가 되도록 생성망을 학습한다. 허구인지 실제인지 구분한 결과는 일반적인 분류망에서와 마찬가지로 교차 엔트로피로 표현된다.

학습용 생성망의 학습을 통해 생성망은 한 단계 진화한다.

진화된 생성망은 또다시 무작위 입력 벡터 z에 대해 이미지 변환을 수행한다.

이렇게 판별망과 생성망의 학습이 한 번씩 진행되면 GAN전체 학습이 한 번 수행된 것이다. 

이 전체 최적화와 각 망들의 부분 최적화를 병행하면서 GAN은 점점 목표한 최적 단계로 발전해나간다.

결국 임의의 정규분포 입력의 분포를 조절하여 실제 이미지를 생성하도록 발전한다.

최적화가 완전히 끝나면 이론적으로는 생성망의 결과와 실제 이미지를 판별망이 구분하지 못하게 된다.

이런 경지를 달성하려면 각 생성망과 판별망이 최적으로 구성되고 둘의 밸런스도 맞아야한다.

## 7.2 확률분포 생성을 위한 완전 연결 계층 GAN 구현 ## 

- GAN을 처음 제안한 논문에 게재된 예제를 구현한다. 이 예제는 GAN으로 정규분포를 생성한다. 

- 생성에 사용하는 무작위 벡터 z는 균등분포 확률신호인데 출력은 정규분포 확률 신호이다.

다음과 같은 순서로 진행된다.

1) 패키지 임포트
2) 코드 수행과 결과 보기
3) 머신 구현하기
4) 데이터 관리 클래스
5) GAN 모델링

### 7.2.1 패키지 임포트 ###

1️⃣ 확률분포 생성 GAN을 구현하는 데 필요한 패키지 임포트 단계와 코드 수행 단계를 먼저 살펴본다.

   ✔ 패키지 임포트
   
   ✔ 머신 수행하기

- GAN의 구현에 필요한 행렬 계산을 다루는 넘파이와 그래픽을 담당하는 맷플롯립 라이브러리를 임포트 한다.

      import numpy as np
      import matplotlib.pyplot as plt
    
- 다음은 인공지능 구현에 필요한 케라스 서브패키지들을 불러온다.

      from keras import models
      from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda
    
- 최적화에 사용되는 클래스와 백엔드 패키지도 임포트한다.

      from keras.optimizers import Adam
      from keras import backend as K
      
   - Adam 클래스를 임포트한 이유는 최적화에 사용하는 파라미터를 코드에서 변경하기 위해서이다. 케라스는 컴파일 단계에서 optimizer를 문자열로 지정하면 파라미터가 기본값으로 최적화된다.
    
     ▶ 따라서 최적화 파라미터값을 변경하길 원한다면 해당 최적화 클래스를 임포트 하면 된다.

### 7.2.2 코드 수행과 결과 보기 ###

2️⃣ GAN을 동작시키는 머신의 인스턴스를 만든다.

    machine = Machine(n_batch = 1, ni_D = 100)   # 매 에포크마다 길이가 100인 벡터 하나를 출력하도록 설정

- 이제 만들어진 머신을 수행한다.

      machine.run(n_repeat = 200, n_show = 200, n_test = 100)
      
    - n_repeat : 전체를 반복하는 횟수
    
    - n_show : 결과를 표시할 총 에포크 수
    
    - n_test : 내부 성능 평가 시 사용할 샘플 수 

- main() 함수가 실행되도록 한다.

      if __name__ == '__main__' :
         main()

    - 이렇게 하면 코드를 임포트할 때는 main()함수를 호출하지 않지만 명령행 코드를 수행하면 main()이 호출된다.

     📍 조건문으로 __ name__을 검사하면 **임포트 시에는 동작하지 않고 명령행으로 수행할 때 동작하는 코드 블럭을 만들 수 있다.** 이 코드블럭은 $python.code.py와 같이 명령할 때만 동작한다. 이는 __ name__이라는 시스템 변수가 '__ main__'으로 설정되어 있는지를 보고 정한다. 시스템 변수 __ name__은 파이썬 코드가 임포트를 통해서 실행되었는지 아닌면 명령행으로 실행되었는지에 따라 값이 다르게 설정된다. 명령행으로 실행하면 '__ main__'으로 설정된다.

위의 machine.run()을 수행하면 진행 상황이 200에포크마다 히스토그램 그래프로 출력된다.

다음 그림에서 스테이지가 10일 때와 199일 때 그래프를 확인할 수 있다.

이 확률 분포 비교 그래프를 보면 에포크가 늘어남에 따라 생성된 허구 데이터가 입력한 무작위 잡음 벡터와 갈수록 달라지는 걸 알 수 있다.

그러다가 stage 199에 이르면 허구 데이터가 실제 데이터의 분포와 꽤 유사해진다.

### 7.2.3 데이터 생성 클래스 ###

3️⃣ GAN에 적용할 데이터 관리 클래스를 다음과 같이 만든다.

    class Data : 
      def __init__(self, mu, sigma, ni_D):
        self.real_sample = lambda n_batch : np.random.normal(mu,sigma,(n_batch,ni_D)   # 흉내 내고자 하는 실제 데이터
        self.in_sample = lambda n_batch : np.random.rand(n_batch,ni_D)   # 무작위 잡음 데이터
  
   - GAN에는 두 가지 데이터가 필요하다. 

   - 첫 번째는 GAN으로 흉내 내고자 하는 실제 데이터, 두 번째는 실제 데이터와 통계적 특성이 다른 무작위 잡음 데이터이다.

   - 이 둘을 만들려면 확률변수를 생성하는 함수가 필요하다. 

      ➡ 정규분포 확률변수는 numpy 아래에 random.normal()함수로 생성한다. 이 함수를 활용해서 확률변수를 생성하는 함수를 lambda로 만들어 반환한다.
      
      ➡ 아규먼트를 위한 확률은 random.rand()을 사용해 연속균등분포로 지정한다.

### 7.2.4 머신 구현하기 ###

4️⃣ **머신**은 데이터와 모델로 GAN을 학습하고 성능을 평가하는 인공신경망 전체를 총괄하는 객체이다. 여기서는 Machine 클래스로 머신 객체를 구현한다. 

- 머신 내부를 구현한다. 멤버 함수별로 구분하여 다음과 같은 순서로 살펴본다.

  1) 클래스 초기화 함수 : __ init__()
  2) 실행 멤버 함수 : run()
  3) 에포크 단위 실행 멤버 함수 : run_epochs()
  4) 학습 진행 멤버 함수 : train()
  5) 매순간 학습 진행 멤버 함수 : train_epoch()
  6) 판별망 학습 멤버 함수 : train_D()
  7) 학습용 생성망 학습 멤버 함수 : train_GD()
  8) 성능 평가 및 그래프 그리기 멤버 함수 : test_and_show()
  9) 상황 출력 정적 함수 : print_stat()

- 머신 클래스 초기화 함수를 만든다.

      class Machine : 
        def __init__(self, n_batch = 10, ni_D = 100) :
          data_mean = 4
          data_stddev = 1.25
          self.data = Data(data_mean, data_stddev, ni_D)

  - ni_D : 판별망이 한꺼번에 받아들일 확률변수 수

- GAN 객체로 GAN모델의 인스턴스를 만든다. GAN을 구성하는 2가지 신경망인 판별망과 생성망 은닉 계층의 노드 수를 모두 50으로 설정한다.

      self.gan = GAN(ni_D = ni_D, nh_D = 50, nh_G = 50)

- 다음으로 배치 단위를 설정한다. 또한 생성망과 판별망의 배치별 최적화 횟수도 정한다.

      self.n_batch = n_batch
      self.n_iter_D = 1   # 배치별 최적화 횟수
      self.n_iter_G = 1

  - 판별망(D)와 생성망(G)의 각 배치마다 에포크를 다르게 가져갈 수도 있다. 
  
  - 기본은 한 번 배치가 수행될 때 판별망이 한 번 학습된 후 생성망이 한 번 학습되는 것이다.
  
  - GAN을 처음 제안한 논문에는 배치별로 판별망을 생성망보다 더 많이 학습하면 도움이 된다고 언급되어 있다.
  
  - 몇 번을 더 학습하는지는 하이퍼파라미터로 설정할 수 있으며 논문에서는 1회씩을 사용했다.

- 머신 클래스의 실행을 담당하는 멤버 함수인 run()을 만든다. 이 함수는 n_show번 학습이 진행될 때마다 그 결과를 그래프로 표시한다.

      def run(self, n_repeat = 30000 // 200, n_show = 200, n_test = 100):
        for ii in range(n_repeat):
          print('Stage', ii, '(Epoch: {})'.format(ii * n_show))
          self.run_epochs(n_show,n_test)
          plt.show()

  - run_epochs()함수는 호출될 때마다 학습을 n_show번 수행한다.
  
- 에포크 단위로 실행을 담당하는 멤버 함수인 run_epochs()를 구현한다. 이 함수는 첫 번째 입력 아규먼트인 epochs만큼 학습을 진행한다.

      def run_epochs(self,epochs,n_test):
        self.train(epochs)
        self.test_and_show(n_test)
     
    - epochs만큼 학습을 진행하는 함수를 호출한 후 학습된 신경망에 내부 성능 평가 데이터를 넣어 그 성능을 결과 그래프로 보여주는 함수를 호출한다.

- GAN의 학습을 진행하는 멤버 함수를 만든다.

      def train(self,epochs):
        for epoch in range(epochs):
          self.train_each()   # 매 에포크마다 멤버 함수인 train_each()를 호출해 학습

    - GAN의 학습은 판별망 D와 학습용 생성망 GD의 반복 학습으로 진행된다.

- D와 G의 학습이 매번 각 1회 이상 수행될 수 있게 했다. 이번 예제에서는 단순화를 위해 각 1회씩 학습한다.

      def train_each(self):
        for it in rnage(self.n_iter_D):   # 판별망은 n_iter_D만큼 학습
          self.train_D()
        for it in range(self.n_iter_G):   # 학습용 생성망도 n_iter_GD만큼 반복 학습
          self.train_GD()

- 판별망을 학습시키는 멤버 함수를 살펴보자.

      def train_D(self):
        gan = self.gan
        n_batch = self.n_batch
        data = self.data
        
- data.real_sample()함수로 실제 데이터에서 n_batch 만큼 샘플을 가져온다. 실제 데이터는 정규분포를 따르는 샘플이다.        
        
      Real = data.real_sample(n_batch)  # (n_batch,ni_D)     
        
- 그리고 임의의 분포를 가지는 입력 샘플을 data.in_sample()함수로 데이터 샘플 수와 같은 수만큼 만든다.

      Z = data.in_sample(n_batch)    # (n_batch,ni_D)       
        
- 다음은 입력 샘플을 생성기에 통과시켜 생성망의 출력으로 바꾼다.

      Gen = gan.G.predict(Z)  # (n_batch,ni_D)       
        
- 판별망은 **학습용 생성망**을 학습할 때는 **학습이 되지 않도록 막아두기 때문에 gan.D.trainable을 True로 바꾸고 학습을 진행해야 한다.**        
        
      gan.D.trainable = True  
        
- 이제 판별망을 학습한다.

      gan.D_train_on_batch(Real,Gen) 
      
- 다음은 머신 클래스에 들어갈 학습용 생성망을 학습하는 멤버 함수이다.      
        
      def train_GD(self):
        gan = self.gan
        n_batch = self.n_batch
        data = self.data
        Z = data.in_sample(n_batch)
        
        gan.D.trainable = False
        gan.GD_train_on_batch(Z)
        
    - 입력이 생성망에 들어가면 모든 판별망이 실제 샘플로 착각하도록 GD_train_on_batch()를 이용해 학습한다.

- 학습용 생성망을 학습할 때는 실제 데이터를 다룰 필요가 없기 때문에 판별망 학습보다는 코드가 간단하다.

- 현재까지 학습된 GAN 신경망의 성능을 평가하고 확률 예측 결과를 그래프로 그리는 멤버 함수를 만들 것이다. 총 n_test만큼 데이터를 만들어서 test()멤버 함수에 입력한다.

      def test_and_show(self,n_test):
        data = self.data
        Gen, Z = self.test(n_test)
        Real = data.real_sample(n_test)  # 실제 이미지를 n_test만큼 가져와서 Real에 저장
        self.show_hist(Real,Gen,Z)
        Machine.print_stat(Real,Gen)

- 학습 진행 경과에 대한 그래프를 그리는 멤버 함수를 구현한다.

      def show_hist(self,Real,Gen,Z):
        plt.hist(Real.reshape(-1), histtype = 'step', label='Real')
        plt.hist(Gen.reshape(-1), histtype = 'step', label = 'Generated')
        plt.hist(Z.reshape(-1), histtype = 'step', label = 'Input')
        plt.legend(loc=0)
    
   - 내부 평가에 사용된 무작위 잡음 데이터(Z)와 생성 데이터(Gen)의 통계적 특성을 같은 수의 실제 데이터(Real)의 통계적 특성과 비교한다.
   
   - Real,Gen,Z의 통계적 특성을 plt.hist()를 사용해 표시한다.

- 생성망이 얼마나 실제 데이터의 확률분포를 따르는 데이터를 만드는지 확인하는 정적 멤버 함수를 만들어 본다.

      @stticmethod
        def print_stat(Real,Gen):
          def stat(d):
            return (np.mean(d),np.std(d))
          print('Mean and Std of Real :', stat(Real))
          print('Mean and Std of Gen :', stat(Gen))
          
### 7.2.5 GAN 모델링 ###          
          
5️⃣ 다음과 같은 순서로 GAN을 모델링한다.

  1) 클래스 초기화 함수 : __ init__()
  2) 판별망 구현 멤버 함수 : gen_D()
  3) 생성망 구현 멤버 함수 : gen_G()
  4) 학습용 생성망 구현 멤버 함수 : make_GD()
  5) 판별망 학습 멤버 함수 : D_train_on_batch()
  6) 학습용 생성망 학습 멤버 함수 : GD_train_on_batch()

- GAN의 판별망과 생성망을 모델링하는 클래스의 초기화 함수를 만들어보자.

      class GAN :
        def __init__(self,ni_D,nh_D, nh_G):
          self.ni_D = ni_D    # 판별망 입력 길이
          self.nh_D = nh_D    # 판별망의 두 은닉 계층의 노드 수 
          self.nh_G = nh_G    # 생성망의 두 은닉 계층의 노드 수 
          
          self.D = self.gen_D()
          self.G = self.gen_G()
          self.GD = self.make_GD()
          
          
- 판별망을 구현하는 멤버 함수를 만든다. GAN모델에서 구현할 판별망은 다음과 같다.

![image](https://user-images.githubusercontent.com/66320010/125069584-e1810000-e0f1-11eb-848d-8cbfa5b9fbd0.png)
         
판별망은 입력과 출력 계층을 포함하여 여섯 계층으로 구성된다. 

입력 계층 다음은 람다 계층이고 그 다음은 두 은닉 계층이고 마지막은 출력 계층이다.

두 은닉 계층과 출력 계층은 모두 완전 연결 계층으로 구성된다.

- 위 그림에 나온 판별망을 케라스의 연쇄 방식으로 구현한다.

      def gen_D(self):
        ni_D = self.ni_D
        nh_D = self.nh_D
        D = models.Sequential()
        D.add(Lambda(add_decorate,output_shape = add_decorate_shape, input_shape(ni_D,)))
        D.add(Dense(nh_D, activation='relu'))
        D.add(Dense(nh_D, activation='relu'))
        D.add(Dense(1, activation='sigmoid'))
        
        model_compile(D)
        return D
        
    - Lambda클래스는 계층의 동작을 처리하는 함수인 add_decorate()와 계층을 통과한 출력 텐서의 모양인 output_shape을 입력받는다.
          
- 람다 계층의 처리 함수인 add_decorate()를 만든다.

      def add_decorate(x):
        m = K.mean(x,axis = -1,keepdims = True)
        d = K.square(x - m)
        return K.concatenate([x,d], axis = -1)
        
   - 이 함수는 입력 벡터에 새로운 벡터를 추가한다. 새로운 벡터는 입력 벡터의 각 요소에서 벡터 평균을 뺀 값을 자승한 값을 가진다(즉, (x-m)^2한 값).
   
   - 벡터의 추가는 K.concatenate()엔진 함수로 구현한다. 입력 벡터와 추가 벡터를 연속으로 붙여서 새로운 벡터를 만들어 준다. 
   
   - 붙이는 위치를 지정하는 아규먼트 axis를 -1로 설정했다. 이는 벡터의 **가장 마지막 차원**을 서로 붙이라는 의미이다.

- 출력 데이터의 모양을 지정하는 add_decorate_shape(input_shape)을 만든다. 

      def add_decorate_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] *= 2
        return tuple(shape)
        
  - 컴파일 단계가 끝나고 학습 단계에서 처리 함수가 계산되면 출력 벡터의 크기가 반환되지만 케라스에서는 이렇게 add_decorate_shape()을 이용해 그 크기를 컴파일 이전에 명시해주어야 한다.
  
  - 왜냐하면 케라스는 컴파일할 때 신경망의 구조를 설정하는데 그러려면 컴파일 시점에서 각 계층의 입력의 출력 크기를 알 수 있어야 하기 때문이다.        
        
- 정의한 신경망들을 컴파일 하는 model_compile()를 구현한다.

      lr = 2e-4
      adam = Adam(lr = lr, beta_1 = 0.9, beta_2 = 0.999)   # beta 최적화 값을 (0.9,0.999)로 설정
      def model_compile(model):
        return model.compile(loss= 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
        
- 생성망을 구현하는 멤버 함수를 만든다. 생성망 구조는 다음 그림과 같다.

![image](https://user-images.githubusercontent.com/66320010/125072490-a41e7180-e0f5-11eb-9038-44e24f8a3e28.png)

생성망 모델링도 판별망처럼 연쇄 방식으로 구성한다.
        
    def gen_G(self):
      ni_D = self.ni_D
      nh_G = self.nh_D
      
      G = models.Sequential() 
      G.add(Reshape((ni_D,1), input_shape=(ni_D,))) 
      G.add(Conv1D(nh_G,1,activation='relu'))   # nh_G은 변환에 사용할 필터 수 , 1은 커널 크기(입력 벡터 간의 상관도를 높여주는 역할)
      G.add(Conv1D(nh_G,1,activation='sigmoid'))
      G.add(Conv1D(1,1))
      G.add(Flatten())  # 생성망 출력을 1차원으로 만듦
      
      model_compile(G)
        return G
      
  - 생성망에서 생성된 확률변수가 주어진 확률변수와 같은 확률분포를 가지는지 판별망으로 판단하려면 확률변수 여럿이 필요하다. 이를 위해서 생성망을 매번 ni_D만큼의 확률변수를 생성한다.
  
  - 각 확률변수를 서로 독립적으로 생성하기 위해서 각 층은 1차원 합성곱을 사용하여 구성한다. 생성망에 들어가는 입력 데이터 모양은 (Batch, input_dim)이다.
  
  - 1차원 합성곱 계층에 벡터 입력을 넣으려면 (Batch,steps,input_dim)으로 데이터 차원을 확대해야한다. Reshape()을 사용하여 input_dim을 steps축 기준으로 전환시키면 된다.

- 학습용 생성망을 구현하는 멤버 함수를 만든다. 학습용 생성망은 생성망을 학습시키는 가상 신경망이다. 이는 생성망의 상단에 판별망을 달아주어 구현한다. 이 때 판별망의 가중치는 학습 중에 변하지 않도록 해야한다.

      def make_GD(self):
        G, D = self.G, self.D
        GD = models.Sequential()
        GD.add(G)
        GD.add(D)
        
        D.trainable = False   # D가 학습되지 않게 trainable을 끔
        model_compile(GD)
        D.trainable = True
        return GD

- 판별망의 학습을 진행하는 함수를 만든다. 새로운 함수가 필요한 이유는 GAN이 비지도형 신경망이라서 일반적인 지도형 학습을 사용할 수 없기 때문이다.

      def D_train_on_batch(self,Real,Gen):
        D = self.D
        X = np.concatenate([Real,Gen],axis = 0)
        y = np.array([1] * Real.shape[0] + [0] *Gen.shape[0])   # 허구를 0, 실제를 1로 판단, 한꺼번에 학습하고자 실제 신호 벡터와 생성망이 만든 허구 신호를 연결
        D.train_on_batch(X,y)
  
  - train_on_batch()는 앞서 사용했던 fit()과는 처리하는 데이터양이 다르다. fit()은 전체 데이터를 받아 배치처리로 반복 학습하는데, train_on_batch()는 배치 크기의 데이터만 받고 1회만 학습한다.
      
- 학습용 생성망을 학습시키는 멤버 함수를 만든다.      
      
      def GD_train_on_batch(self,Z):
        GD = self.GD
        y = np.array([1]*Z.shape[0])
        GD.train_on_batch(Z,y)
        
   - 학습용 생성망도 판별망의 학습과 마찬가지로 어떤 값을 목표로 할지 지정했다.
   
   - **생성망에서 출력되는 허구값을 판별망에서 실제값으로 판별하도록 학습해야하기 때문에 목표 출력값을 모두 1로 설정했다.**
   
   - 그리고 생성망에 들어가는 입력값을 입력으로 하고 모두를 1로 설정한 벡터를 출력으로 하여 train_on_batch()를 이용해 학습하도록 했다.
      
      
## 7.3 필기체를 생성하는 합성곱 계층 GAN 구현 ##      
      
GAN을 이용해 필기체 숫자를 생성하는 인공신경망을 만들어보자.

GAN은 입력한 필기체를 보고 배워 새로운 유사 필기체를 만든다. GAN에 들어있는 두 인공신경망은 **합성곱 계층**을 이용해 만든다.

GAN을 이용한 필기체 숫자 생성은 학습에 오랜 시간이 걸리고 파라미터를 이용한 조절이 중요하기 때문에 명령행에서 아규먼트를 입력받는 방식으로 코드를 구현한다.
      
다음과 같은 순서로 진행한다.

1) 공통 패키지 불러오기
2) 합성곱 계층 GAN 수행하기
3) 합성곱 계층 GAN 모델링
4) 합성곱 계층 GAN 학습하기

### 7.3.1 공통 패키지 불러오기 ###  
      
1️⃣ 공통 패키지 MNIST 데이터셋을 불러오는 케라스 서브패키지를 임포트한다.

    from keras.datasets import mnist
    
- 이미지 처리용 툴을 다루는 PIL의 서브패키지를 임포트한다.

      from PIL import Image
    
- 파이썬 기본 패키지들을 임포트한다.

      import numpy as np   # 행렬
      import math   # 수학
      import os   # 파일
      
- 케라스의 백엔드 패키지를 불러온다.

      import keras.backend as K
      K.set_image_data_format('channels_first')    # 이미지 데이터의 채널이 들어 있는 차원이 1번째가 되도록 설정
      print(K.set_image_data_format)
      
- 끝으로 tensorflow 패키지를 부른다.

      import tensorflow as tf   
     
   - 케라스는 엔진의 함수를 직접 불러서 사용하는 기능도 제공하기 때문에 tensorflow 패키지도 호출했다.
      
- 케라스는 다양한 인공신경망 관련 함수를 제공한다. 그러나 가끔은 사용자가 직접 만드는 경우도 있다. 이번 예의 생성망처럼 신경망의 출력이 스칼라나 벡터가 아니라 다차원일 경우에는 해당 차원에 맞는 손실함수가 필요하다. 4차원 데이터를 이용하는 손실 함수를 케라스 백엔드와 텐서플로로 구현한다.

      def mse_4d(y_true,y_pred):
        return K.mean(K.square(y_pred - y_true),  axis=(1,2,3))   # 평균자승오류 구함
        
      def mse_4d_tf(y_true,y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true), axis= (1,2,3))
      
   - 4차원 데이터이므로 배치 데이터를 나타내는 0축(axis)을 제외하고는 평균 계산이 다른 모든 축에 대해 이루어지도록 했다.
   
   - 케라스에서 제공하는 최소자승오류 계산 함수는 axis = -1, 즉 한 축의 평균을 구한다.   
   
   - 케라스 백엔드가 많은 함수를 제공하기 때문에 호환성을 높이는 차원에서 가능하면 케라스 백엔드로 추가 함수를 구성해주는 것이 좋지만 엔진 고유의 기능을 사용하고자 할 때는 직접 구성해도 좋다.
      
### 7.3.2 합성곱 계층 GAN 수행 ###      
      
2️⃣ 각 단계별 구현에 앞서 합성곱 계층 GAN의 수행 방법과 결과를 살펴보자.      
      
- 아규먼트를 효율적으로 입력받을 수 있도록 argparse 패키지를 임포트한다. 그리고 main()함수를 정의하여 입력 파라미터를 처리한다.

      import argparse 
      
      def main():
        parser = argparse.ArgumentParser()   # 인자값 받을 수 있는 인스턴스 생성
        
        # 입력받을 인자값 등록
        parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size for the networks')  
        parser.add_argument('--epochs', type = int, defalut = 100, help = 'Epochs for the networks')
        parser.add_argument('--output_fold', type = str, defalut = 'GAN_OUT', help = 'Output fold to save the results')   # 결과 저장 폴더
        parser.add_argument('--input-dim', type = int, defalut = 10, help = 'Input dimension for the generator')   # 무작위 벡터 길이
        parser.add_argument('n_train', type = int, defalut = 32, help = 'The  number of training data')   # 사용할 학습 데이터 수 
        
- 명령행으로 입력한 파라미터를 처리한 후 이 파라미터들로 학습을 수행한다.

      args = parser.parse_args()    # 입력받은 인자들을 args에 저장
      train(args)
      
### 7.3.3 합성곱 계층 GAN 모델링 ###        
      
3️⃣ GAN에 포함된 2가지 신경망(생성망과 판별망)을 모델링해보자.

- 우선 모델링에 필요한 케라스 서브패키지들을 불러온다.

      from keras import layers,models,optimizers
      
- 다음은 Sequential를 상속한 클래스를 만든다.

      class GAN(models.Sequential):
      
- 이제부터 클래스에서 제공하는 각 멤버 함수를 살펴보자. 클래스 초기화 함수를 만든다.

      def __init__(self, input_dim = 64):
        super.__init__()
        self.input_dim = input_dim
        
    - 입력 벡터의 크기를 파라미터로 받는다. 

- 생성망과 판별망을 모델링 하는 단계이다.

      self.generator = self.GENERATOR()
      self.discriminator = self.DISCRIMINATOR()
      
    - 두 신경망을 모델링하는 멤버 함수를 호출하여 모델을 만든다.

- **생성망은 판별망의 결과를 활용하여 학습한다**. 따라서 학습을 위한 생성망은 생성망과 판별망이 결합된 형태이다. 이렇게 결합된 생성망을 학습용 생성망이라고 부른다. 단, 결합 시 판별망 쪽은 학습이 진행되지 않도록 만든다. 학습용 생성망을 학습할 때는 이미 학습된 판별망을 사용하기 때문이다.    
      
      self.add(self.generator)
      self.discriminator.trainable = False
      self.add(self.discriminator)
      
지금까지 판별망 모델 1개, 순수 생성망 모델 1개, 판별망이 붙어 있는 생성망 모델 1개를 구현했다.

- 모델들을 사용하려면 컴파일 과정이 필요하다.

      self.compile_all()
      
- 전체 신경망을 컴파일하는 함수를 만든다.

      def compile_all(self):
        # 최적화 방법 정의
        d_optim = optimizers.SGD(lr = 0.0005, momentum = 0.9, nesterov = True)   # 최적화 인스턴스
        g_optim = optimizers.SGD(lr = 0.0005, momentum = 0.9, nesterov = True)    
        
        self.generator.compile(loss=mse_4d, optimizer = "SGD")  # 학습용 생성망 컴파일
        self.compile(loss='binary_crossentropy', optimizer= g_optim)    # 순수 생성망을 최적화 인스턴스를 사용해 최적화 
        
        # 판별망 컴파일
        self. discriminator.trainable = True
        self.discriminator.compile(loss= 'binary_crossentropy', optimizer = d_optim)
   
   - 모델을 정의했다고 모델이 생성되는 것은 아니다. 모델 생성은 컴파일 단계에서 이루어지기 때문에 케라스 인공신경망에 있어서 컴파일 단계는 필수이다.
   
   - 학습을 하지 않고 기존의 가중치를 사용한다고 해도 컴파일 단계를 반드시 수행해야 한다.
   
   - 모델들을 컴파일 하기에 앞서 컴파일에서 필요한 최적화 방법을 정의한다.
   
   - SGD 최적화 함수의 파라미터를 설정하여 판별망과 생성망의 학습에 사용될 최적화 인스턴스를 가져온다.
   
   - 이제 순수 생성망 뒤에 판별망을 붙인 학습용 생성망을 컴파일 한다.
   
   - 순수 생성망은 학습 시 최적화 인스턴스인 g_optim을 사용해 최적화한다.
   
   - 판별망은 학습용 생성망이 학습될 때는 자신은 학습되지 않고 고정되어 있으면서 생성망의 학습을 돕는다. 그래서 학습용 생성망이 학습되는 동안 판별망이 학습되지 않도록 막았다.
   
   - 이번에는 판별망을 학습하므로 판별망의 가중치를 최적화 해야한다. 그래서 컴파일 단계에서 학습 옵션(trainable)을 다시 켜주었다.
   
✔ 지금까지 3가지 신경망을 컴파일하는 코드를 만들었다. **주의할 점은 학습용 생성망은 순수 생성망과 판별망이 결합된 형태로 새로 추가된 신경망 구조와 가중치가 없다는 점이다.** 따라서 실제 학습은 판별망과 학습용 생성망에 대해서만 진행된다. 
      
- 생성망을 정의하는 함수를 만들어보자.  

      def GENERATOR(self):
        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024,activation='tanh', input_dim = input_dim))
        model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((128,7,7), input_shape = (128 * 7 * 7,)))
        model.add(layers.UpSampling2D(size=(2,2)))
        model.add(layers.Conv2D(64,(5,5),padding='same', activation='tanh'))
        model.add(layers.UpSampling(size = (2,2)))
        model.add(layers.Conv2D(1,(5,5),padding='same', activation='tanh'))

        return model

   - 합성곱 방식을 이용해 다음과 같은 순서로 생성망을 구성했다.
   
    1) 생성에 사용될 입력은 1차원 행렬이다. 이 1차원 행렬은 input_dim만큼의 원소로 구성된다.
    
    2) 완전 연결 계층을 사용하여 입력 행렬 데이터를 1024개의 원소로 확장한다.
    
    3) 또 한번 완전 연결 계층 이용해 6772로 더 확장한 후 이미지의 채널, 가로 및 세로를 128 * 7 * 7 모양의 3차원 행렬로 재조정한다.
    
    4) 배치 정규화를 위한 BatchNormalization()계층을 포함한다.
    
    5) 데이터 모양을 (128,7,7)로 재조정한다.
    
    6) UpSampling2D()로 이미지 가로 및 세로를 각각 2배 확대한다.
    
    7) 64개의 5 * 5 크기의 2차원 필터로 구성된 합성곱 계층을 적용한다.
    
    8) 또 한번 이미지를 (2,2)배 확장한다.
    
    9) 필터 하나로 구성된 5 * 5 합성곱 계층을 적용하여 최종 허구 이미지를 출력하도록 모델을 구성한다.
    
    10) 구성된 모델을 출력값으로 반환한다.

- 입력된 이미지가 실제 이미지인지 생성망으로 만든 이미지인지를 판별하는 판별망 모델을 생성하는 멤버 함수를 만든다.

      def DISCRIMINATOR(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64,(5,5),padding='same', activation='tanh', input_shape = (1,28,28)))   # 64 * 28 * 28 
        model.add(layers.MaxPooling2D(pool_size = (2,2)))    # 64 * 14 * 14 
        model.add(layers.Conv2D(128,(5,5),activation='tanh'))   # 128 * 14 * 14 
        model.add(layers.MaxPooling2D(pool_size = (2,2))   # 128 * 7 * 7 
        model.add(layers.Flatten())   # 3차원 -> 1차원
        model.add(layers.Dense(1024,activation='tanh'))   # 완전 연결 계층
        model.add(layers.Dense(1,activation='sigmoid'))   # 이진 판별

        return model
        
     1) 입력받은 이미지에 5 * 5필터 64개로 합성곱을 적용한다. 입력 데이터는 채널이 하나인 28 * 28 크기의 이미지이다. 64개의 합성곱 필터를 적용하므로 결과는 28 * 28크기의 추상화된 이미지 64개이다.
     
     2) 이 결과에 맥스풀링을 적용해 가로 및 세로를 반으로 줄인 14 * 14 크기의 이미지를 64개 만든다.
     
     3) 여기서 적용한 맥스풀링은 28 * 28 크기의 이미지를 2 * 2 픽셀 단위로 나누어 14 * 14개 픽셀군으로 나눈 뒤 각 군에서 가장 큰 값을 출력하기 때문에 이미지 가로세로가 반씩 줄어든 것이다.
     
     4) 또 한 번 필터 128개로 구성된  5 * 5크기의 합성곱 계층을 곱하고 그 결과에 맥스풀링을 적용한다. 이렇게 되면  7 * 7 크기의 이미지 128개가 나온다. 다시 말해 7 * 7 크기의 이미지들로 구성된 특징점 정보 128개가 나온 것이다.
     
     5) 여기에 완전 연결 계층을 적용하고자 3차원을 1차원으로 바꾸어주는 Flatten()단계를 수행한다.
     
     6) 노드가 1024개인 완전 연결 계층을 적용하고 그 결과를 최종 노드 1개로 구성된 완전 연결 계층으로 보낸다.
     
     7) 최종 완전 연결 계층은 이진 판별을 수행한다.
     
     8) 끝으로 만들어진 모델을 반환한다.

GAN은 무작위 신호를 입력으로 받아들인다. 무작위 신호를 이용해서 다양한 출력 이미지를 생성한다. 각 출력 이미지의 생성 확률은 적용할 이미지 데이터셋을 따르도록 학습하는 것이 GAN의 목적이다.

- 이제 무작위 신호를 만드는 멤버 함수를 만든다.

      def get_z(self,ln):    # 주어진 input_dim만큼 원소를 가지는 무작위 벡터를 만듦
        input_dim = self.input_dim
        return np.random.uniform(-1,1,(ln,input_dim))    # 배치 크기인 ln으로 무작위 벡터 수를 설정함

- 모델들의 학습을 수행하는 함수를 만들 차례이다. 이 함수는 판별망과 학습용 생성망 둘을 순서대로 학습한다. 먼저 판별망 학습 부분을 구현한다.

      def train_both(self,x): 
        ln = x.shape[0]   # 배치 크기
        # first trial for training discriminator
        z = self.get_z(ln)
  
   - 입력 이미지 데이터의 수를 파악하고 그 수 만큼 무작위 값을 가지는 벡터를 만든다.

- 이 벡터들을 입력으로 하여 각 벡터에 해당하는 이미지들을 생성한다.

      w = self.generator.predict(z, verbose = 0)

  - 이렇게 만들어진 이미지들은 실제 이미지가 아니기 때문에 허구 이미지라고 한다.

- 판별망 학습을 시작한다. 판별망은 실제 이미지인지 허구 이미지인지를 구별하도록 학습하므로 실제 이미지들과 허구 이미지들을 데이터셋 하나로 합친다. 그러고 나서 분류망 학습에 사용하려고 합친 데이터셋에 각 이미지들에 해당하는 레이블 벡터를 만든다.

      xw = np.concatenate((x,w))
      y2 = [1] * ln + [0] * ln
      
    - 벡터의 앞쪽 절반 원소들은 실제 이미지들의 레이블이고 나머지 절반은 허구 이미지들의 레이블에 해당한다.
    
    - 레이블 벡터의 길이는 합쳐진 이미지 데이터셋 길이와 같다. 즉 배치 크기의 2배이다.
    
- 각 배치 단위로 학습하는 코드를 구현한다.

      d_loss = self.discriminator.train_on_batch(xw,y2)

  - 배치 단위로 무작위 벡터인 w를 생성해야하고 분류망 학습 다음에 학습용 생성망 학습이 필요하기 때문에 배치 단위로 학습을 진행한다.
  - 🔔 주어진 입력에 대해 무작위 입력 벡터를 만들고 이에 대해 판별망을 학습하고 이 둘을 바탕으로 학습용 생성망을 학습하는 단계로 매 배치마다 전체 학습이 진행된다.

- 이제 학습용 생성망의 학습을 진행할 차례이다. 다시 한 번 배치 수만큼 무작위 벡터를 생성한다.

      z = self.get_z(ln)   # 배치 수 만큼 무작위 벡터 생성
  
   - 이 무작위 벡터를 학습에 제공한다. 그렇게 하면 생성할 이미지들이 실제 이미지에 가까워지도록 생성망의 가중치들을 조정하면서 학습을 진행하게 된다.

- 학습에 앞서 학습용 생성망에 포함된 판별망의 가중치들이 학습되지 않도록 학습 플래그를 끈다.

      self.discriminator.trainable = False

- 앞서 만든 z를 통해 학습용 생성망을 배치 단위로 학습시킨다. 학습이 끝나면 다음 단계인 판별망의 별도 학습에서는 판별망의 가중치가 반영되어 학습이 진행되도록 학습 플래그를 다시 켜준다.

       g_loss = self.train_on_batch(z,[1] * ln)
       self.discriminator.trainable = True

지금까지 판별망과 학습용 생성망이 1회씩 학습되도록 만들었다. 끝으로 학습 과정에서 구해진 손실값들을 반환하여 손실의 변화를 모니터링 한다.

      return d_loss, g_loss

### 7.3.4 합성곱 계층 GAN 학습 수행 ### 

4️⃣ 앞서 만든 GAN의 모델링과 학습용 클래스를 사용해 실제로 GAN을 학습하는 코드를 구현한다.      
      
- 학습 수행에 필요한 파라미터들을 로컬 변수에 저장한다.      
      
      def train(args):
        BATCH_SIZE = args.batch_size
        epochs = args.epochs
        output_fold = args.output_fold    # 학습과정에서 생성된 이미지 중 일부를 간헐적으로 출력하는 폴더의 이름
        input_dim = args.input_dim   # 무작위 벡터의 길이
        
        os.makedirs(output_fold, exist_ok = True)
        print('Output_fold is', output_fold)
      
- MNIST 데이터를 불러오고 전처리를 한다.

      (X_train,y_train), ( _ , _ ) = mnist.load_data()
      X_train = (X_train.astype(np.float32) - 127.5) / 127.5
      X_train = X_train.reshape((X_train.shape[0],1) + X_train.reshape[1:])   # 채널 차원 추가
        
 - MNIST 데이터셋을 불러오고 0에서 255까지 정수로 된 이미지 각 필셀이 -1부터 1사이의 실수가 되도록 했다.
 
 - 합성곱 계층의 처리를 위해 흑백 이미지이지만 채널 차원을 하나 추가 했다       
        
- GAN 모델의 인스턴스를 만든다.

      gan = GAN(input_dim)

- 학습의 진행을 위해서 에포크 단위로 1차 순환문을 구성한다.

      d_loss_ll = []
      g_loss_ll = []
      for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []

- 케라스에서 model.fit()함수로 수행하면 에포크 단위로 자동으로 진행되지만 여기서는 GAN구조의 특성상 에포크 단위로 처리하는 model.train_on_batch()함수로 구현을 하고 있어 에포크 단위의 순환문을 만든다.

      for index in range(int(X_train.shape[0] / BATCH_SIZE)):
        x = get_x(X_train,index, BATCH_SIZE)   # x : 배치 크기 만큼의 입력 데이터
        
        d_loss, g_loss = gan.train_both(x)   # gan에 전달
        
        d_loss_l.append(d_loss)
        g_loss_l.append(g_loss)

- 에포크가 매 10회가 진행될 때 마다 결과 이미지를 파일로 저장한다. 단 마지막 에포크는 10번 이전에 종료되더라도 결과를 저장한다.

      if epoch % 10 == 0 or epoch  == epochs -1 :
        z = gan.get_z(x.shape[0])
        w = gan.generator.predict(z, verbose = 0)
        save_images(w, output_fold, epochm index)
        
   - 무작위 잡음인 z를 생성하고 이 z를 이용해 현재 GAN의 생성망에서 새로운 이미지들을 생성했다.
   
   - 생성된 이미지들을  외보에서 볼 수 있도록 파일로 저장했다.

- 배치 단위로 저장된 손실값들을 신경망별로 에포크 단위 리스트에 저장한다.

      d_loss_ll.append(d_loss_l)
      g_loss_ll.append(g_loss_l)
      
- 순환문이 끝나면 가중치와 손실값을 저장한다.

      gan.generator.save_weights(output_fold + '/' +'generator', True)
      gan.discriminator.save_weights(output_fold + '/' + 'discriminator', True)
      
      np.savetext(output_fold + '/' + 'd_loss', d_loss_ll)
      np.savetext(output_fold + '/' + 'g_loss', g_loss_ll)
      
    - 먼저 save_weights()함수를 이용해 각 신경망별 모델들의 가중치를 저장하고 모아둔 손실값들도 저장한다.






