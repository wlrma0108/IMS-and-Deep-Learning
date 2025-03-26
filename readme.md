지능형 멀티미디어 시스템 프로젝트

2025-03 ~ 2025-06

프로젝트의 목표는 cnn기반 모델:vgg, convnext, dense, alex 와 visiontransformer을 비교해 보는 것 총 5개의 모델에 같은 데이터셋을 적용하여 어떤 모델이 가장 적합한지 파악하는 것이 프로젝트의 목표이다. 이를 장고를 통해 배포하고 결과를 확인 할 수 있게한다. 

1. 폐 ct사진 분류 인공지능 사용
2. 모델을 장고를 통해서 배포
3. 도커와 쿠버, 클라우드를 통해서 관리

kaggle의 베이스라인을 끌고 올려했지만 배포를 해야하는 프로젝트에서 적합하지 않다고 생각된다. 
참고를 하되 전처리, 시각화 코드 일부를 참고하고 나머지는 다른 참고자료를 찾는다. 
django를 통해 앱을 생성 및 관리 할려했지만 문제가 생각보다 많이 생긴다. 모델과 전처리 코드가 함께 동작하는건가?
tensorflow는 파이썬의 버전문제로 인해 사용이 귀찮을 듯 하다. torch, torchvision을 사용하자. 
sklearn을 사용하여 모델을 배포하는 자료가 생각보다 많다. 이유를 찾아보자. 

기존의 프로젝트 설계는
1. pytorch를 통해 모델을 pth파일로 변환
2. djnago를 통해 업로드, json을 통해 송수신
3. 도커와 aws를 통해 배포 였다.

모델을 만드는 것은 크게 어렵지 않을 것 같다. 다만 장고를 통해 배포하고 몇가지 기능을 집어 넣는게 생각보다 어렵다. 단순 배포는 간단하다. 

사용스택
pytorch, django , nginx, docker, aws or azure
