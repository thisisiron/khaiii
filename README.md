khaiii-keras
====
khaiii-keras는 카카오에서 개발한 세 번째 형태소분석기를 케라스 형태로 변환하여 학습, 평가합니다. 해당 프로젝트의 목적은 **다양한 모델을 구현**하여 기존의 카카오 CNN모델과 비교하는 것입니다.

Pytorch로 구현된 카카오 형태소 분석기를 Keras로 변환하여 Keras 프레임워크 사용자라면 쉽게 모델을 구현하여 평가할 수 있도록 만들고 있습니다.

실제 구현된 형태소 분석기를 사용하도록 배포하는 프로젝트가 아니라 여러 모델을 구현 및 비교하여 테스트해볼 수 있는 프로젝트입니다. 카카오 형태소 분석기를 사용하고 싶으시다면 [다음](https://github.com/kakao/khaiii) 링크를 통해 들어가셔서 사용하시면 됩니다.


데이터 기반
----
기존 버전이 사전과 규칙에 기반해 분석을 하는 데 반해 khaiii는 데이터(혹은 기계학습) 기반의 알고리즘을 이용하여 분석을 합니다. 학습에 사용한 코퍼스는 국립국어원에서 배포한 [21세기 세종계획 최종 성과물](https://ithub.korean.go.kr/user/noticeView.do?boardSeq=1&articleSeq=16)을 저희 카카오에서 오류를 수정하고 내용을 일부 추가하기도 한 것입니다.

전처리 과정에서 오류가 발생하는 문장을 제외하고 약 85만 문장, 천만 어절의 코퍼스를 사용하여 학습을 했습니다. 코퍼스와 품사 체계에 대한 자세한 내용은 [코퍼스](https://github.com/kakao/khaiii/wiki/%EC%BD%94%ED%8D%BC%EC%8A%A4) 문서를 참고하시기 바랍니다.


기존 카카오 형탯오 분석기 알고리즘
----
기계학습에 사용한 알고리즘은 신경망 알고리즘들 중에서 Convolutional Neural Network(CNN)을 사용하였습니다. 한국어에서 형태소분석은 자연어처리를 위한 가장 기본적인 전처리 과정이므로 속도가 매우 중요한 요소라고 생각합니다. 따라서 자연어처리에 많이 사용하는 Long-Short Term Memory(LSTM)와 같은 Recurrent Neural Network(RNN) 알고리즘은 속도 면에서 활용도가 떨어질 것으로 예상하여 고려 대상에서 제외하였습니다.

CNN 모델에 대한 상세한 내용은 [CNN 모델](https://github.com/kakao/khaiii/wiki/CNN-%EB%AA%A8%EB%8D%B8) 문서를 참고하시기 바랍니다.

## TODO
- [X] 가상 음절 수정

수정 후 가상 음절 테이블

| 가상 음절 | 의미                   |
|-----------|------------------------|
| \<u\>       | Out of Vocabulary 음절 |
| \<w\>       | 어절 경계              |
| \<cls\>     | 문장의 시작            |
| \<sep\>     | 문장의 마침            |

- [X] 데이터 형태 변환

|                         | -7    | -6  | -5 | -4  | -3 | -2  | -1  | 0  | 1   | 2  | 3   | 4    | 5  | 6  | 7   |
|-------------------------|-------|-----|----|-----|----|-----|-----|----|-----|----|-----|------|----|----|-----|
| 기존 카카오 데이터 형태 | \<s\>   | \<s\> | 프 | \<u\> | 스 | 의  | \<w\> | 세 | \<u\> | 적 | 인  | \</w\> | 의 | 상 | 디  |
| 변환 후 데이터 형태     | \<cls\> | 프  | 랑 | 스  | 의 | \<w\> | 세  | 계 | 적  | 인 | \<w\> | 의   | 상 | 디 | ... |

기존의 CNN 모델에 맞춘 형태의 데이터를 일반적으로 MAX_LEN을 설정 후 부족하다면 <PAD>를 넣고 초과한다면 해당문자까지 자르는 형태로 변환

- [X] 케라스로 간단한 모델 구현 후 학습
- [ ] Pytorch 의존성 삭제
- [X] 띄어쓰기 처리 문제
Input 음절마다 형태소를 부여하기 때문에 공백의 경우 그에 맞는 가상의 품사를 태깅하도록 함 -> I-PAD
- [ ] F1 Score, 음절 매칭 점수 

License
----
This software is licensed under the [Apache 2 license](LICENSE), quoted below.

Copyright 2018 Kakao Corp. <http://www.kakaocorp.com>

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this project except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.
