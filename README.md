# Hustar_ML_contest
## 구성원 : 강나현, 권헌진, 김민호, 오승환
### 6.07(화)
- custom model 학습 종료 (public accuracy 0.968299)
> - 단일 모델로써는 비약적인 성능 향상
> - 다만 학습 종료때까지 val loss가 꾸준히 감소
>   - underfitting의 우려가 있음
> - custom model을 추가하여 Hard Voting (public accuracy 0.975255)

### 6.06(월)
- custom model 구현 성공 (학습중)
> - classifier의 parameter를 늘림
> - data는 민호의 랜덤샘플 30만개(이전 무작위 데이터 수량 감소), epoch=3
- 추가 제출 (public accuracy 0.973826)
> - 지금까지 만든 모델 중 acc 0.95 이상인 모델들을 Hard Voting
> - 총 5개의 모델이 Ensemble됨. 성능 향상이 극적이었음.
>   - 앞으로 만들어질 모든 모델들의 Ensemble도 기대됨

### 6.05(일)
- 추가 제출 (public accuracy 0.958764)
> - 신규 데이터는 각 문제별 1000개의 무작위 negative pair, positive pair를 구성
>   - 총 train 60만, test 6만
- 신규 데이터로 학습했더니 약간의 성능 상승이 있었음
> - 기존 데이터는 약 500만개라 토크나이징에 너무 큰 시간이 소요됨
> - 또, 기존 데이터는 BM25기반 유사도 측정으로 생성된 데이터
> - 그러나 Source code에서 BoW와 TF-IDF 기반 유사도 측정은 의미가 적다 판단됨
>   - 학습 시간 단축 및 모델 성능 향상

### 6.02(목)
- 추가 제출 (public accuracy 0.954498)
> - 기존 데이터에 left truncation 추가
> - 코드 앞부분은 주로 외부 라이브러리의 import 부분이라 크게 의미가 없음
> - 따라서 truncation이 left가 되었을 때 성능이 약간 상승

### 6.01(수)
- 추가 제출 (public accuracy 0.929512)
> - 다른 조건은 26일 제출과 같음. 모델만 codeberta small을 사용
>   - graphcodebert의 parameter는 약 1.2억개. 그러나 codeberta small은 약 8천만개.
>   - 그래서 학습속도만 두고 봤을 땐 약 2.3배 가량 빨랐음.
>   - 하지만 어쩔 수 없는 parameter 갯수 차이 때문인지 public accuracy가 상당히 차이남(약 2%)
>   - 즉, submission 용도로는 부적합. 그러나 test용도로는 빠른 학습덕에 써 볼만 해 보임

### 5.26(목)
- 추가 제출 (public accuracy 0.9467)
> - Hyperparameter : lr=1e-5, patience=20
>   - 학습 도중 step-checkpoint를 저장할 용량이 부족해서 학습이 강제 중단됨.
>   - 그럼에도 불구하고 기존 제출물에 비해 눈에 띄게 정확도가 오름
>   - 데이터가 매우 많고, 모델 깊이가 어마어마하기때문에 일단 학습을 오래시키는 것 만으로도 효과적인 성능 향상이 존재
>   - 즉, 가능한 학습을 오래 시켜볼 필요성이 있음 (=overfitting의 발생 가능성이 다소 낮음)
> - 일단 현재 저장된 checkpoint의 parameter에서 새로 구성된 dataset에 대한 실험을 하는것도 괜찮지 않을까 하는 생각이 생김
>   - 모델 경량화 보류. 대신 마지막 classificationhead나 optimizer, loss 측정함수 등을 개선해 보는 쪽이 더 좋을 것 같음

### 5.25(수)
- 승환 추가 제출 (public accuracy 0.930006)
> - early_stopping_patience=5로 늘리고 적용시켰더니 점수가 조금 더 올랐음. 예상대로 기존의 방식은 학습이 너무 빨리 종료됨
> - patience를 20으로 해보고 학습 시켜볼 예정
- 민호 첫 제출 (public accuracy 0.89497)
> - positive, negative pairs의 dataset 크기를 약 60만개로 줄임
> - RoBertaForSequenceClassification 대신 AutoModelForSequenceClassification 사용
> - MAX_LEN = 256, train_batch_size=32, eval_batch_size=32, early_stopping_patience = 5로 설정
>   - MAX_LEN이 작아서 256 이상 코드에 대한 학습이 저조하여 score가 낮게 나온 것으로 예상
>   - early_stopping_patience가 커지면 모델을 저장하는 크기도 그만큼 커져야 최적의 모델을 저장할 수 있을 듯
- 할 일
> - dataset 변경(더 좋은 negative pair와 positive pair)
>   - BM25가 아닌 다른 기법으로도 유사도를 측정해 보는 것도 좋은 방법이 될 수 있음
> - 모델 경량화
>   - 기존 모델은 너무 커서 학습 시간이 오래 걸림
>   - 해당 모델을 경량화 시키면 기존 모델의 특성을 어느정도 유지하면서 학습 시간을 대폭 줄일 수 있을 것으로 기대됨
>   - 경량화 모델에서 여러 실험을 한 후 좋은 결과를 보여준 데이터셋과 하이퍼파라미터로 본 모델에 학습시키기
> - Classifier 개선 (우선도 下)
>   - 현재 사용중인 모델은 마지막의 RobertaClassifier가 분류를 시행중
>   - 언어 모델 전체 개선은 어렵더라도 마지막 분류기를 개선하는 것은 가능할지도 모름
>   - 그러나 세계의 천재들이 만들어낸 모델 구조를 뛰어넘는 것은 쉽지 않을 것으로 보임. 따라서 우선도가 낮음

### 5.24(화)
- 나현 첫 제출 (public accuracy 0.49929)
> - 모델을 전혀 학습시키지 않고 바로 predicting
> - 이 경우에는 결과가 별로 좋지 않았음
> - 최소한의 학습은 있어야 할 것으로 보임
- 승환 첫 제출 (public accuracy 0.92825)
- 생각한 내용
> - 학습환경은 2080 두개. batch size를 작게 쓸 수 밖에 없었음 (model이 너무 크다...)
> - 그래서 hyperparameter가 max_len=512, train_batch=4, eval_batch_size=16, patience=2, eval_steps=500으로 설정됨
> - 이에따라 early stopping이 3000 step에서 진행됐는데, 너무 일찍 되지 않았나 하는 우려가 있음
- 할 일
> - max_len을 256으로 줄이고 해본다면 오히려 어떨까?
> - max_len을 512로 유지하되 dataset의 크기를 줄이고 batch_size를 키울 수 있다면 더 좋을것 같은데
> - distillation, quantization, pruning, weight sharing 등 경량화 기법을 사용해서 모델 자체를 가볍게 하는게 좋을수도?
> - early_stopping_patience=2는 너무 작긴함. 좀 더 키우고 학습시켜보자

### 05.23(월)
- Tokenize에 시간이 상당히 소요됨
- max_len 512로 tokenize해서 두개의 문제를 cross-encording하는 기법을 사용하기로 함

### 05.22(일)
- 코드 유사성 판단 AI 경진대회 참가 (Link : https://dacon.io/competitions/official/235900/overview/description)
- EDA 개시
- graphCodeBert 사용하기로 결정

### 05.02(월)
- Git repository 생성
- 프로젝트 목표 설정
