---
layout: single
excerpt: ""
title: "제목으로 상품 카테고리 추천하기"
date: 2020-04-09 13:50:00 +0900
tags: scikit-learn linear-regression fasttext category-suggester
---

우리가 주위에서 쉽게 볼 수 있는 e-commerce 서비스에서는 다양하고 수많은 상품이 등록된다. 서비스에 등록된 상품들은 원활한 검색을 위해 정확한 특정 카테고리로 분류될 필요가 있지만, 판매자나 관리자가 일일히 올바른 카테고리를 찾아 분류를 하는 것은 비효율적이다. 

이 글에서는 이러한 문제를 해결하기 위해 판매자가 등록한 상품의 제목과 카테고리를 훈련 예로 삼아 학습하고 학습된 모델를 이용하여 카테고리를 추천해주는 과정을 공유한다. 그 과정중에는 약 3년 전 개발을 시작했을 때부터 최근 진행한 추가 개선까지 진행하고 경험했던 내용을 같이 작성하고자 한다.

# working with text data

진행했던 과정을 공유하기 앞서, 이 기능을 만들기 위해 참고했던 글 하나를 소개하고자 한다.

[working with text data] 에서는 (문서, 주제)의 쌍을 이용하여 문서의 주제를 추론하기 위해 sklearn의 다양한 기능을 사용하는 과정을 설명한다. 이 과정은 아래와 같이 6가지로 구분하여 설명하고 있다.

1. 학습에 필요한 데이터 쌍(텍스트, 카테고리) 불러오기
1. feature 추출 
1. 분류기(classifier) 훈련 
1. 파이프라인 구축
1. 정확도 평가
1. 탐욕 탐색(grid search)를 이용한 파라메터 최적화

## 학습에 필요한 데이터 불러오기

[working with text data]에서는 학습과정을 설명하기 위해 `Twenty Newsgroups`라는 데이터 셋을 이용한다. `Twenty Newsgroups`는 텍스트 분류나 클러스터링에 자주 사용되는 샘플로 약 20000개의 글을 20개의 토픽으로 구분되어있는 데이터 셋으로 아래와 같이 쉽게 가져올 수 있다.

```
>>> from sklearn.datasets import fetch_20newsgroups
>>> twenty_train = fetch_20newsgroups(subset='train',
...     categories=categories, shuffle=True, random_state=42)
```

## feature 추출

학습 대상이 되는 데이터는 학습을 진행하기 전에 알맞은 형태로 데이터 가공을 해야 한다. 이 글에서는 그 과정을 아래와 같이 3단계로 나누어 설명한다.

1. Bow(Bags of words)
1. scikit-learn을 이용한 형태소 분석
1. TD-IDF를 이용한 빈도 수 계산

#### Bow(Bags of words)

텍스트를 학습하기 용이한 형태로 변경하기 위해서 하는 작업으로 문서를 구성하는 단어를 ID(unique integer)로 치환하고 각 문서마다 (ID, 등장 횟수)의 쌍으로 데이터를 변경한다. 이렇게 변경된 값은 단어와 단어의 횟수로 구분되므로 단어의 순서는 고려하지 않고 순수하게 단어의 등장 횟수만을 고려한다. 아래의 예를 살펴보자.

|                  | 0(나이키) | 1(아디다스) | 2(신발) | 3(파란색) |
|------------------|-----------|-------------|---------|-----------|
| 0(나이키 신발)   | 1         | 0           | 1       | 0         |
| 1(아디다스 신발) | 0         | 1           | 1       | 0         |
| 2(파란색 나이키) | 1         | 0           | 0       | 1         |

`나이키 신발`, `아디다스 신발`, `파란색 나이키`라는 문서가 있다고 가정하자. 행(row)는 문서, 열(column)은 단어 ID를 의미한다. 각 문서는 문서를 구성하는 단어를 숫자로 변환하여 위와 같은 matrix를 생성할 것이다. 문서의 양이 늘어나고 그에 따라 사용되는 단어가 늘어다면 feature(이 예에서는 숫자로 변환한 단어)의 수가 늘어나고 0로 표기된 값(불필요하게 메모리를 차지하는 값)들이 늘어나게 된다. 이렇게 sparse한 matrix를 효율적으로 처리하기 위해 scipy.sparse를 이용하여 저장한다. scipy.sparse를 이용하면 아래와 같이 0이 아닌 matrix의 값만을 압축하여 저장함으로써 저장공간을 절약할 수 있다.

#### scikit-learn을 이용한 형태소 분석

각 문서의 (단어, 등장 횟수)의 쌍은 일일히 구할 수 있지만, sklearn의 `CountVectorizer`를 이용하면 쉽게 구할 수 있다. 뿐만 아니라, 다양한 옵션을 이용하여 ngram이나 별도의 형태소 분석기를 활용하여 단어를 분리할 수도 있다. `CountVectorizer`는 아래와 같이 쉽게 사용할 수 있고 자세한 설명은 [count vectorizer]를 참고하면 된다.

```
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> count_vect = CountVectorizer()
>>> X_train_counts = count_vect.fit_transform(twenty_train.data)
>>> X_train_counts.shape
(2257, 35788)
```

#### TD-IDF를 이용한 빈도 수 계산

단어의 발생 빈도 수는 문서 분류를 위한 학습에 도움이 되지만, 문서 텍스트가 길어지만 길어질수록 여러 단어를 포함할 가능성이 높기 때문에 변별력이 떨어지게 된다. 이러한 문제를 해결하기 위해 중요도가 높은(변별력이 높은) 단어를 계산할 수 있는 TF-IDF를 이용할 수 있다.

TF-IDF는 TF와 IDF를 곱한 값으로, TF(term frequency)는 특정 문서에서 특정 단어가 등장한 횟수를 의미하고, IDF(inverse document frequency)는 특정 단어가 전체 문서에서 등장한 횟수에 반비례하는 수를 의미한다. 즉, TF-IDF는 특정 단어가 전체적으로 자주 등장하면 중요도를 줄이고 해당 문서에서 자주 등장하면 중요도를 올려서 특정 문서에서만 중요도를 가지는 단어에 가중치를 주게 된다.

앞서 `CountVectorizer`에서 구한 count matrix를 아래와 같이 `TfidfTransformer` 모듈에 넣으면 TF-IDF가 적용된 결과를 아래와 같이 쉽게 구할 수 있다.

```
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> tf_transformer = TfidfTransformer().fit(X_train_counts)
>>> X_train_tf = tf_transformer.transform(X_train_counts)
>>> X_train_tf.shape
(2257, 35788)
```

## 분류기(classifier) 훈련

자 이제 학습을 위한 훈련 예는 정제되어 준비되었다. 이제는 정제된 훈련 예를 이용하여 카테고리를 예측할 수 있도록 다양한 학습 모델을 이용할 수 있다. 학습 모델은 SVD, linear regression 등 다양한 모델이 있으니 테스트를 해보고 적합한 모델을 선택하면 된다. 아래에서는 Naive Bayes 모델을 적용하여 학습 결과를 보여준다. 

```
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

>>> docs_new = ['God is love', 'OpenGL on the GPU is fast']
>>> X_new_counts = count_vect.transform(docs_new)
>>> X_new_tfidf = tfidf_transformer.transform(X_new_counts)

>>> predicted = clf.predict(X_new_tfidf)

>>> for doc, category in zip(docs_new, predicted):
...     print('%r => %s' % (doc, twenty_train.target_names[category]))
...
'God is love' => soc.religion.christian
'OpenGL on the GPU is fast' => comp.graphics
```

## 파이프라인 구축

지금까지 설명한 과정 중 훈련 예를 가공하고 학습모델을 학습시키는 과정은 파이프라인을 구축하여 아래와 같이 쉽게 이용할 수 있다

```
>>> from sklearn.pipeline import Pipeline
>>> text_clf = Pipeline([
...     ('vect', CountVectorizer()),
...     ('tfidf', TfidfTransformer()),
...     ('clf', MultinomialNB()),
... ])
```

## 정확도 평가

학습된 결과물은 검증 예(test example)을 이용하여 정확도를 평가할 수 있다. 아래의 예에서는 Naive Bayes 모델과 SVM 모델을 이용하여 학습한 모델의 정확도를 비교하는 코드를 볼 수 있다.

```
>>> import numpy as np
>>> twenty_test = fetch_20newsgroups(subset='test',
...     categories=categories, shuffle=True, random_state=42)
>>> docs_test = twenty_test.data
>>> predicted = text_clf.predict(docs_test)
>>> np.mean(predicted == twenty_test.target)
0.8348...

>>> from sklearn.linear_model import SGDClassifier
>>> text_clf = Pipeline([
...     ('vect', CountVectorizer()),
...     ('tfidf', TfidfTransformer()),
...     ('clf', SGDClassifier(loss='hinge', penalty='l2',
...                           alpha=1e-3, random_state=42,
...                           max_iter=5, tol=None)),
... ])

>>> text_clf.fit(twenty_train.data, twenty_train.target)
Pipeline(...)
>>> predicted = text_clf.predict(docs_test)
>>> np.mean(predicted == twenty_test.target)
0.9101...
```

Naive Bayes 모델을 이용한 경우 83.5%의 정확도를 보였지만, SVM 모델을 이용한 경우 약 91%의 정확도를 보인다. 이와 같이 동일한 훈련 예라고 하더라도 학습 모델에 따라 성능이 다르므로 훈련 예에 맞는 학습 모델 선정을 위해 테스트를 해보는 것이 중요하다.

## 탐욕 탐색(grid search)를 이용한 파라메터 최적화

학습의 결과물은 학습 모델 뿐만 아니라 어떻게 훈련 예를 feature engineering을 하는지 하이퍼 파라메터(사용자가 직접 세팅해주는 파라메터)를 어떤 값으로 설정하는지에 따라 다르다. 그러므로 다양한 조건으로 학습을 진행하여 정확도를 확인하고 가장 정확도가 높은 조합을 찾아야 한다.

보통 하이퍼 파라메터는 최적화된 값을 구하는 규칙이 있지 않고 다양한 경험을 통해 최적의 하이퍼파라메터를 찾을 수 있다. 여기서는 CounterVectorizer의 ngram 크기, TF-IDF의 IDF 여부, 학습 모델의 alpha 값을 들고 있다. 이 값들은 최적의 값을 실험하기 전에는 알기 어려우므로 값을 조정하여 여러 번 테스트를 해줘야 하는데 탐욕 검색(GridSearch)을 이용하면 이 과정을 간단하게 처리할 수 있다.

```
>>> from sklearn.model_selection import GridSearchCV
>>> parameters = {
...     'vect__ngram_range': [(1, 1), (1, 2)],
...     'tfidf__use_idf': (True, False),
...     'clf__alpha': (1e-2, 1e-3),
... }

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

>>> gs_clf.best_score_
0.9...
>>> for param_name in sorted(parameters.keys()):
...     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
...
clf__alpha: 0.001
tfidf__use_idf: True
vect__ngram_range: (1, 1)
```

# 실제 서비스에 적용하기

위에서 살펴본대로 [working with text data]에서는 (문서, 주제) 쌍을 훈련 예로 삼아 학습을 하고 학습 결과를 이용하여 문서의 알맞은 주제를 예측하는 방법에 대해 설명하고 있다. 이 방법을 활용하면 상품 제목을 이용하여 상품 카테고리도 추론할 수 있을 것이라고 예상했다. 만약 상품 제목으로 상품 카테고리를 정확하게 추론할 수 있으면 개인이 상품을 등록하는 중고거래에서 상품을 등록하는 과정을 간소화할 수 있고, 판매자가 잘못된 카테고리에 상품을 등록하는 문제를 해결할 수 있을 것이라 예상했다. 

우선 (상품 제목, 상품 카테고리)의 훈련 예를 준비했다. 훈련 예는 서비스에 등록된 상품 중 최근에 등록된 상품을 조회하여 추출하였고, 한글, 영문, 숫자를 제외하고는 모든 문자를 제거하여 불필요한 문자들을 제거했다. 준비한 훈련 예는 8:2의 비율로 나누어 80%만 훈련 예로 사용하고 나머지는 검증을 위한 검증 예로 남겨주었다.

훈련 예를 이용하여 CountVectorizer, TfidfTransformer, Naive Bayes 모델을 이용하여 동일하게 학습을 진행해보았다. 

50프로의 정확도밖에 나오지 않는다. 무엇이 잘못된 것일까? [working with text data]와 내가 개발한 기능과 차이점을 비교하며 하나씩 개선해보았다.

가장 먼저 개선한 것은 제목을 바로 쓰는 것이 아니라 형태소 분석기를 이용하여 정제를 해보았다. 형태소 분석기를 이용하면 동의어 처리나 불필요한 단어들이 없어져서 정확도가 높아질 것으로 예상했다. 형태소 분석기를 이용하여 전처리한 훈련 예는 예상대로 약 65%의 정확도를 보였다.

그 다음으로 개선한 것은 학습 모델을 변경하였다. sklearn에서는 여러 개의 클래스 중 최적의 클래스를 찾을 수 있는 학습 모델(multiclass classification)을 지원한다. 이 학습 모델들을 하나씩 적용해서 학습 결과를 비교하였을 때, multi

가장 큰 차이점은 카테고리 수였다. [working with text data]에서는 20개의 주제로 분류하지만 내가 만든 기능에서는 약 200개의 카테고리를 분류해야 했다. 

1. 모든 카테고리 정보 가져오기
1. 카테고리별 상품 제목 가져오기
1. 형태소 분석
1. CountVectorizer, TfidfTransformer, LogisticRegression
1. API serving

## 차이점

상품 제목을 이용하여 상품 카테고리를 예측하는 작업은 [working with text data]의 `Twenty Newsgroups`에서 문서 주제를 예측하는 것과 유사한 작업이지만 약간의 차이점이 있었다.

우선 예측해야 할 카테고리의 수가 달랐다. `Twenty Newsgroups`의 경우 20개의 토픽을 예측하는 것이라면 내가 개발해야 할 상품의 카테고리는 200개가 조금 넘었다. 예측해야 할 카테고리의 숫자가 많아지면 예측 모델의 정확도가 떨어질 수 밖에 없다는 것을 감안해야 한다.

그리고 사용자가 등록한 상품 정보를 훈련 예로 이용해야 하기 때문에 부정확한 훈련 예가 존재할 수 있었다. 이는 교사학습(supervisor learning)의 중요한 문제로써 학습에 사용되는 훈련 예가 정확해야 학습 결과의 정확도가 높으므로, 훈련 예의 정확성을 최대한 보장해줄 수 있어야 한다.

1. 모든 카테고리 정보 가져오기
1. 카테고리별 상품 제목 가져오기
1. 형태소 분석
1. CountVectorizer, TfidfTransformer, LogisticRegression
1. API serving

## linear regression

모델 크기가 너무 큼
학습 시간 오래 걸림
학습을 위해 기존 데이터를 보관하고 있어야 함

평가 

## fasttext

점진적으로 학습 가능

유의점

성능이 좋은 형태소 분석기
훈련 예가 완벽해야 성능도 올라간다


## 마치며

[working with text data]: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
[count vectorizer]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
