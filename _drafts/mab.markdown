---
layout: single
excerpt: ""
title: "정보 검색(Information Retrieval) 평가는 어떻게 하는 것이 좋을까?(1/2)"
date: 2020-03-16 23:07:00 +0900
tags: information-retrieval evaluation accuracy precision recall
---

오늘 날 우리는 엄청나게 많은 컨텐츠 속에 살고 있다. 아마 사용자가 이 많은 컨텐츠를 일일히 찾아보고 자신이 원하는 정보를 찾는다는 것은 불가능에 가까울 것이다. 컨텐츠 기반의 모든 서비스는 사용자의 정보를 활용하여 수많은 컨텐츠 중 사용자가 원하는 양질의 정보를 사용자에게 제공하기 위해 노력한다.(수 만개의 뉴스 중 사용자의 입맛에 맞는 뉴스만 선별하여 제공하고, 업로드된 이미지로 사물을 인식하여 사용자에게 관련 정보를 알려주는 것처럼 말이다.)

이러한 서비스들은 인기 컨텐츠, 키워드 검색과 같이 다양한 형태로 서비스를 제공하며 이렇게 원하는 내용과 관련있는 결과를 얻어내는 것을 정보 검색(Information Retrieval, 이하 IR)이라고 한다. IR의 효율성을 분석하기 위해 여러가지 기준을 가지고 평가를 필요가 있고, 이 기준을 평가모델이라 부른다.

이 글에서는 평가모델로 사용할 수 있는 다양한 기준들에 대해 알아보고자 한다.

## Accuracy, Precision, Recall

[precision and recall] 위키피디아 글을 보면, precision(정밀도)와 recall(재현율)에 대하여 설명하고 있다.

precision과 recall에 이해하기 전에 먼저 몇 가지 개념에 대해 이해를 해보도록 하자.

1. true positive
1. true negetive
1. false positive
1. false negative






Accuracy, Precision, Recall
MRR, MAP, NDCG

[precision and recall]: https://en.wikipedia.org/wiki/Precision_and_recall
