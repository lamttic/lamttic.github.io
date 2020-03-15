---
layout: single
classes: wide
title:  "IR(Information Retrieval) 평가는 어떻게 하는 것이 좋을까?(1/2)"
date:   2020-03-10 16:07:00 +0900
---

오늘 날 우리는 정보 검색 분야에 사용자에게 다양한 서비스를 제공한다.

어떤 서비스는 수 만개의 뉴스 중 사용자의 입맛에 맞는 뉴스만 선별하여 제공하기도 하고, 또 다른 서비스는 업로드된 이미지로 사물을 인식하여 어떤 사물인지 추측하여 사용자에게 결과를 알려주기도 한다.

이 다양한 서비스들은 각자의 목적에 맞게 임무를 수행하고, 수행한 임무가 얼마나 잘 동작했는지 판단을 하기 위해 평가모델이 필요하다.

이 글에서는 각 서비스들이 목적에 맞게 잘 동작하는지에 대한 평가를 하기 위해 어떤 기준들을 활용해야 하는지에 대해 살펴보도록 한다.

## Accuracy, Precision, Recall

[precision and recall] 위키피디아 글을 보면, precision(정밀도)와 recall(재현율)에 대하여 설명하고 있다.

precision과 recall에 이해하기 전에 먼저 몇 가지 개념에 대해 이해를 해보도록 하자.

정보 검색 






Accuracy, Precision, Recall
MRR, MAP, NDCG

[precision and recall]: https://en.wikipedia.org/wiki/Precision_and_recall
