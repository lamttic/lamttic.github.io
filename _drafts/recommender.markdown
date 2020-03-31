---
layout: single
excerpt: ""
title: "추천 시스템을 만들어보자."
date: 2020-03-16 23:07:00 +0900
tags: recommender
---

사용자 클러스터링

인기
성별 & 연령
카테고리
행동기반

모델 기반의 학습에서 실시간 추천 학습으로

feature matrix를 주기적으로 저장
검색 엔진에 저장(ranking을 매길 수 있게)
query cluster에서 검색엔진에 요청(feature를 쿼리로 삼아 랭킹된 결과를 near realtime으로 가져오도록)
grpc 통신

부분적으로 아이템 정보를 확실하게 가져올 수 있도록 in memory cache 를 둠

