# KLUE Baseline

KLUE Baseline 내 DP 모델을 활용하여 데이터를 인입하여 추론하는 모듈입니다.

기존 klue 데이터 양식에 맞춰 [데이터](data/03_tok.tsv)를 준비하면(더미태깅 포함) [스크립트](main_tom.ipynb)를 통해 모델이 추론한 [결과](data/03_DP.tsv)로 반환합니다.




## Dependencies

Make sure you have installed the packages listed in requirements.txt.

```
pip install -r requirements.txt
```

All expereiments are tested under Python 3.7 environment.




## Reference

If you use this code or KLUE, please cite:

```
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contribution

Feel free to leave issues if there are any questions or comments. To contribute, please run ``make style`` before creating pull requests.
