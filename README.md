# learning-to-identify-electrons-clone

> "Replication — The confirmation of results and conclusions from one study obtained independently in another — is considered the scientific gold standard." - JASNY, Barbara R., et al. Again, and again, and again…. Science, 2011, 334.6060: 1225-1225.

## Python 설정

* [venv](https://docs.python.org/3.10/library/venv.html)

```
$ python3 -m venv venv
$ .\venv\Scripts\activate  # windows
$ source venv/bin/activate # macOS, Linux
$ (venv) python3
Python 3.10.9 (tags/v3.10.9:1dd9be6, Dec  6 2022, 20:01:21) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

## M1/M2 기반 macOS에서 Python 설치
- [ ] TODO. 비디오 강의 작성

- 기존에 사용하던 Python은 삭제
```
$ brew uninstall python@3.x
```
- `xcode` 개발자 도구 설치 
```
$ xcode-select --install
```
- `miniconda` [다운로드](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh) 후 설치(설치 후 터미널 재실행)
```
$ bash ./Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda
```
- 가상환경 만들기
```
$ conda create -n tf python=3.9
$ conda activate tf
```
- 텐서플로우 설치
```
$ (tf) conda install -c apple tensorflow-deps
$ (tf) python -m pip install tensorflow-macos
$ (tf) python -m pip install tensorflow-metal
```
- 설치 확인
```
$ python
>>> import tensorflow
>>> tensorflow.__version__
```

## 필수 프로그램 설치

```
$ (venv) python3 -m pip install -U pip setuptools wheel
$ (venv) pip install jupyterlab
$ (venv) pip install energyflow
$ (venv) pip install pandas
$ (venv) pip install scikit-learn
$ (venv) pip install seaborn
$ (venv) pip install tensorflow
$ (venv) pip freeze > requirements.txt
```

## 라이브러리 재현 가능성 테스트

```
$ python3 -m venv tutorials
$ .\tutorials\Scripts\activate
$ (tutorials) pip install -r requirements.txt
$ (tutorials) jupyter-lab
```

## data

* http://mlphysics.ics.uci.edu/data/2020_electron/에서 다운로드
* 다운로드 받은 파일 3개(`data.h5`, `efps.h5`, `unscaled_data.h5`)를 data 폴더에 복사 또는 이동
