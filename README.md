# learning-to-identify-electrons-clone

> "Replication — The confirmation of results and conclusions from one study obtained independently in another — is considered the scientific gold standard." - JASNY, Barbara R., et al. Again, and again, and again…. Science, 2011, 334.6060: 1225-1225.

> [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), Tesla의 AI 총괄이었고, 지금은 [OpenAI](https://openai.com/)로 복귀한 [`Andrej Karpathy`](https://scholar.google.co.kr/citations?user=l8WuQJgAAAAJ&hl=ko&oi=ao)가 만드는 신경망 개발 코스 입니다. 역전파(`backpropagation`)의 기본부터 시작해서 최신 GPT같은 딥러닝 개발에 관련된 내용을 다루고 있으며, 언어모델을 사용해서 딥러닝을 설명하고 있습니다. **컴퓨터 비전 같은 곳으로 확장해 가더라도 대부분의 학습내용이 즉시 적용 가능할만큼 양질의 내용을 담고 있습니다.** 단점이라면 YouTube 영상이라서 학습하실 때 시간이 제법 걸리는 점이 있는데, 여가 시간을 활용하시면 좋을 듯 합니다. 여러분에게 이 과정을 소개하는 이유는 이 [사진](https://twitter.com/karpathy/status/1127792584380706816)으로 갈음하도록 하겠습니다. 좋은 성과 있으셨으면 좋겠습니다.

## Python 설정

* [venv](https://docs.python.org/3.10/library/venv.html)에 관한 내용은 공식문서를 참고하세요.

```
$ python3 -m venv venv
$ .\venv\Scripts\activate  # windows
$ source venv/bin/activate # macOS, Linux
$ (venv) python3
Python 3.10.9 (tags/v3.10.9:1dd9be6, Dec  6 2022, 20:01:21) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

### M1/M2 기반 macOS에서 Python 설치 [(YouTube 링크)](https://www.youtube.com/watch?v=iqUEutQKd04)

<details>
<summary> 세부적인 사항은 해당 항목을 클릭하세요!</summary>

> M1/M2 기반 `macOS`(>= 13.x) 진행, macOS에서 지원하는 `Terminal`이나 자신이 사용하는 터미널 프로그램을 실행하여 진행하며, `brew`는 설치되어 있다고 가정

0. Python 3.9 버전 사용시 필수 설치
```
$ brew install rust
```

1. `xcode` 개발자 도구 설치  
```
$ xcode-select --install
$ xcode-select: error: command line tools are already installed, use "Software Update" in System Settings to install updates <= 이미 설치되어 있으니 신경쓰지 마세요.
```

2. 기존에 사용하던 `Python`은 삭제하시길 권고
```
$ brew uninstall python@3.x
$ python3 
Python 3.9.6 (default, Oct 18 2022, 12:41:40) <= macOS에 기본 설치된 python
[Clang 14.0.0 (clang-1400.0.29.202)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

3. 기존에 설치된 `*conda` 삭제하시길 권고
(다운로드 형태로 설치하신 분들도 삭제하시길 권고)
```
$ conda deactivate
$ brew uninstall miniconda
or
$ brew uninstall anaconda
```

4. `zsh` 환경설정(`.zshrc`)에서 `conda` 설정을 삭제
```
$ cd
$ vi .zshrc # or code .zshrc

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/sd/miniconda/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/sd/miniconda/etc/profile.d/conda.sh" ]; then
        . "/Users/sd/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/Users/sd/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

5. `miniconda` [다운로드](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh) 후 설치
```
$ bash ./Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda

[...]
Downloading and Extracting Packages

Preparing transaction: done
Executing transaction: |
done
installation finished. <= 정상적으로 설치
```

6. conda 환경 설정
```
$ ./miniconda/bin/conda init zsh
[...]
modified      /Users/sd/.zshrc

==> For changes to take effect, close and re-open your current shell. <==
```

7. 가상환경 만들기
```
$ (base) conda create -n tf python=3.9
$ (base) conda activate tf
$ (tf) python
Python 3.9.16 (main, Jan 11 2023, 10:02:19)
[Clang 14.0.6 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

8. 텐서플로우 설치
```
$ (tf) pip install -U pip setuptools wheel
$ (tf) conda install -c apple tensorflow-deps
$ (tf) pip install tensorflow-macos==2.9.2
$ (tf) pip install tensorflow-metal==0.5
```

9. 설치 확인
```
$ python
>>> import tensorflow
>>> tensorflow.__version__
'2.9.2'
```

10. 가상 환경을 외부에 저장
좀 더 자세항 사항은 `conda env create -h`를 참고
```
(tf) conda env export > tf.yaml
(base) conda env create -f tf.yaml
```
</details>
<hr>

## 필수 프로그램 설치

```
$ (venv) python3 -m pip install -U pip setuptools wheel
$ (venv) pip install jupyterlab
$ (venv) pip install energyflow
$ (venv) pip install pandas
$ (venv) pip install scikit-learn
$ (venv) pip install seaborn
$ (venv) pip install tensorflow # M1/M2 사용자 제외
$ (venv) pip install tqdm
$ (venv) pip install pyarrow
$ (venv) pip install igraph
$ (venv) pip install natsort
$ (venv) pip install pycairo
$ (venv) pip freeze > requirements.txt
```

## 라이브러리 재현 가능성 테스트

```
$ python3 -m venv tutorials
$ .\tutorials\Scripts\activate
$ (tutorials) pip install -r requirements.txt
$ (tutorials) jupyter-lab
```

## 예제 사용법

- [이 곳](http://mlphysics.ics.uci.edu/data/2020_electron/)에서 3개 파일(`data.h5`, `efps.h5`, `unscaled_data.h5`)을 다운로드 하세요.
### H5 파일 검증에 사용된 코드 사용법

- `h5file/data` 폴더에 `data.h5` 파일을 복사하세요.
- `cd` 명령어를 사용해서 `h5file` 폴더로 이동하세요.
```
$ cd h5file
$ python 01_generate_prep_data.py       # H5 -> PKL
$ python 02_generate_efp.py             # PKL -> feather
$ python 03_identify_duplicate_efp.py   # 중복제거
$ python 04_generate_efp_graphs.py      # feather -> 이미지
```
- H5 압축 해제 및 변환 작업에 대략적으로 10시간 정도가 소요됩니다.

### Collado, Julian, et al. "Learning to identify electrons." Physical Review D 103.11 (2021): 116028. 논문 관련 코드

- `EID/data` 폴더에 `data.h5`, `unscaled_data.h5` 파일을 복사하세요.

- `cd` 명령어를 사용해서 `EID` 폴더로 이동하세요. `EID` 폴더에서 학습을 진행하세요.

```
$ cd EID
$ python train/training.py
```

- 일반적인 환경(RTX 3090)에서는 대략 10시간 정도가 필요합니다.

