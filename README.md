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

## M1/M2 기반 macOS에서 Python 설치[YouTube](https://www.youtube.com/watch?v=iqUEutQKd04)
- M1/M2 기반 `macOS`(>= 13.x) 진행, macOS에서 지원하는 `Terminal`이나 자신이 사용하는 터미널 프로그램을 실행하여 진행하며, `brew`는 설치되어 있다고 가정

- 0. Python 3.9 버전 사용시 필수 설치
```
$ brew install rust
```

- 1. `xcode` 개발자 도구 설치  
```
$ xcode-select --install
$ xcode-select: error: command line tools are already installed, use "Software Update" in System Settings to install updates <= 이미 설치되어 있으니 신경쓰지 마세요.
```

- 2. 기존에 사용하던 `Python`은 삭제하시길 권고
```
$ brew uninstall python@3.x
$ python3 
Python 3.9.6 (default, Oct 18 2022, 12:41:40) <= macOS에 기본 설치된 python
[Clang 14.0.0 (clang-1400.0.29.202)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

- 3. 기존에 설치된 `*conda` 삭제하시길 권고
(다운로드 형태로 설치하신 분들도 삭제하시길 권고)
```
$ conda deactivate
$ brew uninstall miniconda
or
$ brew uninstall anaconda
```

- 4. `zsh` 환경설정(`.zshrc`)에서 `conda` 설정을 삭제
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

- 5. `miniconda` [다운로드](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh) 후 설치
```
$ bash ./Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda

[...]
Downloading and Extracting Packages

Preparing transaction: done
Executing transaction: |
done
installation finished. <= 정상적으로 설치
```

- 7. conda 환경 설정
```
$ ./miniconda/bin/conda init zsh
[...]
modified      /Users/sd/.zshrc

==> For changes to take effect, close and re-open your current shell. <==
```

- 8. 가상환경 만들기
```
$ (base) conda create -n tf python=3.9
$ (base) conda activate tf
$ (tf) python
Python 3.9.16 (main, Jan 11 2023, 10:02:19)
[Clang 14.0.6 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

- 9. 텐서플로우 설치
```
$ (tf) pip install -U pip setuptools wheel
$ (tf) conda install -c apple tensorflow-deps
$ (tf) pip install tensorflow-macos==2.9.2
$ (tf) pip install tensorflow-metal==0.5
```

- 10. 설치 확인
```
$ python
>>> import tensorflow
>>> tensorflow.__version__
'2.9.2'
```

- 11. 가상 환경을 외부에 저장
좀 더 자세항 사항은 `conda env create -h`를 참고
```
(tf) conda env export > tf.yaml
(base) conda env create -f tf.yaml
```

## 필수 프로그램 설치

```
$ (venv) python3 -m pip install -U pip setuptools wheel
$ (venv) pip install jupyterlab
$ (venv) pip install energyflow
$ (venv) pip install pandas
$ (venv) pip install scikit-learn
$ (venv) pip install seaborn
$ (venv) pip install tensorflow # M1/M2 사용자 제외
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
