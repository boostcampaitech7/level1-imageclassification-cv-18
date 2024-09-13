# AI stages linux server init setting (VS-code)
[***chan-note : AI Stages 서버 세팅***](https://watery-monkey-d20.notion.site/AI-Stages-fc11af229b504cffbde024a394500b48)


---
***(주의) 올바른 경로에서 작업하는게 매우 중요합니다!***

***(주의) 서버가 설정 되어 있다면, step 3의 git setting 만 진행하면 됩니다.***


***(참고) 부분적인 setting 이 필요하다면 위 주소를 참고해 직접 코드를 입력해주세요.***

***(참고) ssh 첫 연결을 개인페이지로 열었을 때 zsh 설정이 초기화 되는 경우가 있는데, 터미널에 *zsh* 를 입력해주면 됩니다.***



## step 1 

- **bash_setting.sh**, **zsh_setting_1.sh**, **zsh_setting_2.sh** 를 다운 받아 준비해주세요.
- ssh 연결 후 */data/ephemeral/home* 파일을 열어주세요.
-  */data/ephemeral/home* 폴더에 다운로드한 .sh 파일을 저장해주세요.

- bash shell 에서 시작합니다.
    ```bash
    # bash shell

    # ~ 경로는 vs-code에서 open 한 경로입니다. (자동으로 설정되어 있습니다.)
    cd ~
    pwd

    # /data/ephemeral/home 경로면 진행    
    chmod +x bash_setting.sh
    ./bash_setting.sh
    ```

## step 2

- *data/ephemeral/home* 으로 ssh를 열었는 지 한번 더 확인해주세요.
    ```bash
    # zsh shell 전환
    zsh

    cd ~
    pwd

     # /data/ephemeral/home 경로면 진행    
    chmod +x zsh_setting_1.sh
    ./zsh_setting_1.sh

    chmod +x zsh_setting_2.sh
    ./zsh_setting_2.sh

    source ~/.zshrc
    ```

## step 3 

- 개인 폴더를 만들고 github와 연동합니다.

    현재위치를 꼭 확인해주세요. (개인 폴더인지 확인)

    ```bash
    cd /data/ephemeral/home
    pwd

    # /data/ephemeral/home 경로면 진행
    mkdir [user]
    cd [user]

    git clone [레포지토리 주소]
    ```
    아이디, 비밀번호를 요구하면 다음을 참고해주세요. [***git token 사용***](https://watery-monkey-d20.notion.site/AI-Stages-fc11af229b504cffbde024a394500b48)  




- vs-code 로 생성한 레포지토리 폴더를 열어줍니다.
    


- commit 설정을 해줍니다.
    ```bash
    
    git config user.name "your name"
    git config user.email "your email(github email)"
    ```

- 가상환경 설정
    
    만약 가상환경이 없다면 만들어 줍니다.
    ```bash
    conda create -n [] --clone base
    conda activate []
    ```

    ***(주의)라이브러리를 설치할 때 가상환경을 항상 확인해주세요.***
    









