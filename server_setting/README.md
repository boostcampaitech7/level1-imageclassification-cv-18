# AI stages linux server init setting (VS-code)
[***chan-note : AI Stages 서버 세팅***](https://watery-monkey-d20.notion.site/AI-Stages-fc11af229b504cffbde024a394500b48)


---
- ***올바른 경로에서 작업하는게 매우 중요합니다.***


- 재접속 시 설정이 반영 안되어있을 땐 bash shell 에서 ***zsh*** 를 입력하여 활성화 할 수 있습니다.
    
    ```bash
    bash
    zsh
    ```  


- step 1
    - ***install_zsh.sh*** 

        -  zsh shelll 을 설치합니다.

-  step 2
    - ***zsh_setting_1.sh*** 

        - 필요한 기본 기능을 설치합니다.(curl, wget, nano, libglib2.0-0)

        - oh-my-zsh 테마를 설치합니다.

    - ***zsh_setting_2.sh*** 

        -  zsh shell을 customize 합니다.

        -  conda 설정을 적용합니다.
        
    - ***download_sketch_data.sh*** 
        -  home 디렉토리에 데이터를 다운로드합니다.

- step 3
    - github 연동
    - conda 가상환경 설정

## step 1 



- **(중요) vs-code로 ssh 연결 후 */data/ephemeral/home* 폴더를 열어줍니다.**


-  */data/ephemeral/home* 폴더에 다운로드한 .sh 파일을 저장합니다.
    ```bash
    # 폴더 구조
    /data/ephemeral/home

    home
    ┣ install_zsh.sh
    ┣ download_sketch_data.sh
    ┣ zsh_setting_1.sh
    ┗ zsh_setting_2.sh 
    ```

- bash shell 에서 시작합니다.
    ```bash
    # bash shell

    # ~ = /data/ephemeral/home
    cd ~
    pwd

    # /data/ephemeral/home 경로면 진행    
    chmod +x install_zsh.sh
    ./install_zsh.sh
    ```

## step 2

- ***(중요) *data/ephemeral/home* 으로 ssh를 열었는 지 한번 더 확인해주세요.***

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

    ```bash
    cd ~
    pwd

    # /data/ephemeral/home 경로면 진행    

    chmod +x download_sketch_data.sh
    ./download_sketch_data.sh
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
    아이디, 비밀번호를 요구하면 다음을 참고해주세요. [***git token 사용***](https://watery-monkey-d20.notion.site/github-clone-11c1326a129980a3a6c2d0e73443716e)  




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
    









