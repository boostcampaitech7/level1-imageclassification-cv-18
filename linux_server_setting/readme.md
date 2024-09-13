# AI stages linux server init setting (with vs-code)
.sh 파일을 이용 안할 경우 다음 주소 참고해주세요. [**chan-note : AI Stages 서버 세팅**](https://watery-monkey-d20.notion.site/AI-Stages-fc11af229b504cffbde024a394500b48)

(참고) ssh 첫 연결을 개인페이지로 열었을 때 zsh 설정이 초기화 되는 경우가 있는데, 터미널에 *zsh* 를 입력해주면 됩니다.

**bash_setting.sh**, **zsh_setting_1.sh**, **zsh_setting_2.sh** 를 다운 받아 준비해주세요.

## step 1 

- ssh 연결 후 */data/ephemeral/home* 파일을 열어주세요.
-  */data/ephemeral/home* 폴더에 다운로드한 .sh 파일을 저장해주세요.

- bash shell 에서 시작합니다.
    ```bash
    # bash shell
    cd ~
    chmod +x bash_setting.sh

    ./bash_setting.sh
    ```

## step 2

- *data/ephemeral/home* 으로 ssh를 열었는 지 한번 더 확인해주세요.
    ```bash
    zsh

    cd ~

    chmod +x zsh_setting_1.sh
    ./zsh_setting_1.sh

    chmod +x zsh_setting_2.sh
    ./zsh_setting_2.sh

    source ~/.zshrc
    ```

## step 3

- 개인 폴더를 만들고 github와 연동합니다. ([]는 각자 custom 해주세요.)

    현재위치를 꼭 확인해주세요 (개인 폴더인지 확인)

    ```bash
    cd /data/ephemeral/home
    
    mkdir [user]
    cd [user]

    git clone [레포지토리 주소]
    ```



- vs-code 로 생성한 레포지토리 폴더를 열어줍니다.
    
    ex. /data/ephemeral/home/[user]/[레포지토리 폴더명]



- commit 설정을 해줍니다.
    ```bash
    
    git config user.name "[]"
    git config user.email "[]"
    ```

- 가상환경 설정
    
    만약 가상환경이 없다면 만들어 줍니다.
    ```bash
    conda create -n server[server 번호] --clone base
    conda activate server[server 번호]
    ```

    라이브러리를 설치할 때 가상환경을 항상 확인해주세요.
    

자세한 내용은 다음 주소를 참고해주세요.

[chan-note : AI Stages 서버 세팅](https://watery-monkey-d20.notion.site/AI-Stages-fc11af229b504cffbde024a394500b48)






