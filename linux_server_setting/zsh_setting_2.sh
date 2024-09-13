#!/bin/zsh

# 패키지 설치 (apt-get 수정 완료)
apt-get install nano -y

# ~/.zshrc 파일이 없으면 생성
touch ~/.zshrc

# ZSH 테마 설정
if grep -q '^ZSH_THEME=' ~/.zshrc; then
    sed -i 's/^ZSH_THEME=.*$/ZSH_THEME="agnoster"/' ~/.zshrc
else
    echo 'ZSH_THEME="agnoster"' >> ~/.zshrc
fi

# Git 브랜치 강조 설정
if grep -q 'ZSH_THEME_GIT_PROMPT_PREFIX' ~/.zshrc; then
    sed -i 's/^ZSH_THEME_GIT_PROMPT_PREFIX=.*$/ZSH_THEME_GIT_PROMPT_PREFIX="%F{cyan}("/' ~/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_SUFFIX=.*$/ZSH_THEME_GIT_PROMPT_SUFFIX=")%f"/' ~/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_DIRTY=.*$/ZSH_THEME_GIT_PROMPT_DIRTY="%F{red}*%f"/' ~/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_CLEAN=.*$/ZSH_THEME_GIT_PROMPT_CLEAN=""/' ~/.zshrc
else
    echo 'ZSH_THEME_GIT_PROMPT_PREFIX="%F{cyan}("' >> ~/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_SUFFIX=")%f"' >> ~/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_DIRTY="%F{red}*%f"' >> ~/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_CLEAN=""' >> ~/.zshrc
fi

#프롬프트 설정
if grep -q '^PROMPT=' ~/.zshrc; then
    sed -i 's/^PROMPT=.*$/PROMPT="%F{yellow}%n@%m%f %F{blue}%~%f $(git_prompt_info)\n%F{green}→ %f"/' ~/.zshrc
else
    echo 'PROMPT="%F{yellow}%n@%m%f %F{blue}%~%f $(git_prompt_info)\n%F{green}→ %f"' >> ~/.zshrc
fi

# PATH 설정
if grep -q 'export PATH="/opt/conda/bin:$PATH"' ~/.zshrc; then
    echo "PATH 설정이 이미 존재합니다."
else
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.zshrc
fi

# Conda 활성화
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    echo "Conda를 찾을 수 없습니다."
fi

# Conda 초기화 및 버전 확인
conda --version
/opt/conda/bin/conda init zsh

# 설정 적용
source ~/.zshrc