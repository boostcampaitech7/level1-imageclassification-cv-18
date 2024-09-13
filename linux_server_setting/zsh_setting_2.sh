#!/bin/zsh

apt-get install nano -y

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

if grep -q '^ZSH_THEME=' ~/.zshrc; then
    sed -i 's/^ZSH_THEME=.*$/ZSH_THEME="agnoster"/' ~/.zshrc
else
    echo 'ZSH_THEME="agnoster"' >> ~/.zshrc
fi

echo '' >> ~/.zshrc 

sed -i 's/^plugins=(.*)/plugins=(git zsh-syntax-highlighting)/' ~/.zshrc

echo '' >> ~/.zshrc 

# Git 브랜치 강조 설정
if grep -q 'ZSH_THEME_GIT_PROMPT_PREFIX' ~/.zshrc; then
    sed -i 's/^ZSH_THEME_GIT_PROMPT_PREFIX=.*$/ZSH_THEME_GIT_PROMPT_PREFIX="%F{magenta}["/' ~/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_SUFFIX=.*$/ZSH_THEME_GIT_PROMPT_SUFFIX="]%f"/' ~/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_DIRTY=.*$/ZSH_THEME_GIT_PROMPT_DIRTY="%F{red}*%f"/' ~/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_CLEAN=.*$/ZSH_THEME_GIT_PROMPT_CLEAN=""/' ~/.zshrc
else
    echo 'ZSH_THEME_GIT_PROMPT_PREFIX="%F{magenta}["' >> ~/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_SUFFIX="]%f"' >> ~/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_DIRTY="%F{red}*%f"' >> ~/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_CLEAN=""' >> ~/.zshrc
fi

echo '' >> ~/.zshrc 

sed -i '/^PROMPT=/,+1d' ~/.zshrc

echo "PROMPT='%F{yellow}> %F{blue}%~%f \$(git_prompt_info)" >> ~/.zshrc
echo "%F{green}> %f'" >> ~/.zshrc

echo '' >> ~/.zshrc 

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    echo "Conda를 찾을 수 없습니다."
fi

conda --version
/opt/conda/bin/conda init zsh

source ~/.zshrc