#!/bin/zsh

# install zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

# change theme > agnoster
if grep -q '^ZSH_THEME=' ~/.zshrc; then
    sed -i 's/^ZSH_THEME=.*$/ZSH_THEME="agnoster"/' ~/.zshrc
else
    echo 'ZSH_THEME="agnoster"' >> ~/.zshrc
fi

echo '' >> ~/.zshrc 

# plugins > git & zsh-syntax-highlighting
sed -i 's/^plugins=(.*)/plugins=(git zsh-syntax-highlighting)/' ~/.zshrc

echo '' >> ~/.zshrc 

# set git branch prompt
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

# remove 기존 prompt 형식
sed -i '/^PROMPT=/,+1d' ~/.zshrc

# set prompt
echo "PROMPT='%F{yellow}> %F{blue}%~%f \$(git_prompt_info)" >> ~/.zshrc
echo "%F{green}> %f'" >> ~/.zshrc

echo '' >> ~/.zshrc 

# conda 기능 활성화
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    echo "Conda를 찾을 수 없습니다."
fi

# conda 초기화
conda --version
/opt/conda/bin/conda init zsh

apt-get update
source ~/.zshrc