#!/bin/bash

cd /root/

apt-get update
apt-get install zsh
zsh --version
which zsh

chsh -s $(which zsh)
zsh

#!/bin/zsh

apt-get update

apt-get install curl

apt install git
git --version 

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

apt-get install nano

touch /root/.zshrc

if grep -q '^ZSH_THEME=' /root/.zshrc; then
    sed -i 's/^ZSH_THEME=.*$/ZSH_THEME="agnoster"/' /root/.zshrc
else
    echo 'ZSH_THEME="agnoster"' >> /root/.zshrc
fi

if grep -q 'ZSH_THEME_GIT_PROMPT_PREFIX' /root/.zshrc; then
    sed -i 's/^ZSH_THEME_GIT_PROMPT_PREFIX=.*$/ZSH_THEME_GIT_PROMPT_PREFIX="%F{cyan}("/' /root/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_SUFFIX=.*$/ZSH_THEME_GIT_PROMPT_SUFFIX=")%f"/' /root/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_DIRTY=.*$/ZSH_THEME_GIT_PROMPT_DIRTY="%F{red}*%f"/' /root/.zshrc
    sed -i 's/^ZSH_THEME_GIT_PROMPT_CLEAN=.*$/ZSH_THEME_GIT_PROMPT_CLEAN=""/' /root/.zshrc
else
    echo 'ZSH_THEME_GIT_PROMPT_PREFIX="%F{cyan}("' >> /root/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_SUFFIX=")%f"' >> /root/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_DIRTY="%F{red}*%f"' >> /root/.zshrc
    echo 'ZSH_THEME_GIT_PROMPT_CLEAN=""' >> /root/.zshrc
fi

if grep -q '^PROMPT=' /root/.zshrc; then
    sed -i 's/^PROMPT=.*$/PROMPT="%F{yellow}%n@%m%f %F{blue}%~%f $(git_prompt_info)\\n%F{green}→ %f"/' /root/.zshrc
else
    echo 'PROMPT="%F{yellow}%n@%m%f %F{blue}%~%f $(git_prompt_info)\n%F{green}→ %f"' >> /root/.zshrc
fi

source /root/.zshrc

if grep -q 'export PATH="/opt/conda/bin:$PATH"' /root/.zshrc; then
    echo "PATH 설정이 이미 존재합니다."
else
    echo 'export PATH="/opt/conda/bin:$PATH"' >> /root/.zshrc
fi

conda --version

/opt/conda/bin/conda init zsh

source /root/.zshrc
