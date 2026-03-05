# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.02-py3

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# 1. Handle User Creation FIRST
RUN apt-get update && apt-get install -y sudo \
    && if getent passwd $USER_UID; then \
        OLD_USER=$(getent passwd $USER_UID | cut -d: -f1); \
        usermod -l $USERNAME $OLD_USER; \
        usermod -d /home/$USERNAME -m $USERNAME; \
        groupmod -n $USERNAME $(getent group $USER_GID | cut -d: -f1) || true; \
    else \
        groupadd --gid $USER_GID $USERNAME || true; \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 2. Prepare the Workspace with correct permissions
RUN mkdir -p /workspaces/CogniVault && chown -R $USERNAME:$USER_GID /workspaces
WORKDIR /workspaces/CogniVault

# 3. Install dependencies as the USER
# This ensures pip cache and site-packages are accessible
COPY --chown=$USERNAME:$USER_GID requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

USER $USERNAME
CMD ["/bin/bash"]