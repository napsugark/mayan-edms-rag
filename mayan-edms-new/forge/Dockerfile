# syntax=docker/dockerfile:1.2
# vim:set ft=dockerfile:

FROM ubuntu:noble-20250910

COPY config.env /config.env

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Debian package caching.
ARG APT_PROXY

RUN --mount=type=cache,target=/var/cache/apt \
set -x \
&& if [ "${APT_PROXY}" ]; \
    then echo "Acquire::http { Proxy \"http://${APT_PROXY}\"; };" > /etc/apt/apt.conf.d/01proxy \
; fi \
&& apt-get update \
&& apt-get install --no-install-recommends --yes \
    ca-certificates clamav file fuse ghostscript git gnupg1 graphviz libarchive-zip-perl libfile-mimeinfo-perl libimage-exiftool-perl libmariadb3 libpq5 libreoffice-calc-nogui libreoffice-draw-nogui libreoffice-impress-nogui libreoffice-math-nogui libreoffice-writer-nogui poppler-utils sane-utils sudo supervisor tesseract-ocr \
    g++ gcc libffi-dev libjpeg-dev libldap2-dev libpng-dev libsasl2-dev libssl-dev libtiff-dev make pkg-config zlib1g-dev \
    enchant-2 graphviz libcairo2 \
    bash-completion curl gettext less software-properties-common squid-deb-proxy-client vim virtualenvwrapper \
    default-libmysqlclient-dev \
    libpq-dev \
    openssh-client rsync \
    make python3-dev python3-venv \
    tesseract-ocr-deu

RUN --mount=type=cache,target=/var/cache/apt \
set -x \
&& if [ "${APT_PROXY}" ]; \
    then echo "Acquire::http { Proxy \"http://${APT_PROXY}\"; };" > /etc/apt/apt.conf.d/01proxy \
; fi \
&& add-apt-repository --yes ppa:deadsnakes/ppa \
&& apt-get install --yes python3.11 python3.11-dev python3.10 python3.10-dev

RUN echo "ubuntu ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

COPY forge/rootfs /

RUN echo "export WORKON_HOME=\$HOME/.virtualenvs" >> /home/ubuntu/.bashrc \
&& echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> /home/ubuntu/.bashrc

RUN echo "PS1_ORIGINAL=\"\$PS1\"" >> /home/ubuntu/.bashrc \
&& echo "PS1=\"🛠️  \$PS1_ORIGINAL\"" >> /home/ubuntu/.bashrc

# Transifex
RUN curl --location "https://github.com/transifex/cli/releases/download/v1.6.17/tx-linux-amd64.tar.gz" | tar --extract --gunzip tx \
&& mv tx /usr/local/bin/txcli \
&& echo "alias tx='/usr/local/bin/txcli --root-config /.transifexrc'" >> /home/ubuntu/.bashrc

# Virtualenv
SHELL ["/bin/bash", "-lc"]

RUN set -a \
&& sed -E "s/=(.*)/\=\"\1\"/g" /config.env > /tmp/config.env \
&& . /tmp/config.env \
&& rm /tmp/config.env \
&& set +a \
&& export WORKON_HOME=/home/ubuntu/.virtualenvs \
&& source /usr/share/virtualenvwrapper/virtualenvwrapper.sh \
&& mkdir /workspace \
&& mkvirtualenv "mayan-edms-4.10" -a /workspace -p /usr/bin/python3.13 \
&& workon "mayan-edms-4.10" \
&& /venv_dependencies.py \
&& deactivate \
&& rmdir /workspace \
&& chown --recursive ubuntu:ubuntu /home/ubuntu

WORKDIR /workspace

CMD ["/bin/bash"]

RUN apt-get clean autoclean \
&& apt-get autoremove --purge --yes \
&& rm --force /var/cache/apt/archives/*.deb \
# Remove temporary files owned by root from the platform_template step.
&& rm --force /tmp/* \
# Keep displaying log messages to stdout
&& find /var/log -type f | while read f; do echo -ne '' > $f; done \
# Delete Debian package proxy used for the base image.
&& rm --force /etc/apt/apt.conf.d/01proxy
