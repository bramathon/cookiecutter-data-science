# Install pipenv
FROM python:3.8 as packages
RUN pip install --upgrade pip \
 && pip install pipenv==2020.11.15

# Copy the Pipfile and Pipfile.lock
ARG PACKAGES_DIR=/opt/python/.venv
WORKDIR ${PACKAGES_DIR}/..
COPY Pipfile Pipfile.lock ./
RUN PIPENV_VENV_IN_PROJECT=1 \
    pipenv install \
    --deploy

# Base Image
{%- if cookiecutter.gpu == "y" -%}
FROM tensorflow/tensorflow:2.4.1-gpu

{%- else -%}
FROM tensorflow/tensorflow:2.4.1

{% endif %}

# Install some utilities
# Link python3.8 to 'python' and 'python3'
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash-completion=1:2.8-1ubuntu1 \
        graphviz=2.40.1-2 \
        emacs \
        curl=7.58.0-2ubuntu3.12 \
        git=1:2.17.1-1ubuntu0.7 \
        jq=1.5+dfsg-2 \
        less=487-0.1 \
        openssh-client=1:7.6p1-4ubuntu0.3 \
        python3.8=3.8.0-3~18.04 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf $(which python3.8) $(which python || echo '/usr/local/bin/python') \
 && ln -sf $(which python3.8) $(which python3 || echo '/usr/local/bin/python3') \
    && python -m pip install --no-cache-dir --upgrade pip

# nodejs
RUN curl -sL https://deb.nodesource.com/setup_15.x | bash - \
 && apt-get install -y nodejs

# aws cli
COPY --from=amazon/aws-cli:2.1.9 /usr/local/aws-cli/v2/current /usr/local

# Set up default user 'scientist'
ARG USER=scientist
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} ${USER} \
 && adduser \
    --disabled-password \
    --gecos ${USER} \
    --gid ${GID} \
    --uid ${UID} \
    ${USER}

# Default directory is /home/scientist/{{ cookiecutter.repo_name }}
ARG APP_DIR=/home/${USER}/{{ cookiecutter.repo_name }}
RUN mkdir -p ${APP_DIR} \
 && chown ${USER}:${USER} ${APP_DIR}

WORKDIR ${APP_DIR}

# install the Pipfile
COPY --chown=${USER}:${USER} Pipfile Pipfile.lock ./
RUN PIP_NO_CACHE_DIR=1 \
    PIP_UPGRADE=1 \
    pipenv install \
    --deploy \
    --dev \
    --system

# jupyterlab extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install jupyterlab-plotly
RUN jupyter labextension install plotlywidget
RUN jupyter labextension install @aquirdturtle/collapsible_headings
RUN jupyter labextension install @jupyterlab/toc

# install the src package
COPY --chown=${USER}:${USER} . .
RUN pip install --no-cache-dir --editable .

USER ${USER}
ENV SHELL /bin/bash
