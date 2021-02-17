# Base Image
{% if cookiecutter.gpu == "Use GPU" %}
FROM tensorflow/tensorflow:2.4.1-gpu
{% else %}
FROM tensorflow/tensorflow:2.4.1
{% endif %}

# Install basic packages
# Link python3.8 to 'python' and 'python3'
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
        gnuplot \
        jq \
        less \
        openssh-client \
        python3.8=3.8.0-3~18.04 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf $(which python3.8) $(which python || echo '/usr/local/bin/python') \
 && ln -sf $(which python3.8) $(which python3 || echo '/usr/local/bin/python3') \
    && python -m pip install --no-cache-dir --upgrade pip

# Set up default user '{{ cookiecutter.author_name }}'
ARG USER={{ cookiecutter.author_name }}
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} ${USER} \
 && adduser \
    --disabled-password \
    --gecos ${USER} \
    --gid ${GID} \
    --uid ${UID} \
    ${USER}

# Default directory is /home/scientist/data-science-project
ARG APP_DIR=/home/${USER}/data-science-project
RUN mkdir -p ${APP_DIR} \
    && chown ${USER}:${USER} ${APP_DIR}

# Install dev packages...
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - \
  && apt-get install -y --no-install-recommends \
        bash-completion \
        groff \
        nano \
        emacs \
        nodejs \
        python3.8-dev=3.8.0-3~18.04 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pipenv==2020.11.15

# Install the pipenv
COPY Pipfile Pipfile.lock setup.py ./
RUN PIP_NO_CACHE_DIR=1 \
    PIP_UPGRADE=1 \
    pipenv install \
    --deploy \
    --dev \
    --system

# Jupyterlab build
RUN jupyter labextension install --no-build \
        jupyterlab-plotly@4.14.3 \
        plotlywidget@4.14.3 \
        @aquirdturtle/collapsible_headings \
 && jupyter lab build

# Install the src package
COPY --chown=${USER}:${USER} . .
RUN pip install --no-cache-dir --editable .

# AWS cli
COPY --from=amazon/aws-cli:2.1.9 /usr/local/aws-cli/v2/current /usr/local

USER ${USER}
ENV SHELL /bin/bash
