FROM ucsdets/datahub-base-notebook:2021.3-stable

USER root

RUN /opt/conda/bin/conda create -y -n capstone python=3.7 
ENV PATH /opt/conda/envs/capstone/bin:$PATH
RUN /bin/bash -c "source activate capstone"

USER jovyan
WORKDIR /home/jovyan

RUN git clone https://github.com/TanveerMittal/Feature_Type_Inference_Capstone.git
RUN cd Feature_Type_Inference_Capstone
WORKDIR Feature_Type_Inference_Capstone
COPY . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
WORKDIR Transformer_Modeling
RUN python run.py