FROM ucsdets/datahub-base-notebook:2021.3-stable

USER root

RUN /opt/conda/bin/conda create -y -n capstone python=3.7 
ENV PATH /opt/conda/envs/capstone/bin:$PATH
RUN /bin/bash -c "source activate capstone"

USER jovyan
WORKDIR /home/jovyan

RUN git clone https://github.com/TanveerMittal/Feature_Type_Inference_Capstone.git
RUN cd Feature_Type_Inference_Capstone
RUN cd Transformer_Modeling
RUN pip install -r requirements.txt
RUN python run.py