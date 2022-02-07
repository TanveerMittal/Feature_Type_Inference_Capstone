FROM ucsdets/datahub-base-notebook:2021.3-stable

USER root

RUN conda create -n capstone python=3.7

USER jovyan
WORKDIR /home/jovyan

RUN git clone https://github.com/TanveerMittal/Feature_Type_Inference_Capstone.git
RUN cd Feature_Type_Inference_Capstone/Transformer_Modeling
RUN pip install -r requirements.txt
RUN python run.py