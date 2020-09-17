Digital Patient
======================

This repository contains the implementation of
a "digital twin" model of patients,
i.e. a general framework that composes advanced
AI approaches and integrates mathematical modelling
in order to provide a panoramic view over current and
future physiological conditions of patients.

**Objective**: Modern medicine needs to shift from a wait and react
, curative discipline to a preventative, interdisciplinary science
aiming at providing personalised, systemic and precise treatment
plans to patients. The aim of this work is to present how the
integration of machine learning approaches with mechanistic
computational modelling could yield a reliable infrastructure
to run probabilistic simulations where the entire organism is
considered as a whole.

**Methods**: We propose a general framework that composes advanced
AI approaches and integrates mathematical modelling in order to
provide a panoramic view over current and future physiological
conditions. The proposed architecture is based on a graph neural
network (GNNs) forecasting clinically relevant endpoints (such as
blood pressure) and a generative adversarial network (GANs)
providing a proof of concept of transcriptomic integrability.

**Results**: We show the results of the investigation of pathological
effects of overexpression of ACE2 across different signalling
pathways in multiple tissues on cardiovascular functions. We
provide a proof of concept of integrating a large set of
composable clinical models using molecular data to drive local
and global clinical parameters and derive future trajectories
representing the evolution of the physiological state of the patient.

**Significance**: We argue that the graph representation of a
computational patient has potential to solve important
technological challenges in integrating multiscale computational
modelling with AI. We believe that this work represents a step
forward towards a healthcare digital twin.


Publications
---------------
If you find this repository useful in your research,
please consider citing the following papers.

Digital patient::

    @article{barbiero2020graph,
      title={Graph representation forecasting of patient's medical conditions: towards a digital twin},
      author={Barbiero, Ramon Vi\~nas Torn\'e, Pietro Li\'o},
      journal={arXiv preprint},
      year={2020}
    }


Computational patient::

    @article{barbiero2020computational,
      title={The Computational Patient has Diabetes and a COVID},
      author={Barbiero, Pietro and Li{\'o}, Pietro},
      journal={arXiv preprint arXiv:2006.06435},
      year={2020}
    }


Architecture
---------------

.. figure:: https://github.com/pietrobarbiero/digital-patient/blob/master/img/architecture.png
    :height: 600px


Simultions
--------------

The model can be used to actively monitor and forecast
clinical endpoints predicting the evolution of
patient's conditions.


.. figure:: https://github.com/pietrobarbiero/digital-patient/blob/master/img/results.png
    :height: 400px



Authors
-------

Pietro Barbiero, Ramon Viñas Torné and Pietro Liò.

Licence
-------

Copyright 2020 Pietro Barbiero, Ramon Viñas Torné and Pietro Liò.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
