```mermaid
---
config:
  theme: redux
  layout: dagre
---
flowchart LR
 subgraph modeling["Multiscale Modeling"]
        openmc["OpenMC"]
        pka["PKA spectra"]
        surrogate["Atomistic
        AI Surrogate"]
        cascades["Defect production"]
        mesoscale["Mesoscale models"]
  end
 subgraph validation["Validation"]
        computed["Microstructural evolution
    (Predicted)"]
        expresult["Microstructural evolution
    (Observed)"]
  end
    component["Component of interest"] --> openmc & experiment["Irradiation experiments
    MIT LMNT"]
    openmc --> pka
    pka --> surrogate
    surrogate --> cascades
    cascades --> mesoscale
    mesoscale --> computed
    experiment ----> expresult

    pka@{ shape: internal-storage}
    cascades@{ shape: internal-storage}
    computed@{ shape: terminal}
    expresult@{ shape: terminal}
    component@{ shape: manual-input}
     openmc:::Process
     pka:::Data
     surrogate:::AI
     cascades:::Data
     mesoscale:::Process
     computed:::Terminal
     expresult:::Terminal
     component:::Input
     experiment:::Process
    classDef Input fill:#c6eebf
    classDef Data stroke-width:1px, stroke-dasharray:none, stroke:#FBB35A, fill:#FFEFDB, color:#8F632D
    classDef Process stroke-width:1px, stroke-dasharray:none, stroke:#374D7C, fill:#E2EBFF, color:#374D7C
    classDef Terminal stroke-width:1px, stroke-dasharray:none, stroke:#FF5978, fill:#FFDFE5, color:#8E2236
    classDef AI stroke-width:1px, stroke-dasharray:none, fill:#f4d3fb
    style surrogate fill:#E1BEE7
  ```
