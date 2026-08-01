Lava / Loihi 2 simulation code (from Alwin's hybrid SNN-ANN project).

This is a REFERENCE/TEMPLATE for running a trained SNN through Intel Lava's
Loihi2SimCfg (bit-accurate fixed-point Loihi 2 simulation on CPU), then
estimating energy from the simulated spike activity.

Files:
  src_utils/lava_snn.py          - builds a Lava process network (Conv->LIF...)
                                    from a trained SNN, folds BN into conv,
                                    quantizes to int8, runs Loihi2SimCfg
                                    (select_tag="fixed_pt"), monitors spikes.
  src_utils/power_estimation.py  - per-operation energy constants + power report.
  src_utils/bn_folding.py        - BN-into-conv folding used before quantization.
  scripts/run_lava_simulation.sh - how it is invoked (python -m src.utils.lava_snn).
  scripts/check_lava_validation.py - validation of the Lava path vs PyTorch.

NOTE: lava_snn.py is wired to my BLADE model/data modules (models/, nas/,
data/), so it is not standalone-runnable as-is. Use it as the template for how
to build and run the Loihi2SimCfg simulation, and adapt the network-building
part for the SPARQ model.
