# Vertical Fish Analysis (Python ver.)

Code related to Zhu Y, Auer F, Ehrlich DE, Schoppik D., 2022.

This code is for analysis and visualization of data generated using free-swimming vertical fish apparatus.

## Requirements

Build with Python3.8. See `environment.yml` for required packages.

## Usage

### Contents

`src` folder contains all the scripts for data analysis and visualization.

`docs` contains a copy of catalog files generated after running `.../src/vf_analysis/vf_analysis.py` and expected number of swim bouts captured over 24 hrs per box in constant dark condition.

`sample figures` contains plots that can be generated using scripts under `.../src/vf_visualization/`

### Analyze raw data files

To analyze data generated using the free-swimming apparatus:

1. Run `vf_analysis.py` under `.../src/vf_analysis/`.
2. Follow the instruction and input the root path that contains data files (.dlm) and corresponding metadata files (.ini). Data can be directly under the root directory or under subfolders within the root directoty. See notes for details.
3. Follow the instruction and input the frame rate (in integer). See notes for details.
4. The program will go through every data file in each subfolder (if there is any) and extract swim attributes.

When done, there will be three hdf5 files (.h5) under each directory that contains data file(s) together with catalog files that explains the parameters extracted. A copy of catalog files can be found under `docs`.

All the extracted swim bouts under `bout_data.h5` are aligned at the time of the peak speed. Each aligned bout contains swim parameters from 500 ms before to 300 ms after the time of the peak speed.

**Notes** on data analysis

- All the .dlm data files under the same directory will be combined for bout extraction. To analyze data separately, please move data files (.dlm) and corresponding metadata files (.ini) into subfolders under the root path.
- Analysis program will stop if it fails to detect any swim bout in a data file (.dlm). To avoid this, please make sure all data files to be analyzed are reasonably large so that it contains at least one swim bout. Generally, we found > 10 MB being a good criteria.
- Please input the correct frame rate as this affects calculation of parameters. This program only accepts one frame rate number for each run. Therefore, all data files under the root path need to be acquired under the same frame rate.

### Make figures

To generate figures:

1. Run individual scripts under `.../src/vf_visualization/`.
2. Alternatively, one may run `plot all.py` to plot all figures.
3. Figures will be saved under `.../figures`.

**Visualization scripts and function** explained

- `plot_timeseries.py` plots basic parameters as a function of time. Modify "all_features" to select parameters to plot. This script contains two functions: `plot_aligned()`, `plot_raw()`.

- `plot_parameters.py` plots swim parameters and their distribution. This script contains function: `plot_parameters`

- `plot_IBIposture.py` plots Inter Bout Interval (IBI or IEI) posture distribution and standard deviation. This script contains function: `plot_IBIposture()`. This script looks for "prop_Bout_IEI2" in the "prop_bout_IEI_pitch" data which includes mean of body angles during IEI.

- `plot_bout_timing.py` plots bout frequency as a function of pitch angle and fiitted coefs of function `y = a * ((x-b)^2) + c`. Jackknife resampling is applied if contains data from multiple experiments (subfolders under the root path). This script contains function: `plot_bout_frequency()`

- `plot_kinetics.py` plots bout kinetics: righting gain, set point, steering gain. Jackknife resampling is applied if contains data from multiple experiments (subfolders under the root path). This script contains function: `plot_kinetics()`
- `plot_fin_body_coordination.py` plots attack angle as a function of rotation and calculates the maximal slope which is termed the fin-body ratio. This script contains function: `plot_atk_ang_posture_chg()`

### Parameters

| Parameters                | Unit | Definition                                                                                |
| ------------------------- | ---- | ----------------------------------------------------------------------------------------- |
| Pitch angle               | deg  | Angle of the fish on the pitch axis relative to horizonal                                 |
| Peak speed                | mm/s | Peak speed of swim bouts                                                                  |
| Initial pitch             | deg  | Pitch angle at 250 ms before the peak speed                                               |
| Post-bout pitch           | deg  | Pitch angle at 100 ms after the peak speed                                                |
| End pitch                 | deg  | Pitch angle at 200 ms after the peak speed                                                |
| Acceleration phase        |      | Before time of the peak speed                                                             |
| Deceleration phase        |      | After time of the peak speed                                                              |
| Total rotation            | deg  | Pitch change from initial (250 ms before) to end (200 ms after) time of the peak speed    |
| Bout trajectory           | deg  | Tangential angle of the trajectory at the time of the peak speed                          |
| Bout displacement         | mm   | Displacement of fish from pre-bout to post-bout                                           |
| Inter-bout interval       | s    | Duration between two adjacent swim bouts                                                  |
| Inter-bout-interval pitch | deg  | Mean pitch angle during inter-bout interval                                               |
| Trajectory deviation      | deg  | Deviation of bout trajectory from initial pitch (250 ms before)                           |
| Steering rotation         | deg  | Change of pitch angle from initial (250 ms before) to the time of the peak speed          |
| Steering gain             |      | Slope of best fitted line of posture vs trajectory at the time of the peak speed          |
| Early rotation            | deg  | Change of pitch angle from initial (250 ms before) to 50 ms before time of the peak speed |
| Attack angle              | deg  | Deviation of bout trajectory from pitch at time of the peak speed                         |
| Fin-body ratio            |      | Maximal slope of best fitted sigmoid of attack angle vs early rotation                    |
| Righting rotation         | deg  | Change of pitch angle from time of the peak speed to post bout (100 ms after peak speed)  |
| Righting gain             |      | Numeric inversion of the slope of best fitted line of righting rotation vs initial pitch  |
| Set piont                 | deg  | x intersect of best fitted line of righting rotation vs initial pitch                     |

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact
