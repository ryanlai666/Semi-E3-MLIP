# Metal and oxide visualization gallery

Static force animations show held-out configurations and camera motion, not continuous AIMD. Each pair shares geometry and force-arrow scale. Model-only MD panels use identical initial positions/velocities at two timesteps. Numerical stability is separate from DFT accuracy.

[Complete quantitative results](results.md) | [Real external AIMD comparisons](results.md#external-aimd-comparison)

## Silicon

Selected-example mean force MAE: 0.242 eV/A.

![DFT and model force comparison for silicon](assets/forces_silicon.gif)

Model-only NVE timestep comparison (100 fs; not AIMD):

![Model-only MD timestep comparison for silicon](assets/md_silicon.gif)

## Copper

Selected-example mean force MAE: 0.052 eV/A.

![DFT and model force comparison for copper](assets/forces_copper.gif)

Model-only NVE timestep comparison (100 fs; not AIMD):

![Model-only MD timestep comparison for copper](assets/md_copper.gif)

## Silica

Selected-example mean force MAE: 0.047 eV/A.

![DFT and model force comparison for silica](assets/forces_silica.gif)

Model-only NVE timestep comparison (100 fs; not AIMD):

![Model-only MD timestep comparison for silica](assets/md_silica.gif)

## Alumina

Selected-example mean force MAE: 0.319 eV/A.

![DFT and model force comparison for alumina](assets/forces_alumina.gif)

Model-only NVE timestep comparison (100 fs; not AIMD):

![Model-only MD timestep comparison for alumina](assets/md_alumina.gif)

## Hafnia

Selected-example mean force MAE: 0.443 eV/A.

![DFT and model force comparison for hafnia](assets/forces_hafnia.gif)

Model-only NVE timestep comparison (100 fs; not AIMD):

![Model-only MD timestep comparison for hafnia](assets/md_hafnia.gif)

## Titania

Selected-example mean force MAE: 0.251 eV/A.

![DFT and model force comparison for titania](assets/forces_titania.gif)

Model-only NVE timestep comparison (100 fs; not AIMD):

![Model-only MD timestep comparison for titania](assets/md_titania.gif)

The smaller timestep reduced the total-energy range in five of six cases. Copper was slightly nonmonotonic: 2.733e-6 versus 2.992e-6 eV/atom at 0.5 and 0.25 fs. These short checks do not establish long-term stability.

[Exact structure IDs and plotting scales](../reports/visualizations/cases.json)
