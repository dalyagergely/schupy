# schupy -- A python package for modeling Schumann resonances

schupy is an open-source python package aimed at modeling and analyzing Schumann resonances (SRs), the global electromagnetic resonances of the Earth-ionosphere cavity resonator in the lowest part of the extremely low frequency band (<100 Hz).

## Usage

#### `forward_tdte` function
The `forward_tdte` function of schupy uses the analytical solution of the 2-D telegraph equation (TDTE) obtained for uniform cavity and is able to determine SRs generated by an arbitrary number of sources located in given positions and returns the theoretical power spectral density of the field components for arbitrarily located observing stations. The sources can be either pointsources or extended ones with a specified size.

The function takes the following arguments:

| Name        | Type           | Description  |  Unit | Default value |
| ------------- |:-------------:| ------------- | ------------ |  ---------- |
| `s_lat` | LIST | Geographical latitude(s) of the source(s) | deg |
| `s_lon` | LIST | Geographical longitude(s) of the source(s) | deg |
| `s_int` | LIST | Intensities of the sources | C^2 km^2 s^-1 |
| `m_lat` | FLOAT | Geographical latitude of the observing station | deg |
| `m_lon` | FLOAT | Geographical longitude of the observing station | deg |
| `f_min` | FLOAT | Minimum frequency | Hz | 5 |
| `f_max` | FLOAT | Maximum frequency | Hz | 30 |
| `f_step` | FLOAT | Resolution in frequency | Hz | 0.1 |
| `radius` | FLOAT | Radius of the extended sources | Mm | 0 |
| `n` | INT | Maximal order of Legendre-polynomials to sum |  | 500 |
| `mapshow` | BOOL | Sets whether to show a map of the sources and the station or not |  | True |
| `mapsave` | BOOL | Sets whether to save the map of the sources and the station or not |  | False |
| `mapfilename` | STR | Name of the file to save the map into |  | schupy_map.png

| Name        | Type           | Description  |  Default value | Possible values |
| ------------- |:-------------:| ------------- | ------------ |  ---------- |
| `h` | STRING | Method of calculatingcomplex ionospheric heights | mushtak | mushtak, kulak |
| `ret` | STRING | Values returned | all | all, Er, Btheta, Bphi |

By setting the `radius` value to any number grater than zero, the user can model extended sources with randomly distributed pointsources inside a circle having the given radius, whose intensities sum up to the given `s_int`.

By specifying `h` the user can choose the preferred method of calculating complex ionospheric heights. The two methods are described in *Mushtak and Williams (2002)* and in *Kulak and Mlynarczyk (2013)*.

schupy can visualize the specified sources and observing station on a world map. The station is shown as a rectangle while the sources are indicated by circles whose sizes are proportional to their intensities. Visualization relies on the `cartopy` package: https://scitools.org.uk/cartopy/docs/latest/

The function plots and returns the following quantities at the location of the given observing station:
 - `Er`: the vertical component of the electric field
 - `Btheta`: the N-S component of the magnetic field measured by E-W orianted magnetic coils 
 - `Bphi`: the E-W component of the magnetic field measured by N-S orianted magnetic coils

 
An exaple to how to run the function:
~~~~
import schupy as sp

source_latitudes = [10.0, 0.0, 0.0]
source_longitudes = [10.0, -80.0, 110.0]
source_intensities = [1e5, 8e4, 7e4]
obs_latitudes = 47.6
obs_longitudes = 16.7

sp.forward_tdte(source_latitudes, source_longitudes, source_intensities, obs_latitudes, obs_longitudes, h='mushtak', ret='Bphi', radius = 0, mapshow = False, mapsave = False)
~~~~
 


## Acknowledgement

The schupy package is developed by G. Dálya, T. Bozóki, K. Kapás, J. Takátsy, E. Prácser and G. Sátori. Please send your questions and comments to `dalyag@caesar.elte.hu`. If you use the Schupy package for your research, please cite our paper.

