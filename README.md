# schupy -- A python package for modeling Schuman resonances

Schupy models Schumann resonances (SRs), the electromagnetic eigenmodes of the Earth-ionosphere cavity resonator in the lowest part of the extremely low frequency band (<100 Hz). The code uses the solution of the 2-D telegraph equation obtained for uniform cavity and is able to determine the theoretical SR spectrum for arbitrary source-observer configuration. It can be applied for both modeling extraordinary large SR-transients (Q-bursts) or "background" SRs induced by incoherently superimposed strokes.

## Usage

#### `forward_tdte` function
The `forward_tdte` function of schupy models SRs caused by an arbitrary number of sources located in given positions and returns the electric and magnetic fields measured by arbitrarily located observers. The sources can be either pointsources or extended ones with a specified size.

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
| `h` | STRING | Method of calculating height | mushtak | mushtak, kulak |
| `ret` | STRING | Values returned | all | all, Er, Btheta, Bphi |

By setting the `radius` value to any number grater than zero, the user can model an extended source. Extended sources are modeled as a number of randomly distributed pointsources inside a circle having the given `radius`, whose intensities sum up to the given `s_int`.

By specifying `h` the user can choose the preferred method of height calculation. The two methods are described in *Mushtak and Williams (2002)* and in *Kulak and Mlynarczyk (2013)*.

schupy can visualize the specified sources and observing station on a world map. The station is shown as a rectangle while the sources are indicated by circles whose sizes are proportional to their intensities. Visualization relies on the `cartopy` package: https://scitools.org.uk/cartopy/docs/latest/

The function plots and returns the following quantities at the location of the given observing station:
 - `Er`: the vertical component of the electric field
 - `Btheta`: the N-S component of the magnetic field
 - `Bphi`: the E-W component of the magnetic field
 


## Acknowledgement

The schupy package is developed by G. Dálya, T. Bozóki, K. Kapás, J. Takátsy, E. Prácser and G. Sátori. Please send your questions and comments to `dalyag@caesar.elte.hu`. If you use the Schupy package for your research, please cite our paper.

