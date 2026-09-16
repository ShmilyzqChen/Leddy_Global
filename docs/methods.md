# Methods and numerical conventions

## 1. Input fields

The stratification climatology is the annual objectively analysed WOA23
temperature and practical salinity on the native 0.25-degree, 102-standard-depth
grid. Pressure, Absolute Salinity, Conservative Temperature, and buoyancy
frequency are calculated with TEOS-10 (`gsw`). The effective bottom is derived
from the official WOA `landsea_04.msk`: code 1 is land; for a bottom first
encountered at standard level `n`, the effective depth is halfway between the
deepest wet level `n-1` and level `n`. Water deeper than the available WOA
profile is truncated at 5500 m.

EKE is calculated independently at every native 0.125-degree DUACS cell from
daily anomalous geostrophic velocities:

\[
EKE=\frac12\overline{u'^2+v'^2},
\]

where the overbar covers 1 January 1993 through 31 December 2024. Dates with
`flag_ice == 1` and packed missing values are excluded. There is no additional
temporal demeaning, error-variance subtraction, or velocity quality screening.
A native cell must contain at least 365 valid days distributed over at least
five years. Eligible native climatologies are then averaged equally in aligned
2-by-2 blocks to the WOA grid.

## 2. Stratification handling

For a retained water column, TEOS-10 supplies (N^2) between standard depths.
Isolated nonpositive or missing values are replaced by interpolation in
\(\log N^2\), with a numerical floor of \(10^{-8}\;\mathrm{s^{-2}}\). The
fraction of total water-column thickness affected by this operation is saved.
The short cap from the deepest observed standard level to the effective bottom
uses the deepest positive observed (N^2); this is a boundary construction and
is not counted as an unstable-profile correction.

Raw and corrected standard-midpoint (N^2) fields and a levelwise correction
flag are saved in the intermediate stratification file.

## 3. First baroclinic mode

The hydrostatic vertical-displacement form is

\[
-\frac{d^2G}{dz^2}=\lambda N^2G,\qquad
G(0)=G(H)=0,\qquad \lambda=\frac{1}{c^2}.
\]

It is equivalent to the rigid-lid, flat-bottom horizontal-structure problem.
Piecewise-linear finite elements are constructed directly on the nonuniform WOA
standard depths. With element thickness (h_e), the stiffness contribution is

\[
K_e=\frac{1}{h_e}
\begin{bmatrix}1&-1\\-1&1\end{bmatrix}.
\]

The (N^2)-weighted mass matrix is row-sum lumped, yielding a diagonal positive
mass matrix. Transforming (Kx=\lambda Mx) by (M^{-1/2}) gives a symmetric
tridiagonal eigenproblem. Its smallest eigenvalue is the first baroclinic mode,
and (c_1=\lambda_1^{-1/2}).

The diagnostic WKB estimate is

\[
c_{1,\mathrm{WKB}}=\frac{1}{\pi}\int_{-H}^{0}N(z)\,dz.
\]

The eigenmode estimate is primary. WKB is used if the eigenproblem fails or if
more than 20% of total water-column thickness required (N^2) correction. Both
estimates and the method flag are retained.

## 4. Horizontal dynamical scales

With Earth rotation rate \(\Omega\), radius \(a\), and latitude \(\phi\),

\[
f=2\Omega\sin\phi,\qquad
\beta=\frac{2\Omega\cos\phi}{a}.
\]

The globally continuous deformation-radius expression is

\[
L_d=\frac{c_1}{\sqrt{f^2+2\beta c_1}}.
\]

It approaches (c_1/|f|) away from the equator and
\(\sqrt{c_1/(2\beta)}\) at the equator. Therefore no artificial equatorial
latitude switch is needed.

The velocity and Rhines conventions are

\[
U_{rms}=\sqrt{2EKE},\qquad L_R=\sqrt{\frac{2U_{rms}}{\beta}}.
\]

Finally,

\[
L_{eddy,\,dyn}=\min(L_d,L_R),\qquad
\lambda_{eddy}=2\pi L_{eddy,\,dyn}.
\]

The first quantity is a dynamical length/radius convention. The second is the
corresponding wavelength convention used for direct comparison with spectral
axes; published products using different estimators need not differ by exactly
\(2\pi\).

## 5. Conservative spatial correction

EKE and final (c_1) are assessed separately in log space using a 5-by-5 ocean
neighbourhood. A candidate must differ from the local median by more than six
MAD-based robust standard deviations, have at least eight compatible neighbours,
and belong to an isolated component no larger than four cells. Small missing
components no wider than two cells can also be filled. Replacement uses a
distance-weighted geometric mean of adjacent valid values.

Neighbours must be ocean cells and their effective depths cannot differ by more
than a factor of two. This reduces interpolation across land and major shelf
breaks. Larger gaps remain missing. Raw values, corrected values, and flags are
all retained; derived scales are recomputed from corrected component fields.
