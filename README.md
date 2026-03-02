# Feature engineering, PPG og regression

- **Lecture specific files**: files/* – `En mappe med filer til øvelser og eksempler fra undervisningen.`

---

## Forberedelse til lektionen

Følg denne guide nøje for at være klar til undervisningen:

### 1. Literatur

**Primær litteratur:**
- - [Data Wrangling with Python af Jacek Gołębiewski (PDF)](https://datawranglingpy.gagolewski.com/datawranglingpy.pdf)
  - 5.1.1 Measures of location
  - 5.1.2 Measures of dispersion
  - 5.1.4 Box (and whisker) plots
  - 7.4 Visualising multidimensional data
  - 8.4 Pairwise distances and related methods
  - 9.2.2 from data to linear models
  - 9.2.3 Least Square Method
  - 9.2.4 Analysis od residuals
  - 9.2.7 Descriptive vs Predictive Power
  - 9.2.8 Regression with scikit learn

- [Databeskyttelsesloven (Retsinformation)](https://www.retsinformation.dk/eli/lta/2018/502)
  - Fokus: dataminimering, behandling af følsomme persondata og sikker opbevaring

**Supplerende litteratur:**
- [GeeksforGeeks: ML | Linear Regression](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)
- [TutorialsPoint: SciPy - Linear Curve Fitting](https://www.tutorialspoint.com/scipy/scipy_linear_curve_fitting.htm)
- [NumPy Documentation](https://numpy.org/doc/)
  - https://numpy.org/doc/stable/reference/generated/numpy.diff.html
  - https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
  - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
  - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html

---

### 2. Installationer og opsætning
- Sørg for at Python og VS Code er installeret (se evt. tidligere guides).
- Tjek at du har følgende extensions i Visual Studio Code:
  - `Python`
  - `jupyter`

- Download eller opdater materialet:
> ```zsh
> git clone https://github.com/AAU-ST2-Programming/signals_3.git
> cd signals_3
> git pull
> ```

---

## Lektionens fokus

- Feature engineering fra biosignaler (EKG/PPG)
- Variation, usikkerhed og visualisering med error bars
- Introduktion til lineær regression og residualer
- Fortolkning af sammenhænge mellem features

---

## Forventninger til forberedelse og undervisning

- **Før/efter kursusgang:**
  - Gennemgå tidligere kursusgange og kodeeksempler
  - Læs nyt materiale som beskrevet ovenfor
- **Tidsforbrug:**
  - 4 timers forberedelse (hjemme, før undervisning)
  - 4 timers undervisning og gruppeopgaver
  - 4 timers individuel opgaveregning (hjemme, efter undervisning)

---

## Spørgsmål og opgaver

- Til hver opgave i undervisningen vil der være:
  - En opgavebeskrivelse
  - En guide til hvordan opgaven løses
  - Svar på opgaven
- Opgaverne bygger videre på hinanden og bliver gradvist sværere.
- Til eksamen vil der kun være en opgavebeskrivelse – du skal selv kunne vurdere, hvordan opgaven løses.

---

**Husk:** Brug "Data Wrangling with Python" og regressions-eksemplerne i notebooken som din primære kilde til feature-analyse.
