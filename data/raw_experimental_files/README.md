The files in here are available online at https://www.nonlinearbenchmark.org/benchmarks and were the non-linear benchmarking sources of the data used in this research work. 

# The four real nonlinear benchmark datasets

These are the exact datasets used in our L4DC paper, with their official download pages:
Benchmark	Download	Files our code expects:

 - Silverbox — electronic Duffing-like oscillator	Silverbox dataset	SNLS80mV.csv
 - EMPS — Electro-Mechanical Positioning System, with nonlinear friction	EMPS dataset	DATA_EMPS.mat, DATA_EMPS_PULSES.mat
 - Wiener–Hammerstein — nonlinear electronic system	Wiener–Hammerstein dataset	WienerHammerBenchmark.csv
 - Cascaded Tanks — coupled fluid tanks with nonlinear flow and overflow	Cascaded Tanks dataset	dataBenchmark.csv

N.B.
The repository already has an optional downloader for the first three:

```bash
python -m pip install -r requirements-download-optional.txt
python scripts/fetch_benchmarks.py --dataset all --download
```


Cascaded Tanks requires manual retrieval from the 4TU dataset record. Place dataBenchmark.csv in external_data/.

The repository also contains Lorenz-63 and Duffing simulated nonlinear systems, plus the simulated two-degree-of-freedom predictive-maintenance twin.
