Name	Year	Stack	Sample granularity	Vulnerability sample granularity	Size in functions (k)	Balance (vuln/non)	CWEs	CVEs	Source	Real-world?	Labeling technique	Known high label-inconsistency?	Format	Available?
PrimeVul	2024	C/C++	Function	Function	235		93		BigVul, CrossVul, CVEfixes, DiverseVul	Yes	Automatic	Low	json	Drive
RealVul	2024	C++	Project	Function 	271	0.02/0.98	63	1600 (from 2011 to 2019)	BigVul subset (most vulnerable)	Yes	Automatic	Low		Hugging face
DiverseVul	2023				350	0.05/0.95	150					Medium		
Project KB	2019	Java, Python						1300 (?)	NVD, MITRE	Yes			yaml, xml, steady	Github
DARPA AIxCC	2024						8		Linux Kernel, Jenkins, Nginx, SQLite3, Apache Tika	No 				No
Concoction	2024	C/C++				0.01/0.101			SARD, CVE	Partially	Manual			tarball
BigVul		C/C++	Vulnerability fixing commits	Function					348 OSS Github project	No	Automatic	High		
Devign									Linux Kernel‚ QEMU, Wireshark, FFmpeg	No		High		
ReVeal									Chromium, Debian	No		High		
SVEN	2024				1.6		9				Manual	High		
DBGBench														


https://huggingface.co/datasets/bigcode/the-stack-v2
https://huggingface.co/datasets/bigcode/the-stack-v2-dedup