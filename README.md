# Diplomová práce: Průzkum techniky destilace modelu pro efektivní hluboké učení klasifikačních transformerů

Tento repozitář obsahuje využité notebooky při tvorbě diplomové práce s názvem: Průzkum techniky destilace modelu pro efektivní hluboké učení klasifikačních transformerů.

Notebooky slouží k ověření provedených experimentů. Pro jejich bezchybný běh se předpokládá využití image NVIDIA Pytorch 2.5.0. na JupyterHubu od e-Infra. V případě jiného prostředí je nutné dokonfigurovat podporu GPU pro běh tréninků.

Před spuštěním notebooků s jednotlivými tréninky, porovnáním doby běhů nebo prohledáváním hyperparametrů, je třeba učinit následující kroky.
1) Nahrání souborů base.py a requirements.txt do kořenového adresáře JupyterHub serveru.
2) Připojení IDE k běhovému prostředí serveru.
3) Spuštění notebooku: install_libs.ipynb pro nainstalování dalších využitých knihoven
4) Spuštění notebooku: get_datatasets.ipynb pro stažení a základní úpravu využívaných datasetů a vytvoření adresářové struktury.
5) Spuštení notebooků s předpočítáním logitů pro každý dataset: podsložka preprocess.

Po provedení těchto kroků je prostředí připraveno na spouštění ostatních notebooků. 

