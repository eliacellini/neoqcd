# Inventario dell'integrazione upstream

Confronto di riferimento: mega merge `a6640c9` rispetto al primo genitore
`39cd3f0` in `snf_SUN`. L'integrazione è stata adattata al runner standalone
NEOQCD, mantenendo le opzioni e i percorsi locali già presenti.

| Modifica upstream | Esito in NEOQCD |
| --- | --- |
| Training SNF full, progressive e blockwise | Integrato in `main/main_nemcmc_snf.py`. Il full propaga il gradiente attraverso gli aggiornamenti MCMC; progressive congela il prefisso; blockwise ottimizza le perdite locali per blocco. |
| Accumulo gradienti, un all-reduce dei gradienti per epoca e checkpoint optimizer/scheduler | Integrato. `--grad-acc` accumula sui campioni della chain; le metriche di lavoro sono aggregate su tutti i rank e campioni. |
| Protocollo personalizzato e misure intermedie | Integrato con `--protocol-file`, validazione dei target e `--int-meas`; `observables.csv` registra separatamente azione pesata, plaquette semplice e plaquette al quadrato, insieme a lavoro, calore e log-Jacobiano. |
| Salvataggio atomico, resume esatto e output senza duplicati | Integrato. Un unico `snf_checkpoint.pt` contiene chain e RNG per rank, progressione, pesi, optimizer e scheduler. Il resume valida configurazione, RNG, forme, conteggi e header prima di modificare gli output, poi tronca le righe oltre il checkpoint. |
| Continuazione della chain in una nuova valutazione | Integrata con `--load-prior` / `--continue-prior`: carica chain e RNG dal checkpoint, ma apre un nuovo output di valutazione. |
| Transfer della rete rho funzionale tra protocolli di lunghezza diversa | Integrato con `--transfer-rho-functional`; controlla architettura e forme dei tensori ed è separato dal resume esatto. Il transfer in training richiede `--reset-optimizer`. |
| Inizializzazione prior Haar esatta | Integrata con `--prior-hot-start`; cache e checkpoint distinguono la semantica termica aggiornata. |
| MCMC gerarchico intorno al difetto | Integrato con `--hierarchical-levels`, `--hierarchical-updates` e `--hierarchical-buffers`, inclusi default dei buffer e controlli di geometria. I job SNF/NEMCMC sul difetto espongono le stesse opzioni via environment. |
| Jacobiano stout SU(3) full-Haar | Integrato in `neoqcd.smearing_jacobian`: il determinante complesso 9×9 è equivalente al determinante reale 8D sulla misura di Haar. I coefficienti usano formule chiuse e serie di Taylor stabili vicino a `Q=0` e allo spettro degenere; il backward esplicito è selezionabile. Checkpoint classic versionati rifiutano la vecchia convenzione del logdet. |
| Gather dei link selezionati in HBOR e smearing | Integrato. Le operazioni raccolgono i soli link attivi; il calcolo delle staples ruota un orientamento per volta. SU(3) aggiorna le due righe indipendenti richieste invece dell'intera matrice. |
| Kernel SU(N) e subgroup SU(2) | Integrati. La moltiplicazione matriciale evita `einsum` sul percorso CUDA; l'aggiornamento SU(2) opera sulle righe necessarie e corregge la scala `xi` del heatbath (`N/(2βk)`). |
| Maschere riusabili tra flow e HBOR | Integrate con indici `MaskedLinks` condivisi tra istanze basate sullo stesso `FlowPars`; la cache è limitata e si invalida quando cambiano maschera o difetto. I box gerarchici riusano gli indici senza duplicarli per ogni passo del protocollo. |
| Haar esatto, proiezione SU(3) e casi limite del heatbath | Integrati. L'hot start campiona SU(2)/SU(3) esattamente da Haar; `SU3_unitarize` completa le prime due righe ortonormali con la terza riga a determinante uno. Il heatbath SU(2) usa Haar esatto quando lo staple è nullo. La versione RNG/cache è incrementata per non riusare chain termalizzate con la semantica precedente. |
| Azione e osservabili | Integrati. L'azione Wilson usa `D(D−1)/2` piani e il parametro di difetto aggiornato. Il calcolo riusa plaquette già disponibili e restituisce media semplice di `P` e `P²`, separata dall'azione pesata. |
| Checkpoint generici e serializzazione numerica | Integrati in `neoqcd.utils`: salvataggio atomico di pesi, optimizer e scheduler, con versione del logdet classic; gli output scalari usano 17 cifre significative. |
| Checkpoint NEO-REX versionati e collettivi | Integrati in `NEOREX.save_flow/load_flow`: formato atomico con versione della convenzione SU(3), errori condivisi tra rank e broadcast dei pesi caricati. |
| Analisi con metodo Γ per autocorrelazioni | Integrata in `neoqcd.analysis` e nel runner standalone: rank 0 analizza separatamente ogni `measurement`, mantenendo ogni `sample` come replica indipendente lungo le traiettorie. Il report atomico `gamma_analysis.csv` contiene work, varianza, ΔF, ESS, heat, log-J, azione, plaquette/plaquette² e osservabili reweighted con errori, errore dell'errore, τint e finestra. La stessa analisi si può rilanciare su un CSV esistente con `python -m neoqcd.analysis`; resume ricalcola tutto il report. |

## Adattamenti e limiti

La cache del batch prior termalizzato resta separata dalla chain persistente del
checkpoint: il resume carica direttamente quest'ultima e non rifà la
termalizzazione. Le impostazioni di training e le geometrie preesistenti
restano disponibili, mentre i buffer gerarchici sono limitati ai difetti che
possono contenere le relative finestre.

I vincoli upstream SU(3)/D=4 non sono stati estesi alle API NEOQCD che già
supportano SU(2) o D=2. Il Jacobiano classico SU(3) non si applica a SU(2);
quest'ultimo conserva la propria convenzione. I coefficienti tangent-space dei
flow residual mantengono la loro matematica. Il percorso neurale con
`use_nn=True` mantiene il calcolo completo delle staples perché non è
equivalente alla selezione ottimizzata dei link classici.

I campi storici `work_sem` restano errori standard pooled IID per compatibilità;
il summary espone anche `work_sem_gamma_final` e lo stato/percorso del report
Gamma. Le serie con meno di cinque traiettorie per stream riportano
`insufficient_statistics`, senza sostituire l'errore autocorrelato con una
stima IID. L'analisi Gamma non viene applicata alle metriche non stazionarie di
training; `utils.analysis_and_output` mantiene la sua API esistente.
