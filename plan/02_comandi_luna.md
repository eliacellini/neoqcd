# Comandi operativi per Luna

Leggere prima [01_simulazioni_residual_4x4.md](01_simulazioni_residual_4x4.md).
I blocchi sono istruzioni da eseguire successivamente, non job già lanciati.
Usare **Bash** per gli array; non incollarli direttamente in una sessione zsh.
Non eseguire il job SLURM del repository sul Mac.

## 1. Ambiente, directory e test

```bash
set -euo pipefail
cd /Users/eliacellini/Desktop/scripting/neoqcd
PROJECT_DIR="$PWD"
PY="$PROJECT_DIR/.sunenv/bin/python"
CAMPAIGN="$PROJECT_DIR/results/nemcmc_snf/residual_frozen_4x4_20260916_v1"
HIST_DDP="$PROJECT_DIR/results/nemcmc_snf/beta_1p0_to_1p5_D4_T4_L4_B16_K4_100epochs_ddp"
HIST_DAEMON="$PROJECT_DIR/results/nemcmc_snf/beta_1p0_to_1p5_D4_T4_L4_B16_K4_100epochs_daemon"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export WANDB_MODE=disabled
export MPLCONFIGDIR=/private/tmp/neoqcd-mpl

"$PY" -m unittest test.test_residual_normalizing_flows test.test_residual_frozen_convolution -v
```

Se i test falliscono, fermare il preflight e riportare l'errore. Poi, per una
**campagna nuova**, verificare che `CAMPAIGN` non esista; se esiste non cancellarla:
riprendere soltanto i run mancanti oppure scegliere un nuovo suffisso v2.

```bash
test ! -e "$CAMPAIGN"
mkdir -p "$CAMPAIGN"
cp -R "$HIST_DDP/prior_cache" "$CAMPAIGN/prior_cache"
git rev-parse HEAD > "$CAMPAIGN/source_head.txt"
git status --short > "$CAMPAIGN/source_status.txt"
git diff --binary > "$CAMPAIGN/source_uncommitted.patch"
```

Controllare gli exit code di ciascun comando, incluso `test`, prima di proseguire.
Salvare anche hash/copie degli untracked rilevanti, soprattutto il nuovo test;
`git diff` non li contiene. La creazione di questi artefatti avverrà soltanto
quando Luna eseguirà il piano. Nessun `git add` o `git commit`.

## 2. Argomenti comuni

Da usare nello stesso ambiente Bash del blocco precedente:

```bash
COMMON=(
  --domain beta --cpu --backend gloo
  --D 4 --T 4 --L 4 --N 3
  --beta-initial 1.0 --beta-final 1.5
  --batch-size 16 --protocol-steps 4
  --thermal-steps 100 --prior-mcmc-steps 1
  --updates-per-layer 1 --orsteps 0
  --flow-lr 0.002 --grad-clip-norm 10.0
  --hyper-time-embedding-dim 8 --hyper-activation silu
  --hyper-smearing-mode shared
  --hyper-normalize-by-nstep --hyper-scale-by-delta
  --residual-include-imag --residual-quadratic
  --residual-coeff-init 0.001 --residual-coeff-max 0.0
  --hyper-rho-init 0.001 --hyper-rho-eps 0.0 --hyper-rho-max 0.0
  --smearing-rho-init 0.001
  --cfg-cache-dir "$CAMPAIGN/prior_cache"
  --wandb-mode disabled --log-every 1
  --main-dir "$PROJECT_DIR"
)
```

Lanciare direttamente `.sunenv/bin/python`, non `torchrun`: il driver inizializza
Gloo anche per un processo. Verificare che la shell non erediti `WORLD_SIZE`,
`RANK` o `LOCAL_RANK` da un precedente lancio distribuito. La dimensione del batch
rimane globale 16; non dividerla perché i run storici usavano due processi.

## 3. Selezione della riga da eseguire

Per ogni riga del piano assegnare ID, NPT e MOD:

| ID | NPT | MOD |
| --- | ---: | ---: |
| H02 | 0 | 2 |
| H04 | 0 | 4 |
| H12 | 1 | 2 |
| H14 | 1 | 4 |
| H22 | 2 | 2 |
| H24 | 2 | 4 |

Esempio H12:

```bash
ID=H12
NPT=1
MOD=2
ARCH=hyper-residual
HIDDEN=32
DEPTH=2
TRAIN_SEED=1234
MODEL=(
  --nf-architecture "$ARCH"
  --hyper-hidden-dim "$HIDDEN" --hyper-depth "$DEPTH"
  --residual-n-pt "$NPT" --residual-mask-mod "$MOD"
)
```

Ricostruire `MODEL` dopo aver cambiato una variabile: un array Bash non aggiorna
retroattivamente i valori già espansi. Per ogni run registrare il comando effettivo.

## 4. Pilot, tre epoche

```bash
PILOT_DIR="$CAMPAIGN/pilot/$ID/s${TRAIN_SEED}/train"
mkdir -p "$(dirname "$PILOT_DIR")"
/usr/bin/time -p "$PY" -u main/main_nemcmc_snf.py \
  "${COMMON[@]}" "${MODEL[@]}" \
  --algorithm snf --phase train --train-steps 3 \
  --seed "$TRAIN_SEED" \
  --cfg-cache read --cfg-cache-tag beta1to15_D4_B16 \
  --run-name "pilot_${ID}_s${TRAIN_SEED}" --output-dir "$PILOT_DIR" \
  > "$(dirname "$PILOT_DIR")/train_console.log" 2>&1
```

Controllare il codice d'uscita e `summary.json`. Nessun `--overwrite`.
Ripetere per le sei righe serialmente; ricavare la stima di costo prima dei run
lunghi. Se si sceglie il controllo di caricamento del pilot, usare il comando di
evaluation del punto 6 con il checkpoint pilot, `--eval-samples 64` e output
separato sotto `pilot/`; non confonderlo con validation da 1008 campioni.

## 5. Training principale, cento epoche da zero

```bash
TRAIN_DIR="$CAMPAIGN/main/$ID/s${TRAIN_SEED}/train"
mkdir -p "$(dirname "$TRAIN_DIR")"
/usr/bin/time -p "$PY" -u main/main_nemcmc_snf.py \
  "${COMMON[@]}" "${MODEL[@]}" \
  --algorithm snf --phase train --train-steps 100 \
  --seed "$TRAIN_SEED" \
  --cfg-cache read --cfg-cache-tag beta1to15_D4_B16 \
  --run-name "${ID}_s${TRAIN_SEED}_train100" --output-dir "$TRAIN_DIR" \
  > "$(dirname "$TRAIN_DIR")/train_console.log" 2>&1
```

Non passare `--checkpoint` e non riutilizzare il pilot. Il file da valutare è
`$TRAIN_DIR/snf_checkpoint.pt`. Per seed training opzionali 1235/1236, cambiare
`--cfg-cache read` in `auto`: la copia storica contiene solo il seed 1234.

## 6. Validation e test dei modelli nuovi

Per la validation fissare `EVAL_SEED=2234`, `EVAL_LABEL=validation`,
`CACHE_TAG=residual4_val`. Per il test finale fissare a turno 3234/3235/3236,
`EVAL_LABEL=final`, `CACHE_TAG=residual4_test`.

```bash
EVAL_SEED=2234
EVAL_LABEL=validation
CACHE_TAG=residual4_val
EVAL_DIR="$CAMPAIGN/main/$ID/s${TRAIN_SEED}/${EVAL_LABEL}_s${EVAL_SEED}"
/usr/bin/time -p "$PY" -u main/main_nemcmc_snf.py \
  "${COMMON[@]}" "${MODEL[@]}" \
  --algorithm snf --phase evaluate --eval-samples 1008 \
  --seed "$EVAL_SEED" \
  --cfg-cache auto --cfg-cache-tag "$CACHE_TAG" \
  --checkpoint "$TRAIN_DIR/snf_checkpoint.pt" \
  --run-name "${ID}_train${TRAIN_SEED}_${EVAL_LABEL}_${EVAL_SEED}" \
  --output-dir "$EVAL_DIR" \
  > "${EVAL_DIR}_console.log" 2>&1
```

`TRAIN_DIR` deve restare quello del training seed 1234 mentre cambia il seed di
evaluation. Non ricostruirlo da `EVAL_SEED`. Non usare `--eval-steps 1008`: sarebbero
1008 batch, non 1008 campioni. Non usare `--train-steps` nella fase evaluate.
Il runner salva un checkpoint anche nell'output evaluation: conservare come
riferimento il checkpoint di training originale.
Prima di ciascuna evaluation ripristinare ID, MODEL e TRAIN_DIR della rete
corretta; non lasciare nell'array MODEL i valori dell'ultima variante eseguita.

## 7. Rivalutazione delle baseline archiviate

Preparare una riga per ciascuna delle tre baseline residual:

| OLD_ID | ARCH | HIDDEN | DEPTH | CHECKPOINT |
| --- | --- | ---: | ---: | --- |
| old_H32d2 | hyper-residual | 32 | 2 | `$HIST_DAEMON/hyper-residual_h32_d2/train/snf_checkpoint.pt` |
| old_H16d2 | hyper-residual | 16 | 2 | `$HIST_DDP/hyper_residual/train/snf_checkpoint.pt` |
| old_static | residual | 16 | 2 | `$HIST_DDP/residual_static/train/snf_checkpoint.pt` |

Esempio del checkpoint h32 d2:

```bash
OLD_ID=old_H32d2
ARCH=hyper-residual
HIDDEN=32
DEPTH=2
CHECKPOINT="$HIST_DAEMON/hyper-residual_h32_d2/train/snf_checkpoint.pt"
EVAL_SEED=2234
CACHE_TAG=residual4_val
OLD_EVAL_DIR="$CAMPAIGN/historical_recheck/$OLD_ID/s${EVAL_SEED}"
mkdir -p "$(dirname "$OLD_EVAL_DIR")"
/usr/bin/time -p "$PY" -u main/main_nemcmc_snf.py \
  "${COMMON[@]}" \
  --algorithm snf --phase evaluate --eval-samples 1008 \
  --nf-architecture "$ARCH" --hyper-hidden-dim "$HIDDEN" --hyper-depth "$DEPTH" \
  --residual-n-pt 0 --residual-mask-mod 2 \
  --seed "$EVAL_SEED" --cfg-cache auto --cfg-cache-tag "$CACHE_TAG" \
  --checkpoint "$CHECKPOINT" --output-dir "$OLD_EVAL_DIR" \
  --run-name "${OLD_ID}_s${EVAL_SEED}" \
  > "${OLD_EVAL_DIR}_console.log" 2>&1
```

Prima di lanciare verificare `checkpoint_semantics` contro il `run_meta.json`
storico. Tutte e tre le righe usano i default baseline n_pt=0/mod2, così i nuovi
campi non vengono aggiunti alla semantica attesa e i checkpoint originali restano
compatibili. Non adattare un vecchio checkpoint a n_pt>0.

Per il test finale ripetere **solo old_H32d2**, sui tre seed test, usando
`CACHE_TAG=residual4_test`. Le altre due rivalutazioni servono da contesto validation.

## 8. NEMCMC puro

```bash
EVAL_SEED=2234
CACHE_TAG=residual4_val
NEMCMC_DIR="$CAMPAIGN/nemcmc/s${EVAL_SEED}"
mkdir -p "$(dirname "$NEMCMC_DIR")"
/usr/bin/time -p "$PY" -u main/main_nemcmc_snf.py \
  "${COMMON[@]}" \
  --algorithm nemcmc --phase evaluate --nf-architecture smearing \
  --eval-samples 1008 --seed "$EVAL_SEED" \
  --cfg-cache auto --cfg-cache-tag "$CACHE_TAG" \
  --run-name "nemcmc_s${EVAL_SEED}" --output-dir "$NEMCMC_DIR" \
  > "${NEMCMC_DIR}_console.log" 2>&1
```

Nessun checkpoint e nessun flag residual nondefault. `smearing` qui è il valore
accettato dal parser per NEMCMC; il modello non applica layer neurali.
Ripetere sui tre seed test con tag `residual4_test`.

## 9. Ordine e checklist finale

1. Preflight -> sei pilot -> stima costo.
2. Sei training nuovi da 100 epoche, ciascuno seguito dalla propria validation.
3. Tre checkpoint storici rivalutati e NEMCMC sul seed validation.
4. Scrivere la scelta del vincitore senza vedere i seed test.
5. H02, vincitore, old_H32d2 e NEMCMC su tre seed test: dodici evaluation.
6. Diagnostiche dei checkpoint allenati, tabella e rapporto del piano principale.
7. Fermarsi: niente estensioni opzionali, altri volumi o commit automatici.

Campagna base: **6 pilot, 6 training completi, 22 evaluation da 1008 campioni**
(6 validation nuove + 3 storiche + 1 NEMCMC + 12 test), più le eventuali sei
evaluation pilot da 64 campioni. Nessun training nuovo per le baseline archiviate.

In caso di fallimento mantenere log e directory, annotare la causa e non usare
`--overwrite` per nascondere il tentativo. Un eventuale retry deve avere un suffisso
distinto e il medesimo protocollo; modifiche di LR/budget sono esperimenti diversi.
