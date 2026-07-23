## Paper visione classica: PatchMatch

#### Introduzione

Il paper **PatchMatch** presenta strumenti di **image editing interattivo** basati su un algoritmo **randomizzato** capace di trovare rapidamente corrispondenze tra **patch**, formulate come problemi di **Approximate Nearest Neighbors (ANN)**.

L’osservazione di partenza è che molte tecniche di grafica **patch-based** producono risultati di alta qualità, ma risultano computazionalmente costose perché richiedono, per ogni patch, la ricerca della patch più simile all’interno di un’altra immagine o regione. Con **PatchMatch**, questa operazione viene accelerata a tal punto da diventare quasi **interattiva**.

## Progetto: implementazione e valutazione di PatchMatch

Il progetto implementa il calcolo del **Nearest-Neighbor Field (NNF)** e ne dimostra il comportamento tramite esperimenti quantitativi e qualitativi, includendo anche un’estensione per **inpainting multi-scala** in stile _Search + Vote_.

L’idea centrale è separare chiaramente:

- il **core algoritmico** di PatchMatch, cioè la stima approssimata di corrispondenze tra patch di due immagini
- alcune **applicazioni e analisi sperimentali**, tra cui ricostruzione, studio della convergenza, confronto con brute force, toy example passo-passo, failure cases, inpainting e confronto con un metodo deep

### Codici originali repository
repo: [https://github.com/MingtaoGuo/PatchMatch/tree/master](https://github.com/MingtaoGuo/PatchMatch/tree/master)
#### `PatchMatch.py`
Il codice implementa una versione basica di PatchMatch per stimare un campo di corrispondenze tra due immagini. Per ogni pixel o centro-patch dell’immagine **A** viene cercata una patch simile in **B** tramite:
- **inizializzazione casuale** del campo `f` (NNF) e della mappa delle distanze `dist`
- **propagazione** in scanline order (dispari) e reverse order (pari)
- **random search** con finestre decrescenti
- **ricostruzione** finale copiando, per ogni pixel di A, il valore di B nella posizione indicata dal NNF

**Nota su codice originale:** nella versione base, la propagazione seleziona direttamente il match del vicino senza applicare in modo esplicito lo shift coerente con l’idea teorica di propagare il candidato del vicino alla posizione corrente. Inoltre, il random search parte con finestre già relativamente piccole rispetto alla formulazione standard del paper.

#### `PatchMatch_Bidirectional.py`
Il codice contiene una variante che:
- usa una distanza patch che include anche canali o feature aggiuntive
- normalizza globalmente gli input prima del matching
- implementa una propagazione con shift $\pm 1$, più vicina alla descrizione del paper

#### Elementi presi dai codici originali

- **Schema complessivo PatchMatch**: inizializzazione casuale + iterazioni alternate (scanline/reverse) composte da propagazione e random search.
- **Gestione dei bordi tramite padding e mascheramento**: in `PatchMatch.py` il confronto ignora i `NaN`; la stessa idea viene mantenuta nella nuova distanza con `np.nansum` e conteggio dei pixel validi.
- **Propagazione con shift**: la logica “shifted $\pm 1$” coerente col paper è presente nel codice bidirezionale ed è stata ripresa nella nuova implementazione.

#### Modifiche principali
- **Rappresentazione dati più pulita**: NNF come array `(H,W,2)` `int32` e NND come array `float64`, evitando `dtype=object` del codice base.
- **Random search allineato al paper**: finestra che si restringe esponenzialmente con fattore `alpha`, partendo da un raggio massimo pari alla dimensione dell’immagine.
- **Visualizzazione NNF in HSV** (hue = direzione, value = magnitudine), dichiarata come analoga alla rappresentazione intuitiva del campo di offset.
- **Ricostruzione in due modalità**: ricostruzione **pixel-wise** e ricostruzione tramite **media di patch sovrapposte**, utile per ottenere risultati più stabili e meno rumorosi.
- **Storico dell’evoluzione dell’algoritmo**: funzione `patchmatch_history`, utile per visualizzare iterazione per iterazione NNF, RMS map e ricostruzione.
- **Estensione “Inpainting multi-scale Search+Vote”**: implementazione di una pipeline completa (piramide, inizializzazione, PatchMatch vincolato, voto) che non era presente nei codici originali.
#### Inpainting multi-scala

L’inpainting viene eseguito tramite una pipeline coarse-to-fine:
- costruzione di una piramide multi-scala di immagine e maschera
- inizializzazione grezza del buco
- iterazioni tipo EM che alternano una fase **Search** (PatchMatch vincolato a patch sorgenti completamente note) e una fase **Vote** (ricostruzione dei pixel mancanti tramite aggregazione di patch sovrapposte)

Più in dettaglio:
- la sorgente valida viene vincolata a patch interamente contenute nella regione nota
- la distanza usata in inpainting è una **MSE pesata da una confidence map**
- alle scale più fini il contenuto mancante viene inizializzato tramite **upsampling** del risultato della scala precedente

#### `demo.py` e analisi dei risultati

Qui vengono eseguiti i seguenti esperimenti:
- **1**: NNF + NNF visualization + reconstruction
- **2**: patchmatch convergence (RMS per iteration)
- **3**: multiscale inpainting search + vote
- **4**: Root Mean Square (RMS) - patchmatch vs ground truth
- **5**: effect of patch size on quality and time
- **6**: toy example step-by-step    
- **7**: failure cases    
- **8**: patchMatch vs LaMa
- **9**: applications

**Metodologia comune: metrica RMS**
La funzione `rms_from_nnd` definisce RMS come:  
$$  
\text{RMS} = \sqrt{\max(\mathrm{mean}(\text{NND}), 0)}  
$$
dove NND è la mappa delle MSE normalizzate.

La RMS viene usata come misura sintetica della qualità media del matching: valori più bassi indicano corrispondenze migliori.

#### Esperimento 1 - NNF + visualizzazione NNF + ricostruzione

L’esperimento esegue una singola stima del **Nearest-Neighbor Field** tra l’immagine sorgente **A** e la target **B**, ottenendo in output il campo `nnf` e la mappa delle distanze `nnd`.

La visualizzazione finale comprende:
- immagine sorgente **A**
- immagine target **B**
- **NNF visualization** in HSV
- **ricostruzione pixel-wise**
- **ricostruzione da patch sovrapposte**
- **RMS map**
    
![[Pasted image 20260416081612.png]]  
_In alto: immagine sorgente A, immagine target B e visualizzazione del Nearest-Neighbor Field (NNF) in HSV (hue = direzione dell’offset, value = magnitudine). In basso: ricostruzione pixel-wise, ricostruzione ottenuta mediando patch sovrapposte e mappa locale dell’errore RMS._

**Risultati osservati**

- `time = 62.02s`, `RMS = 8.0839`

Il valore RMS indica l’errore medio tra patch di A e patch corrispondenti in B secondo l’NNF stimato: un RMS più basso implica corrispondenze mediamente migliori rispetto all’inizializzazione casuale.

La ricostruzione **pixel-wise** è utile come diagnostica, ma può produrre artefatti locali perché copia un solo pixel per posizione. La ricostruzione **patch-based** tende invece a essere più stabile, perché combina contributi di patch sovrapposte. In questo senso, l’esperimento evidenzia bene la differenza tra semplice uso del NNF come mappa di corrispondenze e uso del matching come base per una vera ricostruzione.

#### Esperimento 2 - Convergenza PatchMatch (RMS per iterazione)

L’esperimento misura la **convergenza** dell’algoritmo registrando la RMS a ogni iterazione. Per rendere i risultati riproducibili, si parte da una inizializzazione fissata.

  ![[Pasted image 20260416081740.png|400]]
_Curva della RMS in funzione delle iterazioni: l’iterazione 0 corrisponde all’inizializzazione casuale del NNF, mentre i punti successivi mostrano il miglioramento progressivo dovuto a propagazione e random search, con rapido calo iniziale e successivo assestamento._

**Risultati osservati**
- iter 0: RMS = 87.9349
- iter 1: 11.2674
- iter 2: 8.9630
- iter 3: 8.3566
- iter 10: 7.7034

La discesa molto rapida dall’iterazione 0 all’iterazione 1 è il comportamento atteso: l’inizializzazione casuale è molto rumorosa, mentre già una singola iterazione di cooperazione locale (propagazione) + esplorazione (random search) migliora drasticamente la qualità del matching.

Dopo poche iterazioni la curva tende ad appiattirsi. Questo è coerente con l’idea del paper secondo cui poche iterazioni sono spesso sufficienti per una convergenza pratica.

#### Esperimento 3 - Inpainting multi-scala Search+Vote

L’esperimento costruisce un caso di **inpainting** introducendo una maschera rettangolare su A e generando un’immagine danneggiata in cui i pixel nel buco vengono neutralizzati. L’inpainting viene poi eseguito con una pipeline coarse-to-fine:
- costruzione di una piramide multi-scala dell’immagine e della maschera
- inizializzazione del buco con un riempimento grezzo
- iterazioni tipo EM in cui si alternano una fase **Search** (PatchMatch vincolato a patch sorgenti completamente note) e una fase **Vote** (ricostruzione dei pixel del buco tramite aggregazione di patch sovrapposte)

![[Pasted image 20260416081800.png]]
_Confronto tra immagine originale, maschera, immagine danneggiata con hole rettangolare e risultato finale dell’inpainting._

**Risultati osservati**  
Tre scale: 60×44 $\rightarrow$ 120×89 $\rightarrow$ 240×179.

- A scale 0 la RMS scende da 15.3950 a 8.7005
- A scale 1 scende da 6.3000 a 5.2321
- A scale 2 oscilla ma resta nell’intorno 4.1-4.7, terminando a 4.3738  
    Tempo totale: 242.54s

Il comportamento multi-scala è coerente con la strategia coarse-to-fine: a bassa risoluzione si impone una struttura globale plausibile, poi alle scale maggiori si rifiniscono texture e dettagli.

Le oscillazioni a scala piena sono compatibili con un ciclo iterativo “Search+Vote”: il voto modifica l’immagine corrente, e quindi modifica anche il paesaggio delle distanze al passo successivo. Non è quindi richiesta una monotonia perfetta dell’errore a ogni iterazione EM.

L’esperimento mostra bene come PatchMatch possa essere usato non solo per stimare corrispondenze, ma anche come motore di una pipeline di **completion/inpainting**.

#### Esperimento 4 - Confronto PatchMatch vs Ground Truth (brute force su subset)

L’esperimento fornisce una validazione quantitativa confrontando PatchMatch con una soluzione **esatta** calcolata tramite ricerca esaustiva su un sottoinsieme di punti query.

![[Pasted image 20260416081816.png]]
_A sinistra: distribuzione della RMS per ground truth e PatchMatch. Al centro: scatter plot RMS(PM) vs RMS(GT) con retta $y=x$. A destra: distribuzione di $\Delta$RMS = RMS(PM) − RMS(GT)._

**Risultati osservati**
- 300 query (stride=10), brute force: 12.39M confronti
- tempi: PatchMatch ≈ 62.50s (6 iterazioni), GT ≈ 115.3s (subset)
- RMS mean: GT = 4.7821, PM = 5.0274, rapporto PM/GT = 1.051
- delta medio (PM−GT) = 0.2452
- violazioni PM < GT: 0/300

Ci si aspetta **PM ≥ GT**: la brute force calcola il vero minimo nel dominio dei candidati testati, mentre PatchMatch è approssimato. L’assenza di “violazioni” è quindi un controllo di coerenza importante.

Il rapporto 1.051 indica che, sul subset testato, PatchMatch produce risultati mediamente solo di circa il 5% peggiori dell’ottimo, a fronte di un tempo inferiore e di una procedura molto più scalabile di una brute force completa.

#### Esperimento 5 - Effetto della patch size su qualità (RMS) e tempo

L’esperimento analizza l’impatto della dimensione della patch ripetendo PatchMatch con diversi valori di `patch_size`, mantenendo fissi gli altri iperparametri principali.

![[Pasted image 20260416081839.png|400]]
_Andamento della RMS al variare della dimensione della patch: patch più grandi impongono vincoli di somiglianza più stringenti e, nel caso in esame, producono un incremento dell’errore medio._

![[Pasted image 20260416081848.png]]
_Confronto visivo tra l’immagine originale A e le ricostruzioni ottenute con diverse dimensioni di patch tramite media di patch sovrapposte. Ogni pannello riporta la RMS corrispondente._

**Risultati osservati**

- ps = 3, RMS = 6.2249, time ≈ 62.36s
- ps = 5, RMS = 8.1291, time ≈ 63.40s
- ps = 7, RMS = 9.9549, time ≈ 65.20s    
- ps = 9, RMS = 11.5504, time ≈ 65.84s

Nel setup considerato, patch più grandi producono valori medi di RMS più elevati. Questo è coerente con il fatto che patch più grandi impongono un vincolo di somiglianza più rigido, rendendo più difficile trovare corrispondenze molto vicine.

Va però ricordato che, cambiando la patch size, cambia anche la nozione operativa di similarità tra patch. Per questo il confronto numerico tra RMS ottenute con patch diverse va letto con cautela e accompagnato dal confronto visivo delle ricostruzioni.

Il tempo cresce moderatamente nel range testato, in modo coerente con l’aumento del costo del calcolo della distanza all’aumentare del numero di pixel per patch.

#### Esperimento 6 - Toy example: visualizzazione step by step

Questo esperimento introduce un **toy example sintetico** in cui l’immagine **B** è ottenuta traslando una configurazione semplice di forme colorate presente in **A**. L’obiettivo è visualizzare in modo chiaro l’evoluzione di PatchMatch su un caso controllato.

Per ogni iterazione vengono salvati:
- la **ricostruzione** ottenuta dal NNF corrnte
- la **visualizzazione del NNF**
- la **RMS map**

Inoltre viene tracciata la curva della RMS nel tempo e vengono mostrati separatamente gli input toy A e toy B.

![[Pasted image 20260416081919.png|400]]

![[Pasted image 20260416081937.png|400]]

![[Pasted image 20260416081946.png]]

**Risultati osservati**
- iter 0: RMS ≈ 94.94
- iter 1: RMS ≈ 18.22
- iter 2: RMS ≈ 11.90
- iter 3: RMS ≈ 11.12
- iter 4: RMS ≈ 10.82
- iter 5: RMS ≈ 10.58

Anche in questo caso si osserva una forte riduzione dell’errore nelle prime iterazioni, ma soprattutto il toy example rende **visivamente interpretabile** il metodo: da una inizializzazione casuale e incoerente si passa progressivamente a un NNF più regolare, a una ricostruzione più simile alla struttura corretta e a una mappa dell’errore più concentrata.

Questo esperimento è utile perché mostra non solo il risultato finale, ma i **passaggi intermedi** del metodo, cioè il modo in cui le corrispondenze migliorano iterazione dopo iterazione.

#### Esperimento 7 - Failure cases

L’esperimento raccoglie due casi in cui un approccio patch-based come PatchMatch-inpainting può andare in difficoltà:
- **texture ripetitive**
- **strutture lunghe che attraversano il buco**

Nel primo caso viene usata una texture a scacchiera; nel secondo, una scena sintetica con linee lunghe diagonali che devono essere ricostruite attraverso una regione mancante.
![[Pasted image 20260416082037.png]]

![[Pasted image 20260416082042.png]]

**Risultati osservati**

Nel caso di **texture ripetitiva**, il matching locale trova molte corrispondenze quasi equivalenti: il risultato può quindi apparire plausibile a livello locale, ma non necessariamente ricostruire in modo univoco la configurazione originaria.

Nel caso di **strutture lunghe**, il limite è più evidente: PatchMatch lavora bene quando può copiare e riaggregare pattern locali già presenti, ma fatica di più quando serve mantenere una **coerenza geometrica globale** attraverso il buco. In questi casi la continuità delle linee può degradarsi o risultare spezzata.

Questo esperimento è importante perché mostra che il metodo non va valutato solo sui casi in cui funziona bene, ma anche sui casi in cui la natura locale e patch-based del matching diventa un limite.

#### Esperimento 8 - Confronto PatchMatch vs LaMa

L’esperimento confronta l’inpainting PatchMatch con **LaMa**, usando un esempio sintetico realistico con due tipi di maschera:
- **small hole**
- **large hole**

Nel progetto, l’output di LaMa viene caricato da file precomputati nella cartella `lama_outputs`, mentre il risultato PatchMatch viene prodotto direttamente dal codice locale. Il confronto viene effettuato sia **visivamente** sia tramite **MSE sulla regione mascherata**.

![[Pasted image 20260416082116.png]]
![[Pasted image 20260416082123.png]]

**Risultati osservati**

Il confronto mette in evidenza una differenza di principio tra i due approcci:
- **PatchMatch** ricostruisce il contenuto copiando e riaggregando patch esistenti
- **LaMa** sfrutta un modello deep in grado di inferire strutture più globali e semanticamente coerenti

Nel caso di **buchi piccoli**, PatchMatch può ancora produrre risultati plausibili quando il contenuto da ricostruire è localmente ben supportato dal contesto.  
Nel caso di **buchi grandi**, i limiti del metodo patch-based emergono più chiaramente: mantenere forma e coerenza globale diventa più difficile, mentre il metodo deep tende a ricostruire meglio strutture e contorni.

Questo esperimento non sostituisce un benchmark completo, ma è utile per collocare PatchMatch rispetto a metodi più moderni: il primo è molto interpretabile e direttamente legato al matching tra patch, mentre il secondo introduce una capacità di completamento più semantica.

#### Esperimento 9 - Possibili applicazioni

L’ultimo esperimento costruisce un pannello qualitativo di possibili applicazioni di un approccio PatchMatch-based all’inpainting:
- **object removal**
- **scratch repair**
- **logo removal**

In tutti i casi si parte da una scena sintetica e si introduce una maschera specifica per simulare il tipo di danno o di editing desiderato.

![[Pasted image 20260416082149.png]]

**Risultati osservati**

L’esperimento mostra che l’inpainting è una delle applicazioni più naturali di PatchMatch: quando l’informazione mancante può essere ricostruita riutilizzando porzioni già presenti dell’immagine, il meccanismo di search + vote risulta particolarmente adatto.

In particolare:
- nella **rimozione di piccoli oggetti** il metodo può funzionare bene se il contesto vicino contiene texture o pattern compatibili
- nel **ripristino di graffi o regioni sottili** l’approccio patch-based è naturale perché il problema è fortemente locale
- nella **rimozione di loghi o elementi sovrapposti** il risultato dipende molto dalla disponibilità, attorno alla maschera, di contenuto simile da cui copiare

## Conclusioni sul progetto

Il progetto realizza una implementazione funzionale e sperimentabile di PatchMatch, separando:

- **core algoritmico** (`patchmatch.py`): NNF/NND, propagazione, random search, ricostruzione, storico dell’evoluzione e visualizzazione
- **validazione sperimentale** (`demo.py`): set di esperimenti riproducibili con metriche e figure
- **estensione applicativa**: inpainting multi-scala, failure cases, confronto con un metodo deep e pannello di applicazioni

Dal punto di vista sperimentale emergono i seguenti aspetti principali:

1. PatchMatch riduce drasticamente l’errore già nelle prime iterazioni
2. l’approssimazione rispetto all’ottimo brute force rimane contenuta
3. il toy example rende visivamente chiari i passaggi del metodo
4. il framework Search+Vote multi-scala rende naturale estendere il metodo a compiti di inpainting
5. i failure cases mostrano i limiti del matching puramente locale
6. il confronto con LaMa evidenzia la differenza tra approcci patch-based classici e metodi deep più recenti
    
Nel complesso, il progetto mostra sia i **punti di forza** di PatchMatch, cioè semplicità concettuale, efficienza pratica, interpretabilità del matching, sia alcuni suoi **limiti strutturali**, soprattutto quando serve ricostruire contenuti con forte coerenza globale o semantica.

## Paper Visione Deep: Deformable Convolutional Networks

### Introduzione

Le **Deformable Convolutional Networks (DCN)** introducono due moduli “drop-in” per reti CNN: la **deformable convolution** e la **deformable RoI pooling** (inclusa la variante **deformable position-sensitive RoI pooling**).

L’idea centrale è rendere **learnable** e quindi **content-adaptive** le posizioni di campionamento usate da convoluzioni e pooling su RoI: invece di usare una griglia regolare fissa, ogni punto del kernel o ogni bin della RoI viene spostato da un **offset** predetto dalla rete stessa. In questo modo il receptive field diventa più adattivo rispetto a forma, scala e pose degli oggetti, migliorando detection e segmentation con overhead relativamente contenuto.


## Progetto su Deformable Convolutional Networks

Questo progetto implementa, in forma semplificata, il contributo centrale del paper, cioè l’idea di sostituire il campionamento a griglia fissa delle convoluzioni standard con un campionamento **deformabile** i cui offset sono **appresi** a partire dalle feature map.

L’obiettivo non è stato replicare l’intera pipeline della repository ufficiale, che è orientata al _detection_ e quindi GPU dipendente, bensì:
- realizzare una versione **utilizzabile su CPU**
- testare la repo per fare degli esperimenti
- produrre output riproducibili: curve di training, metriche, grafici di robustezza e visualizzazioni degli offset

Il progetto è strutturato attorno a tre esperimenti:
1. confronto Standard CNN vs Deformable CNN su **MNIST** e **FashionMNIST**
2. visualizzazione qualitativa degli **offset appresi**
3. misura della **robustezza** a trasformazioni geometriche (rotazione, scala, shear)

#### Repository originale
repo: [https://github.com/msracver/Deformable-ConvNets.git](https://github.com/msracver/Deformable-ConvNets.git)

La repository ufficiale del paper è progettata principalmente per compiti di **object detection** (e task affini) e contiene:
1. **Operatori custom (C++/CUDA)** per _deformable convolution_ e _deformable RoI pooling_.
2. Backbone e **Pipeline di training/evaluation** per modelli di detection (es. varianti tipo Faster R-CNN / R-FCN con integrazione dei layer deformabili)
3. **Configurazioni e script** orientati a dataset complessi (COCO/PASCAL), con dipendenza da GPU per i costi computazionali
Questo rende la repo ottima come riferimento e baseline di implementazione, ma poco pratica su CPU.

### Cosa è stato preso dalla repository

È stata mantenuta la struttura generale **conv standard + branch offsets**:

- un ramo convoluzionale aggiuntivo predice gli offset
- gli offset deformano il pattern di campionamento della convoluzione
- gli offset del ramo sono inizializzati a zero, così da far partire la rete come baseline standard

Per un kernel $k\times k$, con $N=k^2$ punti di campionamento, il ramo offset produce **$2N=2k^2$ canali**.

È stato inoltre scelto di inserire la deformable convolution solo negli **ultimi layer**, mantenendo i primi layer standard. Questa scelta è coerente con l’idea che ai livelli più alti le feature siano più semantiche e possano beneficiare maggiormente di un receptive field adattivo.

### Modifiche per implementazione CPU

- Pipeline in **PyTorch**
- Sostituzione dell’operatore custom CUDA della repository con **`torchvision.ops.DeformConv2d`**, eseguibile anche su CPU
- Sostituzione del contesto detection con un contesto di **classification** su MNIST/FashionMNIST, così da rendere:
    - il training più rapido e riproducibile
    - più immediate le visualizzazioni degli offset
    - più semplici gli esperimenti di robustezza, senza introdurre bounding box o RoI pooling

#### File del progetto
Il progetto è strutturato attorno a cinque esperimenti:
1. confronto **StandardCNN vs DeformableCNN** su **MNIST** e **FashionMNIST**
2. visualizzazione qualitativa degli **offset appresi**
3. misura della **robustezza** a trasformazioni geometriche (**rotazione, scala, shear**)
4. costruzione di **toy visualizations** per spiegare visivamente il funzionamento di convoluzione standard, convoluzione deformabile, pooling e campionamento con **interpolazione bilineare**
5. confronto qualitativo delle **feature responses** nei layer standard e deformabili, per osservare come cambia la risposta dei filtri nei diversi stadi della rete

I nuovi esperimenti 4 e 5 non modificano il modello né il training, ma estendono la parte di **interpretabilità** del progetto. L’idea è passare da una semplice osservazione degli offset a una lettura più completa del comportamento del meccanismo deformabile: da un lato con esempi didattici controllati, dall’altro con un confronto diretto tra le attivazioni interne delle reti standard e deformabile.

### Valutazione risultati

#### Esperimento 1

**Dataset MNIST**  
![[Pasted image 20260414184643.png|400]]

La DeformableCNN raggiunge una **test accuracy finale di 95.8%**, contro **90.4%** della StandardCNN.

La lettura più corretta di questo risultato è che, nel setup sperimentale considerato, la DeformableCNN mostra una migliore capacità di adattarsi alla geometria locale dei digit, ottenendo una performance di test superiore dopo lo stesso numero di epoche.

**Costo**
- Parametri: **148,074 → 174,048**
- Inference: **0.52 ms** vs **2.76 ms**
- Tempo per epoca: circa **3.6s** vs **106s**

Quindi il guadagno in accuracy è accompagnato da un overhead significativo, soprattutto su CPU.

**Dataset FashionMNIST**  
FashionMNIST è più complesso (texture, forme più variabili), quindi è più realistico per la “capacità geometrica” del modello.
![[Pasted image 20260414184752.png|400]]

Anche su FashionMNIST la DeformableCNN mantiene **test loss più bassa** e **test accuracy più alta**: **75.6%** contro **71.0%**.

La StandardCNN mostra un comportamento meno stabile nelle ultime epoche, mentre la DeformableCNN continua a migliorare fino alla terza epoca. Nel contesto di questo esperimento, è corretto parlare di **migliore performance di test nel setup considerato**, non di prova generale di migliore generalizzazione.

**Costo**
- Inference: **0.52 ms** vs **2.97 ms**
- Tempo per epoca deformabile: circa **106-109s**

Poiché le accuracy sono misurate su subset ridotti, i valori assoluti non sono confrontabili con benchmark SOTA, ma restano adeguati per un confronto controllato tra architetture.

#### Esperimento 2

Questo esperimento serve a collegare il **meccanismo** alla **prestazione**, mostrando come il campionamento venga modificato dagli offset appresi.

**Come leggere le figure**
- **Blu** = griglia regolare 3×3
- **Rosso** = punti di campionamento dopo l’applicazione degli offset appresi
- **Stella gialla** = posizione centrale di riferimento

**Dataset MNIST**  
![[Pasted image 20260414184830.png]]

Nei digit sottili, come “1”, i punti rossi **sembrano** disporsi più spesso lungo la direzione dominante dello stroke. Questo suggerisce che il kernel non campioni più in modo perfettamente simmetrico, ma adatti il proprio supporto alle regioni localmente più informative.

Nei digit curvi, come “0”, “6” e “9”, i punti rossi **appaiono compatibili** con un campionamento che segue meglio la geometria locale della cifra.

Andando da **Deform Layer 1** a **Deform Layer 3**, l’ampiezza degli spostamenti tende ad aumentare, suggerendo qualitativamente che i layer più profondi si prendano maggiore libertà nel ridefinire il campo recettivo.

**Dataset FashionMNIST**  
![[Pasted image 20260414184857.png]]
Su oggetti con struttura più complessa, i punti rossi **sembrano** spostarsi più spesso verso bordi e contorni, cioè regioni potenzialmente più informative dal punto di vista discriminativo.

Anche qui, nei layer più profondi la deformazione appare più marcata. Questa osservazione è coerente con l’idea che le deformable convolution possano adattarsi meglio a variazioni di forma intra-classe, ma la visualizzazione resta un’evidenza interpretativa, non una prova diretta.

#### Esperimento 3

Questo esperimento valuta la robustezza dei modelli rispetto a trasformazioni geometriche tramite test set trasformati con **rotazione**, **scala** e **shear**.
##### Transformed samples
![[Pasted image 20260414184919.png]]
![[Pasted image 20260414184926.png]]Le immagini trasformate mostrano che:
- le rotazioni forti introducono spesso cropping, aliasing e orientamenti molto distanti dal training set
- le scale piccole riducono drasticamente il numero di pixel informativi
- shear elevati deformano in modo pesante la geometria locale

##### Geometric robustness

**Dataset MNIST**
![[Pasted image 20260414184942.png]]

**Rotazione**  
La DeformableCNN tende a mantenere un vantaggio per rotazioni moderate, ma per rotazioni molto forti entrambe le reti degradano pesantemente. Questo è coerente con il fatto che la deformabilità locale non equivale a una vera rotation equivariance.

**Scala (0.5 $\rightarrow$ 1.5)**
Qui emerge il risultato più forte: la DeformableCNN è **nettamente più robusta**.
- Scale 0.5: **52.3% vs 14.3%** (gap enorme)
- Scale 1.2: **94.7% vs 73.4%**
- Scale 1.5: **65.9% vs 35.7%**
Con scala diversa, l’oggetto occupa porzioni diverse del campo recettivo. La deformable conv può **spostare i sampling points** per “rincorrere” il segnale e preservare la qualità delle feature anche quando la struttura appare più piccola o più grande.

**Shear (0 → 60)**  
Per shear piccoli e medi la DeformableCNN mantiene un vantaggio, ma per shear elevati degrada più rapidamente e può incrociarsi con la StandardCNN. Questo suggerisce che la deformabilità locale aiuti fino a un certo punto, ma non sostituisca un training esplicito su trasformazioni molto estreme.

**Dataset FashionMNIST**
![[Pasted image 20260414185027.png]]

**Rotazione**  
La DeformableCNN è quasi sempre superiore fino a rotazioni molto elevate:
- 30°: **37.0% vs 26.3%**
- 90°: **6.3% vs 4.9%**
- 150°: **31.2% vs 18.7%**

A 180° i risultati diventano molto simili. Anche qui, oltre una certa soglia la trasformazione altera troppo la distribuzione naturale del dataset.

**Scala**  
La DeformableCNN mantiene un vantaggio consistente lungo tutto l’intervallo:
- Scale 0.7: **56.9% vs 35.3%**
- Scale 1.5: **53.9% vs 45.5%**

Rispetto a MNIST, il guadagno è meno drammatico ma più uniforme.

**Shear**  
Il vantaggio è evidente per shear piccoli, poi le curve tendono a convergere e degradare insieme. Oltre una certa soglia, la sola deformabilità non basta più e diventa importante anche l’augmentation esplicita.

#### Esperimento 4

Questo esperimento è stato introdotto per soddisfare un obiettivo diverso dai primi tre: non misurare soltanto *quanto* la DeformableCNN performi meglio nel setup considerato, ma anche spiegare in modo visivo **che cosa fa** il meccanismo deformabile e come differisce da una pipeline standard.

L’idea è usare **toy examples** piccoli e controllati, nei quali la lettura della figura sia immediata. In questo modo il comportamento dei layer non resta implicito nella formula, ma diventa osservabile direttamente.

##### Toy visualizations

##### Convoluzione standard
![[Pasted image 20260415100346.png]]

Nel toy example della convoluzione standard si vede una griglia regolare **3×3** applicata a una piccola feature map. Il punto importante è che il pattern di campionamento è **fisso**: la convoluzione legge sempre gli stessi punti relativi rispetto al centro, indipendentemente dalla geometria locale del segnale.

Questa è la caratteristica fondamentale della convoluzione classica: la traslazione del kernel nello spazio è regolare, ma la forma del supporto di campionamento non cambia mai. Se una struttura è allineata con il filtro, la risposta può essere elevata; se invece la struttura è curvata, inclinata o leggermente spostata rispetto alla griglia, il kernel continua comunque a campionare gli stessi punti.

##### Convoluzione deformabile
![[Pasted image 20260415100407.png]]

Nel toy example della convoluzione deformabile la griglia di campionamento non resta più regolare: i punti vengono spostati da **offset appresi**, e si vede visivamente che il pattern si piega verso la struttura rilevante della feature map.

Questo è il punto concettuale centrale del paper: la rete non modifica i pesi del filtro in funzione della posizione, ma modifica **dove** il filtro va a leggere l’informazione. In altre parole, la deformable convolution rende adattivo il **supporto spaziale** del kernel.

Nel toy example il messaggio non è che la risposta numerica debba essere sempre “più grande” in assoluto, ma che la risposta cambia perché il campionamento non è più vincolato alla griglia fissa. Quindi il meccanismo deformabile introduce un grado di libertà spaziale che la convoluzione standard non possiede.

##### Pooling
![[Pasted image 20260415100430.png|450]]

La visualizzazione del pooling chiarisce due aspetti diversi della pipeline.

Nel caso di **MaxPool 2×2**, la rete mantiene, in ciascuna finestra locale, il valore massimo. Questo equivale a una selezione competitiva dell’attivazione più forte e produce una riduzione di risoluzione mantenendo i segnali più salienti.

Nel caso di **AdaptiveAvgPool(1)**, invece, l’intera feature map viene compressa in un unico valore medio per canale. Questa operazione perde quasi completamente l’informazione spaziale fine, ma conserva una sintesi globale della presenza della feature. È proprio questa compressione finale che consente alla rete di passare da mappe spaziali a una rappresentazione adatta alla classificazione.

##### Interpolazione bilineare
![[Pasted image 20260415100506.png]]

La figura sull’interpolazione bilineare è particolarmente importante perché spiega un dettaglio tecnico spesso lasciato implicito: quando i punti di campionamento deformati cadono in coordinate **non intere**, il valore letto non coincide con un singolo pixel, ma con una combinazione pesata dei quattro vicini.

Nel toy example si vede un punto di campionamento frazionario e i relativi pesi assegnati ai pixel adiacenti. Questo rende chiaro perché la deformable convolution possa “spostarsi” nello spazio in modo continuo e non soltanto a salti discreti: l’offset appreso non seleziona un nuovo pixel intero, ma modifica il punto di lettura in modo continuo tramite interpolazione.

##### Interpretazione dell’esperimento 4

Questo esperimento rafforza la lettura degli esperimenti precedenti. Gli offset visualizzati nell’esperimento 2 non sono soltanto vettori astratti, ma la traduzione operativa di un cambio di campionamento; i risultati di robustezza dell’esperimento 3, in particolare su variazioni di scala, diventano più comprensibili proprio alla luce di questi toy examples.

In sintesi, le toy visualizations mostrano che:
- la **convoluzione standard** usa una griglia rigida
- la **convoluzione deformabile** sposta i punti di campionamento verso regioni più informative
- il **pooling** riduce la dimensionalità mantenendo informazione saliente o media globale
- l’**interpolazione bilineare** rende possibile il campionamento deformabile in coordinate non intere

Questa parte non dimostra da sola un vantaggio quantitativo, ma aggiunge una spiegazione visiva molto utile del perché il modello deformabile possa adattarsi meglio a strutture locali non perfettamente allineate.

#### Esperimento 5

Dopo aver visualizzato gli offset appresi e aver introdotto toy examples dei layer, è utile osservare direttamente anche la risposta interna della rete. Per questo è stato aggiunto un quinto esperimento che confronta, in alcuni stadi selezionati, le **feature maps** generate dalla **StandardCNN** e dalla **DeformableCNN**.

Dal punto di vista metodologico, i canali mostrati nelle figure non sono scelti manualmente “a occhio”, ma selezionati automaticamente tra quelli con maggiore attivazione media assoluta. Questa scelta non garantisce che il filtro sia semanticamente interpretabile in senso forte, ma riduce il rischio di selezionare mappe poco informative e rende il confronto più sistematico.

L’idea non è confrontare tutte le feature map possibili, ma selezionare alcuni **canali rilevanti** e metterli a confronto a parità di input. Nel codice, i canali mostrati vengono scelti automaticamente tra quelli con risposta media assoluta più elevata su un piccolo support set. Questo non fornisce un’interpretazione semantica completa del singolo filtro, ma permette di concentrare l’analisi su attivazioni effettivamente informative.

Per **MNIST** vengono mostrati esempi di classi come **1, 0, 9 e 6**, mentre per **FashionMNIST** si usano classi come **trouser, sandal, sneaker e bag**, cioè oggetti con geometrie e contorni significativamente diversi.

##### MNIST - stage 1, stage 2, stage 3

![[Pasted image 20260415100610.png]]
Nel **primo stadio** le differenze tra standard e deformabile sono già visibili, ma restano ancora fortemente legate alla geometria elementare dello stroke. Per il digit **“1”**, ad esempio, diverse feature deformabili appaiono più concentrate lungo la struttura verticale dominante; per digit come **“0”**, **“6”** e **“9”**, alcune mappe deformabili sembrano seguire in modo più aderente curve, anelli e regioni di cambiamento della forma. Le mappe standard, invece, risultano spesso più rigide o più diffuse rispetto alla struttura della cifra.

![[Pasted image 20260415100625.png]]
Nel **secondo stadio** il confronto diventa più interessante: le feature map sono meno “pixel-like” e più orientate a pattern di forma. In questa fase le attivazioni della rete deformabile appaiono spesso più selettive, con regioni luminose maggiormente concentrate sulle parti effettivamente descrittive della cifra. Parallelamente, le heatmap della magnitudine degli offset mostrano che l’attenzione deformabile si concentra nelle zone occupate dal digit e molto meno sullo sfondo, coerentemente con l’idea che il receptive field venga adattato dove il segnale è realmente presente.

![[Pasted image 20260415100632.png]]
Nel **terzo stadio** la risoluzione spaziale è ormai molto ridotta e le mappe diventano inevitabilmente più astratte e difficili da interpretare in modo locale. Tuttavia, anche a questo livello si osserva che la rete deformabile continua a produrre attivazioni non uniformi e offset con distribuzione non banale. Questo suggerisce che la deformabilità non agisca solo come correzione superficiale dei primi layer, ma continui a influenzare la rappresentazione anche negli strati più profondi, dove l’informazione è più semantica che geometrica.

Nel complesso, le figure di MNIST sono coerenti con l’idea che la deformable convolution favorisca una migliore aderenza della risposta interna alla geometria dei digit, soprattutto quando la forma contiene stroke curvi o variazioni locali di orientamento.

##### FashionMNIST - stage 1, stage 2, stage 3

Su **FashionMNIST** l’interpretazione è ancora più interessante, perché gli oggetti hanno struttura più complessa dei digit e includono bordi, contorni, parti allungate e regioni con silhouette differenti.

![[Pasted image 20260415100722.png]]
Nel **primo stadio** la differenza più evidente riguarda la capacità di seguire i **contorni principali**. Nel caso del **trouser**, diverse feature deformabili sembrano allinearsi alle due bande verticali della forma; per **sandal** e **sneaker** si osservano attivazioni che enfatizzano la sagoma orizzontale o il profilo della calzatura; nel caso della **bag** emergono risposte concentrate sul bordo rettangolare e sulle discontinuità della silhouette. Le feature standard, pur catturando anch’esse parti rilevanti, appaiono in più casi più diffuse o meno adattate al profilo dell’oggetto.

![[Pasted image 20260415100733.png]]
Nel **secondo stadio** le mappe standard e deformabili mostrano differenze ancora più nette. Le attivazioni della DeformableCNN risultano spesso più localizzate su porzioni strutturalmente rilevanti dell’oggetto, mentre la heatmap degli offset indica che gli spostamenti medi tendono a concentrarsi nelle zone di transizione della forma, cioè dove il contorno e l’organizzazione spaziale dell’item portano più informazione discriminativa.

![[Pasted image 20260415100744.png]]
Nel **terzo stadio** la risoluzione ridotta rende le mappe più grossolane, ma il confronto resta utile: le feature deformabili continuano a mostrare risposte dipendenti dalla classe e non semplicemente uniformi o diffuse. In particolare, per classi con profilo molto diverso tra loro, come **bag** e **sneaker**, la rete deformabile conserva attivazioni che sembrano rispettare meglio la distribuzione globale della forma.

##### Interpretazione dell’esperimento 5

Questo quinto esperimento aggiunge un livello di evidenza qualitativa diverso rispetto all’esperimento 2. La visualizzazione degli offset mostra **come** cambia il campionamento; il confronto delle feature responses mostra invece **che effetto ha** questo cambiamento sulla rappresentazione interna.

La lettura più prudente e corretta è la seguente:
- la **StandardCNN** costruisce feature map coerenti con una griglia di campionamento rigida
- la **DeformableCNN** produce spesso attivazioni più aderenti a stroke, bordi, contorni e parti strutturali dell’oggetto
- l’effetto è più leggibile nei livelli intermedi, dove la rappresentazione è abbastanza astratta da essere informativa, ma non ancora troppo compressa da perdere leggibilità visiva
- le heatmap della magnitudine degli offset rafforzano l’idea che il modello deformabile concentri la deformazione soprattutto nelle regioni informative e non nello sfondo

Questa evidenza resta **qualitativa**: non dimostra da sola una causalità diretta tra singolo offset e singola decisione finale. Tuttavia è coerente sia con i risultati dell’esperimento 1, in cui la DeformableCNN ottiene accuracy superiore, sia con l’esperimento 3, in cui il vantaggio emerge soprattutto in presenza di variazioni geometriche.

### Conclusione sul progetto

Il progetto realizza una pipeline completa, riproducibile e CPU-oriented che:
- implementa una **DeformableCNN** con tracciamento esplicito degli offset
- automatizza training, valutazione e salvataggio degli artefatti
- produce figure qualitative e heatmap per documentare il comportamento del meccanismo deformabile
- misura in modo controllato la robustezza a trasformazioni geometriche
- aggiunge toy examples esplicativi per convoluzione standard, convoluzione deformabile, pooling e interpolazione bilineare
- confronta direttamente le feature responses interne di layer standard e deformabili
- unifica l’intero workflow tramite `demo.py`

Con gli esperimenti 4 e 5 gli offset non vengono più mostrati soltanto come vettori o heatmap, ma vengono inseriti dentro una lettura più completa: prima si chiarisce visivamente il funzionamento dei layer su esempi sintetici, poi si osserva come quel meccanismo si rifletta nelle attivazioni reali dei modelli su input di test.

Il trade-off sperimentale più importante emerso resta l’**overhead computazionale su CPU** della deformable convolution: i guadagni in performance e robustezza osservati nel setup considerato sono accompagnati da tempi di training e inferenza sensibilmente superiori rispetto al baseline standard.



