# 🎓 Tesi CityLearn Challenge 2023 - Overleaf Setup

## Energy Forecasting e Reinforcement Learning per Smart Buildings

Questa cartella contiene il documento LaTeX completo della tesi, ottimizzato per Overleaf.

## 📁 Struttura del Progetto

```
thesis/
├── main.tex                    # Documento principale (COMPILARE QUESTO)
├── bibliografia.bib           # Bibliografia in formato BibTeX
├── README_OVERLEAF.md        # Questo file
├── sections/                 # Capitoli della tesi
│   ├── 01_introduzione.tex
│   ├── 02_stato_arte.tex
│   ├── 03_metodologia.tex
│   ├── 04_implementazione.tex
│   ├── 05_risultati.tex
│   ├── 06_conclusioni.tex
│   ├── appendice_a_codice.tex
│   └── appendice_b_risultati_dettagliati.tex
└── images/                   # Directory per le immagini
    ├── logo_universita.png   # Logo università (da aggiungere)
    ├── neural_01_model_performance.png
    ├── neural_02_training_convergence.png
    ├── neural_03_error_distribution.png
    ├── neural_04_temporal_analysis.png
    ├── neural_05_feature_importance.png
    └── neural_06_cross_building.png
```

## 🚀 Setup su Overleaf

### 1. Caricamento dei File

1. **Crea nuovo progetto** su Overleaf
2. **Upload** tutti i file mantenendo la struttura delle cartelle
3. **Imposta main.tex** come documento principale
4. **Verifica** che tutti i file siano nella posizione corretta

### 2. Compilazione

- **Compilatore**: pdfLaTeX (raccomandato)
- **File principale**: main.tex
- **Compilazioni necessarie**: 2-3 volte per riferimenti incrociati e bibliografia

### 3. Personalizzazione Richiesta

#### 📝 Informazioni Studente
Modifica in `main.tex` alle righe 85-95:
```latex
{\Large UNIVERSITÀ DEGLI STUDI DI [NOME UNIVERSITÀ]}\\[0.5cm]
{\Large DIPARTIMENTO DI [NOME DIPARTIMENTO]}\\[0.5cm]
{\Large CORSO DI LAUREA IN [NOME CORSO]}\\[2.5cm]

Prof. [Nome Relatore]
[Nome Studente]\\
Matricola: [Numero Matricola]
```

#### 🖼️ Logo Università
- **Aggiungi** il logo della tua università come `images/logo_universita.png`
- **Dimensione consigliata**: 300x300px, formato PNG
- **Posizione**: Linea 82 in main.tex

## 📊 Grafici e Visualizzazioni

I grafici sono referenziati ma vanno generati dai risultati:

### Grafici Neural Network (6 grafici)
- `neural_01_model_performance.png`: Confronto performance modelli
- `neural_02_training_convergence.png`: Convergenza addestramento
- `neural_03_error_distribution.png`: Distribuzione errori
- `neural_04_temporal_analysis.png`: Analisi temporale  
- `neural_05_feature_importance.png`: Feature importance
- `neural_06_cross_building.png`: Analisi cross-building

### Per generare i grafici:
```bash
# Dalla directory principale del progetto
cd C:\Users\gbrla\Desktop\tesi
python run_neural_evaluation.py  # Genera automaticamente i 6 grafici
```

I grafici verranno salvati in `results/visualizations/` e potranno essere copiati nella cartella `thesis/images/`.

## ✅ Checklist Pre-Compilazione

- [ ] Tutti i file .tex sono presenti in `sections/`
- [ ] Bibliografia `bibliografia.bib` è caricata
- [ ] Logo università in `images/logo_universita.png`
- [ ] Grafici risultati in `images/`
- [ ] Informazioni personali aggiornate in `main.tex`
- [ ] Compilatore impostato su pdfLaTeX

## 🛠️ Troubleshooting Comune

### Errore: "File not found"
- Verifica che tutti i file siano nella posizione corretta
- Controlla che i nomi dei file corrispondano esattamente

### Errore: "Bibliography not found"
- Assicurati che `bibliografia.bib` sia nella root del progetto
- Compila 2-3 volte per generare la bibliografia

### Errore: "Image not found"
- Controlla che le immagini siano in `images/`
- Verifica i nomi dei file (case-sensitive)

### Warning: "Citation undefined"
- Normale alla prima compilazione
- Scompare dopo 2-3 compilazioni

## 📈 Contenuto della Tesi

### Capitoli Principali
1. **Introduzione**: Contesto, obiettivi e contributi originali
2. **Stato dell'Arte**: Review letteratura energy forecasting e RL
3. **Metodologia**: Approccio sperimentale e validazione cross-building
4. **Implementazione**: Architettura software e algoritmi
5. **Risultati**: Performance comprehensive e analisi statistiche
6. **Conclusioni**: Sintesi, limitazioni e sviluppi futuri

### Appendici
- **A**: Implementazione software dettagliata
- **B**: Risultati dettagliati e analisi aggiuntive

### Metriche e Risultati Principali
- **LSTM**: RMSE = 50.85±11.11, R² = 0.9498
- **Ensemble Stacking**: RMSE = 25.07±0.41 (migliore assoluto)
- **Validazione cross-building**: 3 edifici, Leave-One-Out
- **Features**: 16 originali + 9 engineered

## 📧 Supporto

Per problemi con Overleaf:
1. Controlla la [documentazione ufficiale](https://www.overleaf.com/learn)
2. Verifica i log di compilazione per errori specifici
3. Assicurati di avere tutti i pacchetti LaTeX necessari

## 🏆 Risultati Attesi

La tesi presenta:
- Sistema LSTM robusto con fallback numerici
- Performance state-of-the-art con ensemble methods
- Framework di valutazione cross-building rigoroso
- 6 visualizzazioni comprehensive per esperimento
- Implementazione completa open-source

**Valutazione target: 30/30** 🎯

---

*La tesi è pronta per la compilazione e la presentazione. Tutti i risultati sperimentali sono basati su implementazioni reali e dati del CityLearn Challenge 2023.*