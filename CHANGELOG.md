# Changelog

## 2026-06-05 - Branch-Cleanup und Repository-Konsolidierung

### Ausgangszustand
- Alle Remotes und Tags wurden mit `git fetch --all --prune --tags` aktualisiert.
- Der lokale Worktree war nicht sauber und wurde vor den Merges gesichert: `pre-branch-cleanup-2026-06-05`.
- `origin/main` ist der Default-Branch, `origin/dev` enthielt aber den Großteil der aktuellen Entwicklungsarbeit.
- `origin/main` und `origin/dev` waren divergiert; `origin/main` enthielt einen Revert von altem `SimplePPDB`-Demo-Code, `origin/dev` enthielt neuere Pipeline- und Trainingsarbeit.

### Geplanter Ablauf
- `main` in `dev` integrieren und den erwarteten Konflikt in `src/main.py` aufloesen.
- Noch offene Branches in `dev` integrieren: `feature/asset-sari-trained-model-validation`, `fix-copy-rate`, `t5-pipeline`.
- Verfuegbare Checks ausfuehren.
- `dev` nach `main` mergen.
- Vollstaendig integrierte lokale und Remote-Branches loeschen.

### Durchgefuehrt
- `origin/main` wurde in `dev` gemergt.
- Der Konflikt in `src/main.py` wurde zugunsten des aktuellen `dev`-Pipeline-Codes geloest; der alte `SimplePPDB`-Demo-Code aus `main` bleibt entfernt.
- `origin/feature/asset-sari-trained-model-validation` wurde in `dev` gemergt.
- `origin/fix-copy-rate` wurde in `dev` gemergt.
- Der Konflikt in `src/main.py` wurde zugunsten der neueren `TrainingPipeline`-Einstiegslogik geloest.
- `origin/t5-pipeline` wurde in `dev` gemergt.
- Die von `t5-pipeline` bereits getrackten Run-Artefakte unter `runs/` wurden mit uebernommen.
- Nach den Merges wurden Ruff-Lint-Probleme aus den integrierten Branches behoben.
- Das T5-Skript nutzt jetzt die aktuelle `storage`-Schicht statt der entfernten `evaluation.file_writer`-Datei.
- Checks: `uv run pytest` bestanden, `uv run ruff check .` bestanden.
- `dev` wurde nach `origin/dev` gepusht.
- `dev` wurde in `main` gemergt und nach `origin/main` gepusht.
- Geloeschte Remote-Branches: `apply-BLEU`, `apply-rouge`, `f1-metric`, `feature/asset-sari-trained-model-validation`, `feature/sari-asset-pipeline`, `feature/sari-metric`, `fix-copy-rate`, `fleschkincaid-metric`, `import-OneStopEnglishCorpus`, `import-asset`, `import-simple_ppdb`, `import-wikilarge`, `import-wikismall`, `metric_BERTScore`, `pipeline_wiki_large`, `revert-8-apply-BLEU`, `t5-pipeline`, `train-pipeline-OneStopEnglishCorpus`.
- Geloeschte lokale Branches: `feature/asset-sari-trained-model-validation`, `feature/sari-asset-pipeline`, `feature/sari-metric`, `import-asset`.
