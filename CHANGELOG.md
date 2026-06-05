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
