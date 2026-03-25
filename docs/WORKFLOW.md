# Practical Workflow Commands

Use these commands to operate the scaffold quickly.

## 1) Validate scaffold integrity
```bash
make check-structure
```

## 2) Create today's meeting note from template
```bash
make new-meeting
```

Or with explicit date:
```bash
./scripts/new_meeting_note.sh 2026-03-25
```

## 3) Create a report draft from outline template
```bash
make new-report-draft VERSION=v1
```

This copies `docs/report/report_outline.md` to `docs/report/report_draft_v1.md`.
