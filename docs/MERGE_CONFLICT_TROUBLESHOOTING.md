# Merge Conflict Troubleshooting

If a PR UI still says "This branch has conflicts" after you edited files, use this checklist.

## 1) Verify no conflict markers in working tree

```bash
make check-conflicts
```

This scans for:
- `<<<<<<<`
- `=======`
- `>>>>>>>`

## 2) Ensure you resolved against the latest target branch

A PR can still report conflicts if your branch was resolved against an older base.

Typical workflow:

```bash
git fetch origin
git checkout <your-branch>
git merge origin/main
# resolve files
git add <resolved files>
git commit
```

## 3) Re-run checks after resolving

```bash
make check-conflicts
make check-structure
make build-check
```

## 4) If conflict still appears in UI

- Confirm you pushed the latest commit to the same PR branch.
- Refresh PR page and compare latest commit SHA.
- Ensure no other file still contains markers.

## Note for this repository

Locally, conflict markers are blocked by `scripts/check_conflicts.sh` and by `make check-structure` dependency.
