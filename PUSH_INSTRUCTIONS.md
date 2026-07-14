# Instructions — push code to the same GitHub repo

```bash
# 1. Go to the repo (NOT the worktree, NOT the reproduction folder)
cd /Users/moahaimentalib/Desktop/f_flex_network_code_clean

# 2. Confirm the remote is the correct repo (should print the moahaimen URL)
git remote -v
#   origin  https://github.com/moahaimen/f_flex_network_code.git  (fetch/push)

# 3. See what changed and which branch you're on
git status
git branch --show-current      # the final-report code lives on: hardening/periodic-keepmask-inf

# 4. Stage + commit any new/changed CODE (do NOT try to add checkpoints/data — see note)
git add -A
git commit -m "<clear message describing the change>"

# 5. Push the current branch to the SAME repo (origin)
git push origin HEAD
#   (first push of a new branch: git push -u origin <branch-name>)
```

## Auth

- This Mac uses the `osxkeychain` credential helper with a cached GitHub token, so `git push` works non-interactively. If it ever prompts or returns 403/401:

  ```bash
  gh auth login          # or: set up a Personal Access Token
  ```

  Never paste a password/token into a script or commit it.

## Critical caveat (why a plain clone won't run)

- `.gitignore` line `results/*` excludes the trained checkpoints (`.pt`), traffic matrices (`.npz`), topology files, and result caches. `git push` will not upload them — only the code goes to GitHub.
- So GitHub holds the code; the runnable checkpoints+data are in the separate `FINAL_METHOD_REAL_CODE_RERUN.zip`. If a specific large artifact is needed on GitHub, force-add it individually and keep it under 100 MB:

  ```bash
  git add -f path/to/needed_file        # bypass .gitignore for one file
  ```

  GitHub rejects any single file larger than 100 MB.

## Current state (already done)

- Branch `hardening/periodic-keepmask-inf` = the final report code, already pushed:
  `https://github.com/moahaimen/f_flex_network_code/tree/hardening/periodic-keepmask-inf`
- `main` is unchanged at `15ad7fb`. To publish this as the main line, open a PR:

  ```bash
  gh pr create --base main --head hardening/periodic-keepmask-inf --fill
  ```
